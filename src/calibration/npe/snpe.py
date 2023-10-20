import os
import abc
import time
from abc import ABC
from pathlib import Path
from copy import deepcopy
from warnings import warn

# https://stackoverflow.com/questions/72852/how-to-do-relative-imports-in-python (2nd answer)
from sbi.inference import (
    DirectPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from sbi.inference.posteriors.base_posterior import NeuralPosterior
import numpy as np
import torch
import torchsort
from functorch import vmap
from hypothesis.util.data import NumpyDataset
from torch.nn import functional as F
from torch import nn, ones, Tensor
from torch.optim import Adam
from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_
from sbi.utils import (
    check_estimator_arg,
    check_prior,
    del_entries,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
    x_shape_from_simulation,
)
from sbi.utils.sbiutils import ImproperEmpirical, mask_sims_from_prior
from sbi.utils.get_nn_models import posterior_nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from sbi.utils.sbiutils import get_simulations_since_round
from typing import Any, Callable, Dict, Optional, Tuple, Union
from torch.distributions import Distribution
from sbi.utils.torchutils import check_if_prior_on_device, process_device

from workflows.calnpe.utils import load_checkpoint


class STEFunctionRanksq(torch.autograd.Function):
    """
    Implementation of StraightThroughEstimator.
    """

    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class STEFunctionRankslogq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.where(input > 0, 0, float("-inf"))

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


def get_logq_for_ranks(model, x, y, prior, n_samples):
    embedded_y = model._embedding_net(y)
    if x.shape[-1] == 1:
        sample_shape = x.shape
    else:
        sample_shape = x.shape[:1]
    _noise, _logabsdet = model._transform(x, context=embedded_y)
    noises = [_noise]
    logabsdets = [_logabsdet]
    for _ in range(n_samples):
        _noise, _logabsdet = model._transform(
            prior.sample(sample_shape), context=embedded_y
        )
        noises.append(_noise)
        logabsdets.append(_logabsdet)
    return vmap(model._distribution.log_prob, in_dims=(1, None), out_dims=1)(
        torch.stack(noises, dim=1), embedded_y
    ) + torch.stack(logabsdets, dim=1)


def get_ranks(model, x, y, prior, n_samples, logits=False):
    logq = get_logq_for_ranks(model, x, y, prior, n_samples)
    if logits:
        return (
            (
                logq[:, 1:]
                + STEFunctionRankslogq.apply(
                    logq[:, 0].unsqueeze(1) - logq[:, 1:]
                )
            ).logsumexp(dim=1)
            - logq[:, 1:].logsumexp(dim=1)
        ).exp()
    else:
        q = logq.exp()
        return (
            q[:, 1:] * STEFunctionRanksq.apply(q[:, 0].unsqueeze(1) - q[:, 1:])
        ).sum(dim=1) / q[:, 1:].sum(dim=1)


def get_coverage(ranks, device):
    # Source: https://github.com/montefiore-ai/balanced-nre/blob/main/demo.ipynb
    # As a sample at a given rank belongs to the credible regions at levels 1-rank and below,
    # the coverage at level 1-alpha is the proportion of samples with ranks alpha and above.
    ranks = ranks[~ranks.isnan()]
    alpha = torchsort.soft_sort(ranks.unsqueeze(0)).squeeze()
    return (
        torch.linspace(0.0, 1.0, len(alpha) + 2, device=device)[1:-1],
        1 - torch.flip(alpha, dims=(0,)),
    )


def get_calibration_error(
    model,
    x,
    y,
    device,
    prior,
    n_samples,
    calibration=0,
    logits=False,
    instances_subsample=1,
):
    if instances_subsample < 1:
        idxs = torch.randperm(x.shape[0], device=x.device)
        x = x[idxs[: int(x.shape[0] * instances_subsample)]]
        y = y[idxs[: int(y.shape[0] * instances_subsample)]]
    ranks = get_ranks(
        model,
        x,
        y,
        prior,
        logits=logits,
        n_samples=n_samples,
    )
    coverage, expected = get_coverage(ranks, device)
    if calibration == 0:
        return torch.nn.functional.relu(expected - coverage).pow(2).mean()
        # default in BNRE paper, conservatiness regularizer
    elif calibration == 1:
        return (coverage - expected).pow(2).mean()  # calibration regularizer
    else:
        return (
            (
                (1 - calibration)
                * torch.nn.functional.relu(expected - coverage)
                + calibration * (coverage - expected)
            )
            .pow(2)
            .mean()
        )  # mixture of conservatiness and calibration regularizers


class CustomNeuralInference(ABC):

    """
    Reimplementation of class ``CustomNeuralInference`` from
    https://github.com/mackelab/sbi/blob/8fe45c9fccb086fe3f6b90ae82b0ca347f2e6457/sbi/inference/base.py
    for sake of not having validation DataLoader.

    Abstract base class for neural inference methods.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Base class for inference methods.
        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Must be a PyTorch
                distribution, see FAQ for details on how to use custom distributions.
            device: torch device on which to train the neural net and on which to
                perform all posterior operations, e.g. gpu or cpu.
            logging_level: Minimum severity of messages to log. One of the strings
               "INFO", "WARNING", "DEBUG", "ERROR" and "CRITICAL".
            summary_writer: A `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        self._device = process_device(device)
        check_prior(prior)
        check_if_prior_on_device(self._device, prior)
        self._prior = prior

        self._posterior = None
        self._neural_net = None
        self._x_shape = None

        self._show_progress_bars = show_progress_bars

        # Initialize roundwise (theta, x, prior_masks) for storage of parameters,
        # simulations and masks indicating if simulations came from prior.
        self._theta_roundwise = []
        self._x_roundwise = []
        self._prior_masks = []
        self._model_bank = []

        # Initialize list that indicates the round from which simulations were drawn.
        self._data_round_index = []

        self._round = 0
        self._val_log_prob = float("-Inf")

        # XXX We could instantiate here the Posterior for all children. Two problems:
        #     1. We must dispatch to right PotentialProvider for mcmc based on name
        #     2. `method_family` cannot be resolved only from `self.__class__.__name__`,
        #         since SRE, AALR demand different handling but are both in SRE class.

        # Logging during training (by SummaryWriter).
        self._summary = dict(
            epochs_trained=[],
            best_validation_log_prob=[],
            validation_log_probs=[],
            training_log_probs=[],
            epoch_durations_sec=[],
        )

    def get_simulations(
        self,
        starting_round: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Returns all $\theta$, $x$, and prior_masks from rounds >= `starting_round`.
        If requested, do not return invalid data.
        Args:
            starting_round: The earliest round to return samples from (we start counting
                from zero).
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training.
            warn_on_invalid: Whether to give out a warning if invalid simulations were
                found.
        Returns: Parameters, simulation outputs, prior masks.
        """

        theta = get_simulations_since_round(
            self._theta_roundwise, self._data_round_index, starting_round
        )
        x = get_simulations_since_round(
            self._x_roundwise, self._data_round_index, starting_round
        )
        prior_masks = get_simulations_since_round(
            self._prior_masks, self._data_round_index, starting_round
        )

        return theta, x, prior_masks

    @abc.abstractmethod
    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
    ) -> NeuralPosterior:
        raise NotImplementedError

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        validation_inputs: Union[NumpyDataset, data.TensorDataset] = None,
        validation_outputs: Union[NumpyDataset, data.TensorDataset] = None,
        resume_training: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders for training and validation.
        Args:
            dataset: holding all theta and x, optionally masks.
            training_batch_size: training arg of inference methods.
            validation_inputs: Theta values of validation data.
            validation_outputs: Simulation outputs for the validation data.
            resume_training: Whether the current call is resuming training so that no
                new training and validation indices into the dataset have to be created.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).
        Returns:
            Tuple of dataloaders for training and validation.
        """
        theta, x, prior_masks = self.get_simulations(starting_round)

        dataset = data.TensorDataset(theta, x, prior_masks)

        # Get total number of training examples.
        num_examples = theta.size(0)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        if validation_fraction != 0:
            num_validation_examples = num_examples - num_training_examples
        else:
            assert type(validation_inputs) == type(validation_outputs)
            if isinstance(validation_inputs, NumpyDataset):
                val_inputs_dataset = data.TensorDataset(
                    torch.from_numpy(validation_inputs.storage.data),
                    torch.from_numpy(validation_outputs.storage.data),
                )
            elif isinstance(validation_inputs, data.TensorDataset):
                val_inputs_dataset = data.TensorDataset(
                    validation_inputs.tensors[0], validation_outputs.tensors[0]
                )
            else:
                raise ValueError(f"Unexpected type {type(validation_inputs)}")
            num_validation_examples = len(validation_inputs)
            print(f"num_validation_examples: {num_validation_examples}")

        if not resume_training:
            # Separate indices for training and validation
            permuted_indices = torch.randperm(num_examples)
            self.train_indices, self.val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        if validation_fraction != 0:
            val_loader_kwargs = {
                "batch_size": min(
                    training_batch_size, num_validation_examples
                ),
                "shuffle": False,
                "drop_last": True,
                "sampler": SubsetRandomSampler(self.val_indices.tolist()),
            }
        else:
            val_loader_kwargs = {
                "batch_size": min(
                    training_batch_size, num_validation_examples
                ),
                "shuffle": False,
                "drop_last": True,
            }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(
                train_loader_kwargs, **dataloader_kwargs
            )
            if validation_fraction != 0:
                val_loader_kwargs = dict(
                    val_loader_kwargs, **dataloader_kwargs
                )
            elif (self.val_inputs is not None) and (len(self.val_inputs) > 0):
                val_loader_kwargs = dict(
                    val_loader_kwargs, **dataloader_kwargs
                )
            else:
                raise NotImplementedError

        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        if validation_fraction != 0:
            val_loader = data.DataLoader(dataset, **val_loader_kwargs)
            return train_loader, val_loader
        elif self.val_inputs is not None:
            val_loader = data.DataLoader(
                val_inputs_dataset, **val_loader_kwargs
            )
            return train_loader, val_loader
        else:
            return train_loader

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.
        Checks for improvement in validation performance over previous epochs.
        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.
        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        assert self._neural_net is not None
        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            neural_net.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def _default_summary_writer(self) -> SummaryWriter:
        """Return summary writer logging to method- and simulator-specific directory."""

        method = self.__class__.__name__
        logdir = Path(
            get_log_root(),
            method,
            datetime.now().isoformat().replace(":", "_"),
        )
        return SummaryWriter(logdir)

    @staticmethod
    def _describe_round(round_: int, summary: Dict[str, list]) -> str:
        epochs = summary["epochs_trained"][-1]
        best_validation_log_prob = summary["best_validation_log_prob"][-1]

        description = f"""
        -------------------------
        ||||| ROUND {round_ + 1} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_log_prob:.4f}
        -------------------------
        """

        return description

    @staticmethod
    def _maybe_show_progress(show: bool, epoch: int) -> None:
        if show:
            # end="\r" deletes the print statement when a new one appears.
            # https://stackoverflow.com/questions/3419984/. `\r` in the beginning due
            # to #330.
            print(
                "\r",
                f"Training neural network. Epochs trained: {epoch}",
                end="",
            )

    def _report_convergence_at_end(
        self, epoch: int, stop_after_epochs: int, max_num_epochs: int
    ) -> None:
        if self._converged(epoch, stop_after_epochs):
            print(
                "\r",
                f"Neural network successfully converged after {epoch} epochs.",
                end="",
            )
        elif max_num_epochs == epoch:
            warn(
                "Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
                "but network has not yet fully converged. Consider increasing it."
            )

    def _summarize(
        self,
        round_: int,
    ) -> None:
        """Update the summary_writer with statistics for a given round.
        During training several performance statistics are added to the summary, e.g.,
        using `self._summary['key'].append(value)`. This function writes these values
        into summary writer object.
        Args:
            round: index of round
        Scalar tags:
            - epochs_trained:
                number of epochs trained
            - best_validation_log_prob:
                best validation log prob (for each round).
            - validation_log_probs:
                validation log probs for every epoch (for each round).
            - training_log_probs
                training log probs for every epoch (for each round).
            - epoch_durations_sec
                epoch duration for every epoch (for each round)
        """

        # Add most recent training stats to summary writer.
        self._summary_writer.add_scalar(
            tag="epochs_trained",
            scalar_value=self._summary["epochs_trained"][-1],
            global_step=round_ + 1,
        )
        """
        self._summary_writer.add_scalar(
            tag="best_validation_log_prob",
            scalar_value=self._summary["best_validation_log_prob"][-1],
            global_step=round_ + 1,
        )
        """

        # Add validation log prob for every epoch.
        # Offset with all previous epochs.
        offset = (
            torch.tensor(self._summary["epochs_trained"][:-1], dtype=torch.int)
            .sum()
            .item()
        )
        for i, vlp in enumerate(
            self._summary["validation_log_probs"][offset:]
        ):
            self._summary_writer.add_scalar(
                tag="validation_log_probs",
                scalar_value=vlp,
                global_step=offset + i,
            )

        for i, tlp in enumerate(self._summary["training_log_probs"][offset:]):
            self._summary_writer.add_scalar(
                tag="training_log_probs",
                scalar_value=tlp,
                global_step=offset + i,
            )

        for i, eds in enumerate(self._summary["epoch_durations_sec"][offset:]):
            self._summary_writer.add_scalar(
                tag="epoch_durations_sec",
                scalar_value=eds,
                global_step=offset + i,
            )

        self._summary_writer.flush()

    @property
    def summary(self):
        return self._summary

    def __getstate__(self) -> Dict:
        """Returns the state of the object that is supposed to be pickled.
        Attributes that can not be serialized are set to `None`.
        Returns:
            Dictionary containing the state.
        """
        warn(
            "When the inference object is pickled, the behaviour of the loaded object "
            "changes in the following two ways: "
            "1) `.train(..., retrain_from_scratch=True)` is not supported. "
            "2) When the loaded object calls the `.train()` method, it generates a new "
            "tensorboard summary writer (instead of appending to the current one)."
        )
        dict_to_save = {}
        unpicklable_attributes = ["_summary_writer", "_build_neural_net"]
        for key in self.__dict__.keys():
            if key in unpicklable_attributes:
                dict_to_save[key] = None
            else:
                dict_to_save[key] = self.__dict__[key]
        return dict_to_save

    def __setstate__(self, state_dict: Dict):
        """Sets the state when being loaded from pickle.
        Also creates a new summary writer (because the previous one was set to `None`
        during serializing, see `__get_state__()`).
        Args:
            state_dict: State to be restored.
        """
        state_dict["_summary_writer"] = self._default_summary_writer()
        self.__dict__ = state_dict


class CalibratedPosteriorEstimator(CustomNeuralInference, ABC):
    def __init__(
        self,
        prior: Union[Distribution, torch.nn.Module],
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        calibration: float = 0,
        num_samples: int = 100,
        gamma: float = 100,  # `lambda` in BNRE paper
        instances_subsample: float = 1,
        validation_fraction: float = 0.1,
        problem: str = "weinberg",
    ):
        """
        This class is a reimplementation (mostly copy-paste) of the original SNPE
        class
        (https://github.com/mackelab/sbi/blob/1d982748fa27576a3ab1e3a896e6af751ae48560/sbi/inference/snpe/snpe_base.py).

        Base class for Sequential Neural Posterior Estimation methods.
        Args:
            density_estimator: If it is a string, use a pre-configured network
                of the provided type (one of nsf, maf, mdn, made).
                Alternatively, a function that builds a custom neural network
                can be provided. The function will be called with the first
                batch of simulations (theta, x), which can thus be used for
                shape inference and potentially for z-scoring. It needs to
                return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            calibration: Float between 0 and 1. 0 means conservatiness
                regularizer, 1 calibration regularizer, number in between is
                mixture
            num_samples: Number of samples
            gamma: Lambda (strength of whole balancing regularizer)
            instances_subsample: Number of instances subsamples
            validation_fraction: The fraction of data to use for validation.
            problem: 'weinberg', 'slcp', 'mg1', 'lotka_volterra',
                'gw' or 'spatialsir'
        See docstring of `CustomNeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string
        # or a callable. The function creating the neural network is attached
        # to `_build_neural_net`. It will be called in the first round and
        # receive thetas and xs as inputs, so that they can be used for shape
        # inference and potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = posterior_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

        self._proposal_roundwise = []
        self.use_non_atomic_loss = False

        self.num_samples = num_samples
        self.calibration = calibration
        self.gamma = gamma
        self.instances_subsample = instances_subsample
        self.problem = problem

    def save_checkpoint(self, path="checkpoint.pt"):
        """
        Create model checkpoint, which e.g. contains state dicts of model
        and optimizer.

        :param dict state:
            State of the model, optimizer, and more, if desired
        :param str filename:
            File name of checkpoint
        """
        path += "/checkpoint.pt"
        assert (
            path.endswith(".pt")
            or path.endswith(".pth")
            or path.endswith(".pth.tar")
        ), "The checkpoint does not have a correct file ending!"
        print("--> Saving checkpoint")
        torch.save(
            {
                "state_dict": self._neural_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def append_simulations(
        self,
        theta: data.TensorDataset,
        x: data.TensorDataset,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> "PosteriorEstimator":
        r"""Store parameters and simulation outputs to use them for later
        training. Data are stored as entries in lists for each type of
        variable (parameter/data). Stores $\theta$, $x$, prior_masks
        (indicating if simulations are coming from the prior or not) and an
        index indicating which round the batch of simulations came from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            proposal: The distribution that the parameters $\theta$ were
                sampled from. Pass `None` if the parameters were sampled from
                the prior. If not `None`, it will trigger a different
                loss-function.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. For single-round SNPE, it is fine to discard invalid
                simulations, but for multi-round SNPE (atomic), discarding
                invalid simulations gives systematically wrong results. If
                `None`, it will be `True` in the first round and `False` in
                later rounds.
            data_device: Where to store the data, default is on the same
                device where the training is happening. If training a large
                dataset on a GPU with not much VRAM can set to 'cpu' to store
                data on system memory instead.
        Returns:
            CustomNeuralInference object (returned so that this function is
            chainable).
        """
        if (
            proposal is None
            or proposal is self._prior
            or (
                isinstance(proposal, RestrictedPrior)
                and proposal._prior is self._prior
            )
        ):
            # The `_data_round_index` will later be used to infer if one
            # should train with MLE loss or with atomic loss (see, in
            # `train()`: self._round = max(self._data_round_index))
            current_round = 0
        else:
            if not self._data_round_index:
                # This catches a pretty specific case: if, in the first round,
                # one passes data that does not come from the prior.
                current_round = 1
            else:
                current_round = max(self._data_round_index) + 1

        if exclude_invalid_x is None:
            if current_round == 0:
                exclude_invalid_x = True
            else:
                exclude_invalid_x = False

        x = x.tensors[0]
        theta = theta.tensors[0]
        # print(f'x and theta"s: {x}, {theta}')
        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )
        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        npe_msg_on_invalid_x(
            num_nans, num_infs, exclude_invalid_x, "Single-round NPE"
        )

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )
        self._check_proposal(proposal)

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(
            int(current_round > 0), theta.size(0)
        )

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._proposal_roundwise.append(proposal)

        if self._prior is None or isinstance(self._prior, ImproperEmpirical):
            if proposal is not None:
                raise ValueError(
                    "You had not passed a prior at initialization, but now "
                    "you passed a proposal. If you want to run multi-round "
                    "SNPE, you have to specify a prior (set the `.prior` "
                    "argument or re-initialize the object with a prior "
                    "distribution). If the samples you passed to "
                    "`append_simulations()` were sampled from the prior, "
                    "you can run single-round inference with "
                    "`append_simulations(..., proposal=None)`."
                )
            theta_prior = self.get_simulations()[0].to(self._device)
            self._prior = ImproperEmpirical(
                theta_prior, ones(theta_prior.shape[0], device=self._device)
            )

        return self

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        validation_inputs: Union[NumpyDataset, data.TensorDataset] = None,
        validation_outputs: Union[NumpyDataset, data.TensorDataset] = None,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
        calibration: float = 0,
        path_checkpoint=None,
        problem: str = "weinberg",
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p
        (\theta|x)$.
        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            validation_inputs: Theta values of validation data.
            validation_outputs: Simulation outputs for the validation data.
            stop_after_epochs: The number of epochs to wait for improvement on
                the validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we
                stop training even when the validation loss is still
                decreasing. Otherwise, we train until validation loss
                increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in
                order to prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect
                to the simulations `x`. See Lueckmann, Gonçalves et al.,
                NeurIPS 2017.
            resume_training: Can be used in case training time is limited,
                e.g. on a cluster. If `True`, the split between train and
                validation set, the optimizer, the number of epochs, and the
                best validation log-prob will be restored from the last time
                `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in
                round 1, i.e. from the prior. Training may be sped up by
                ignoring such less targeted samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and
                validation loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to
                the training and validation dataloaders (like, e.g., a
                collate_fn)
            calibration (float): Number between 0 and 1. For 0 calibration, a
                conservativeness regularizer is used, for 1 calibration a
                calibration regularizer. For number in between, a mixture of
                conservativeness and calibration regularizer is used.

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss, (
                "You have already trained this neural network. After you had "
                "trained the network, you again appended simulations with "
                "`append_simulations (theta, x)`, but you did not provide a "
                "proposal. If the new simulations are sampled from the prior, "
                "you can set `.train(..., force_first_round_loss=True`). "
                "However, if the new simulations were not sampled from the "
                "prior, you should pass the proposal, i.e. "
                "`append_simulations(theta, x, proposal)`. If your samples "
                "are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the "
                "result of SNPE will not be the true posterior. Instead, it "
                "will be the proposal posterior, which (usually) is more "
                "narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds
        # as of now. SNPE-A can, by construction of the algorithm, only use
        # samples from the last round. SNPE-A is the only algorithm that has
        # an attribute `_ran_final_round`, so this is how we check for whether
        # or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user.
        # For atomic SNPE, it does not matter what the proposal is. For
        # non-atomic SNPE, we only use the latest data that was passed, i.e.
        # the one from the last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            starting_round=start_idx,
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction,
            validation_inputs=validation_inputs,
            validation_outputs=validation_outputs,
            resume_training=resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network. This is passed into
        # NeuralPosterior, to create a neural posterior which can `sample()`
        # and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring
            # transforms)
            print(
                f"Theta and x shapes: {theta[self.train_indices].shape}, "
                f"{x[self.train_indices].shape}"
            )
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = Adam(
                list(self._neural_net.parameters()), lr=learning_rate
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        # Checkpoint loading if path provided.
        if path_checkpoint is not None:
            load_checkpoint(
                model=self._neural_net,
                checkpoint=torch.load(
                    path_checkpoint + "/checkpoint.pt",
                    map_location=torch.device("cpu"),
                ),
                load_cp_continue=True,
                optimizer=self.optimizer,
            )
            self._best_state_dict = torch.load(
                os.path.join(path_checkpoint, "checkpoint_best.pt")
            )

        """
        # We don't want early stopping.
        while self.epoch < max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
        """
        # Load validation losses of the previous stage (if previous stage
        # exists):
        if (path_checkpoint is not None) and os.path.exists(
            os.path.join(path_checkpoint, "val_losses.npy")
        ):
            self.val_losses = np.load(
                os.path.join(path_checkpoint, "val_losses.npy")
            ).tolist()
            min_val_loss = min(self.val_losses)
        else:
            self.val_losses = []
            min_val_loss = float("inf")
        while self.epoch < max_num_epochs:
            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    None,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=True,
                )
                train_loss = torch.mean(train_losses)
                train_loss = train_loss + self.gamma * get_calibration_error(
                    model=self._neural_net,
                    x=theta_batch,
                    y=x_batch,
                    device=self._device,
                    prior=self._prior,
                    n_samples=self.num_samples,
                    calibration=self.calibration,
                    logits=False,
                    instances_subsample=self.instances_subsample,
                )
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device)  # ,
                        # batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        None,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=True,
                    )
                    val_loss_sum += (
                        torch.mean(val_losses)
                        + self.gamma
                        * get_calibration_error(
                            model=self._neural_net,
                            x=theta_batch,
                            y=x_batch,
                            device=self._device,
                            prior=self._prior,
                            n_samples=self.num_samples,
                            calibration=self.calibration,
                            logits=False,
                            instances_subsample=self.instances_subsample,
                        )
                    ).item()
                    val_log_prob_sum -= val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            self._val_loss = val_loss_sum / len(val_loader)  # type: ignore
            self.val_losses.append(self._val_loss)

            if self._val_loss < min_val_loss:
                min_val_loss = self._val_loss
                self._best_state_dict = self._estimator_cpu_state_dict()

            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(
                time.time() - epoch_start_time
            )
            self.epoch += 1
            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        """
        self._report_convergence_at_end(
            self.epoch, stop_after_epochs, max_num_epochs
        )
        """

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(
            np.max(self._summary["validation_log_probs"])
        )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    @torch.no_grad()
    def _estimator_cpu_state_dict(self):
        # Check if we're training a Data Parallel model.
        self._neural_net.eval()
        self._neural_net = self._neural_net.cpu()
        if isinstance(self._neural_net, torch.nn.DataParallel):
            state_dict = deepcopy(self._neural_net.module.state_dict())
        else:
            state_dict = deepcopy(self._neural_net.state_dict())
        # Move back to the original device
        self._neural_net = self._neural_net.to(self._device)

        return state_dict

    def build_posterior(
        self,
        density_estimator: Optional[nn.Module] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "rejection",
        mcmc_method: str = "slice_np",
        vi_method: str = "rKL",
        mcmc_parameters: Dict[str, Any] = {},
        vi_parameters: Dict[str, Any] = {},
        rejection_sampling_parameters: Dict[str, Any] = {},
    ) -> Union[
        MCMCPosterior, RejectionPosterior, VIPosterior, DirectPosterior
    ]:
        r"""Build posterior from the neural density estimator.
        For SNPE, the posterior distribution that is returned here implements the
        following functionality over the raw neural density estimator:
        - correct the calculation of the log probability such that it compensates for
            the leakage.
        - reject samples that lie outside of the prior bounds.
        - alternatively, if leakage is very high (which can happen for multi-round
            SNPE), sample from the posterior with MCMC.
        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior` or `DirectPosterior`. By default,
                `DirectPosterior` is used. Only if `rejection_sampling_parameters`
                contains `proposal`, a `RejectionPosterior` is instantiated.
        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert self._prior is not None, (
                "You did not pass a prior. You have to pass the prior either at "
                "initialization `inference = SNPE(prior)` or to "
                "`.build_posterior(prior=prior)`."
            )
            prior = self._prior
        else:
            utils.check_prior(prior)

        if density_estimator is None:
            posterior_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            posterior_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator=posterior_estimator,
            prior=prior,
            x_o=None,
        )

        if sample_with == "rejection":
            if "proposal" in rejection_sampling_parameters.keys():
                self._posterior = RejectionPosterior(
                    potential_fn=potential_fn,
                    device=device,
                    x_shape=self._x_shape,
                    **rejection_sampling_parameters,
                )
            else:
                self._posterior = DirectPosterior(
                    posterior_estimator=posterior_estimator,
                    prior=prior,
                    x_shape=self._x_shape,
                    device=device,
                )
        elif sample_with == "mcmc":
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                x_shape=self._x_shape,
                **mcmc_parameters,
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                x_shape=self._x_shape,
                **vi_parameters,
            )
        else:
            raise NotImplementedError

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

    @abc.abstractmethod
    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        raise NotImplementedError

    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        """Return loss with proposal correction (`round_>0`) or without it (`round_=0`).
        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.
        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        if self._round == 0 or force_first_round_loss:
            # Use posterior log prob (without proposal correction) for first round.
            log_prob = self._neural_net.log_prob(theta, x)
        else:
            log_prob = self._log_prob_proposal_posterior(
                theta, x, masks, proposal
            )

        return -(calibration_kernel(x) * log_prob)

    def _check_proposal(self, proposal):
        """
        Check for validity of the provided proposal distribution.
        If the proposal is a `NeuralPosterior`, we check if the default_x is set.
        If the proposal is **not** a `NeuralPosterior`, we warn since it is likely that
        the user simply passed the prior, but this would still trigger atomic loss.
        """
        if proposal is not None:
            check_if_proposal_has_default_x(proposal)

            if isinstance(proposal, RestrictedPrior):
                if proposal._prior is not self._prior:
                    warn(
                        "The proposal you passed is a `RestrictedPrior`, but the "
                        "proposal distribution it uses is not the prior (it can be "
                        "accessed via `RestrictedPrior._prior`). We do not "
                        "recommend to mix the `RestrictedPrior` with multi-round "
                        "SNPE."
                    )
            elif (
                not isinstance(proposal, NeuralPosterior)
                and proposal is not self._prior
            ):
                warn(
                    "The proposal you passed is neither the prior nor a "
                    "`NeuralPosterior` object. If you are an expert user and did so "
                    "for research purposes, this is fine. If not, you might be doing "
                    "something wrong: feel free to create an issue on Github."
                )
        elif self._round > 0:
            raise ValueError(
                "A proposal was passed but no prior was passed at initialisation. When "
                "running multi-round inference, a prior needs to be specified upon "
                "initialisation. Potential fix: setting the `._prior` attribute or "
                "re-initialisation. If the samples passed to `append_simulations()` "
                "were sampled from the prior, single-round inference can be performed "
                "with `append_simulations(..., proprosal=None)`."
            )


class CalibratedSNPE(CalibratedPosteriorEstimator):
    """
    This class is a reimplementation (mostly copy-paste) of the original SNPE
    class (https://github.com/mackelab/sbi/blob/
    8fe45c9fccb086fe3f6b90ae82b0ca347f2e6457/sbi/inference/snpe/snpe_c.py).
    """

    def __init__(
        self,
        prior: Union[Distribution, torch.nn.Module],
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        calibration: float = 0,
        num_samples: int = 100,
        gamma: float = 100,  # `lambda` in BNRE paper
        instances_subsample: float = 1,
        validation_fraction: float = 0.1,
    ):
        r"""SNPE-C / APT [1].
        [1] _Automatic Posterior Transformation for Likelihood-free Inference_,
            Greenberg et al., ICML 2019, https://arxiv.org/abs/1905.07488.
        This class implements two loss variants of SNPE-C: the non-atomic and
        the atomic version. The atomic loss of SNPE-C can be used for any
        density estimator, i.e. also for normalizing flows. However, it
        suffers from leakage issues. On the other hand, the non-atomic loss
        can only be used only if the proposal distribution is a mixture of
        Gaussians, the density estimator is a mixture of Gaussians, and the
        prior is either Gaussian or Uniform. It does not suffer from leakage
        issues. At the beginning of each round, we print whether the non-atomic
        or the atomic version is used. In this codebase, we will automatically
        switch to the non-atomic loss if the following criteria are
        fulfilled:<br/>
        - proposal is a `DirectPosterior` with density_estimator `mdn`, as
        built with `utils.sbi.posterior_nn()`.<br/>
        - the density estimator is a `mdn`, as built with
            `utils.sbi.posterior_nn()`.<br/>
        - `isinstance(prior, MultivariateNormal)` (from `torch.distributions`)
        or `isinstance(prior, sbi.utils.BoxUniform)`

        Note that custom implementations of any of these densities (or
        estimators) will not trigger the non-atomic loss, and the algorithm
        will fall back onto using the atomic loss.

        Args:
            prior: A probability distribution that expresses prior knowledge
                about the parameters, e.g. which ranges are meaningful for
                them.
            density_estimator: If it is a string, use a pre-configured network
                of the provided type (one of nsf, maf, mdn, made).
                Alternatively, a function that builds a custom neural network
                can be provided. The function will be called with the first
                batch of simulations (theta, x), which can thus be used for
                shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the
                strings INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among
                others, log file location (default is `<current working
                directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
            calibration: Float between 0 and 1. 0 means conservatiness
                regularizer, 1 calibration regularizer, number in between is
                mixture
            num_samples: Number of samples
            gamma: Lambda (strength of whole balancing regularizer)
            instances_subsample: Number of instances subsamples
            validation_fraction: The fraction of data to use for validation.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        validation_inputs: Union[NumpyDataset, data.TensorDataset] = None,
        validation_outputs: Union[NumpyDataset, data.TensorDataset] = None,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        use_combined_loss: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        path_checkpoint=None,
    ) -> nn.Module:
        """
        Return density estimator that approximates the distribution
        `p(theta | x)`.
        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            validation_inputs: Theta values of validation data.
            validation_outputs: Simulation outputs for the validation data.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss
        kwargs = del_entries(
            locals(),
            entries=("self", "__class__", "num_atoms", "use_combined_loss"),
        )

        self._round = max(self._data_round_index)
        self.val_inputs = validation_inputs
        self.val_outputs = validation_outputs

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator._distribution, mdn)
                and isinstance(self._neural_net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().train(**kwargs)

    def _set_state_for_mog_proposal(self) -> None:
        """Set state variables that are used at each training step of non-atomic SNPE-C.
        Three things are computed:
        1) Check if z-scoring was requested. To do so, we check if the `_transform`
            argument of the net had been a `CompositeTransform`. See pyknos mdn.py.
        2) Define a (potentially standardized) prior. It's standardized if z-scoring
            had been requested.
        3) Compute (Precision * mean) for the prior. This quantity is used at every
            training step if the prior is Gaussian.
        """

        self.z_score_theta = isinstance(
            self._neural_net._transform, CompositeTransform
        )

        self._set_maybe_z_scored_prior()

        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            self.prec_m_prod_prior = torch.mv(
                self._maybe_z_scored_prior.precision_matrix,  # type: ignore
                self._maybe_z_scored_prior.loc,  # type: ignore
            )

    def _set_maybe_z_scored_prior(self) -> None:
        r"""Compute and store potentially standardized prior (if z-scoring was done).
        The proposal posterior is:
        $pp(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$
        Let's denote z-scored theta by `a`: a = (theta - mean) / std
        Then pp'(a|x) = 1/Z_2 * q'(a|x) * prop'(a) / p'(a)$
        The ' indicates that the evaluation occurs in standardized space. The constant
        scaling factor has been absorbed into Z_2.
        From the above equation, we see that we need to evaluate the prior **in
        standardized space**. We build the standardized prior in this function.
        The standardize transform that is applied to the samples theta does not use
        the exact prior mean and std (due to implementation issues). Hence, the z-scored
        prior will not be exactly have mean=0 and std=1.
        """

        if self.z_score_theta:
            scale = self._neural_net._transform._transforms[0]._scale
            shift = self._neural_net._transform._transforms[0]._shift

            # Following the definintion of the linear transform in
            # `standardizing_transform` in `sbiutils.py`:
            # shift=-mean / std
            # scale=1 / std
            # Solving these equations for mean and std:
            estim_prior_std = 1 / scale
            estim_prior_mean = -shift * estim_prior_std

            # Compute the discrepancy of the true prior mean and std and the mean and
            # std that was empirically estimated from samples.
            # N(theta|m,s) = N((theta-m_e)/s_e|(m-m_e)/s_e, s/s_e)
            # Above: m,s are true prior mean and std. m_e,s_e are estimated prior mean
            # and std (estimated from samples and used to build standardize transform).
            almost_zero_mean = (
                self._prior.mean - estim_prior_mean
            ) / estim_prior_std
            almost_one_std = torch.sqrt(self._prior.variance) / estim_prior_std

            if isinstance(self._prior, MultivariateNormal):
                self._maybe_z_scored_prior = MultivariateNormal(
                    almost_zero_mean, torch.diag(almost_one_std)
                )
            else:
                range_ = torch.sqrt(almost_one_std * 3.0)
                self._maybe_z_scored_prior = utils.BoxUniform(
                    almost_zero_mean - range_, almost_zero_mean + range_
                )
        else:
            self._maybe_z_scored_prior = self._prior

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: DirectPosterior,
    ) -> Tensor:
        """Return the log-probability of the proposal posterior.
        If the proposal is a MoG, the density estimator is a MoG, and the prior is
        either Gaussian or uniform, we use non-atomic loss. Else, use atomic loss (which
        suffers from leakage).
        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
            proposal: Proposal distribution.
        Returns: Log-probability of the proposal posterior.
        """

        if self.use_non_atomic_loss:
            return self._log_prob_proposal_posterior_mog(theta, x, proposal)
        else:
            return self._log_prob_proposal_posterior_atomic(theta, x, masks)

    def _log_prob_proposal_posterior_atomic(
        self, theta: Tensor, x: Tensor, masks: Tensor
    ):
        """Return log probability of the proposal posterior for atomic proposals.
        We have two main options when evaluating the proposal posterior.
            (1) Generate atoms from the proposal prior.
            (2) Generate atoms from a more targeted distribution, such as the most
                recent posterior.
        If we choose the latter, it is likely beneficial not to do this in the first
        round, since we would be sampling from a randomly-initialized neural density
        estimator.
        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Mask that is True for prior samples in the batch in order to train
                them with prior loss.
        Returns:
            Log-probability of the proposal posterior.
        """

        batch_size = theta.shape[0]

        num_atoms = int(
            clamp_and_warn(
                "num_atoms", self._num_atoms, min_val=2, max_val=batch_size
            )
        )

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = (
            ones(batch_size, batch_size)
            * (1 - eye(batch_size))
            / (batch_size - 1)
        )

        choices = torch.multinomial(
            probs, num_samples=num_atoms - 1, replacement=False
        )
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat(
            (theta[:, None, :], contrasting_theta), dim=1
        ).reshape(batch_size * num_atoms, -1)

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        log_prob_posterior = self._neural_net.log_prob(
            atomic_theta, repeated_x
        )
        utils.assert_all_finite(log_prob_posterior, "posterior eval")
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self._prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        utils.assert_all_finite(log_prob_prior, "prior eval")

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[
            :, 0
        ] - torch.logsumexp(unnormalized_log_prob, dim=-1)
        utils.assert_all_finite(
            log_prob_proposal_posterior, "proposal posterior eval"
        )

        # XXX This evaluates the posterior on _all_ prior samples
        if self._use_combined_loss:
            log_prob_posterior_non_atomic = self._neural_net.log_prob(theta, x)
            masks = masks.reshape(-1)
            log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic
                + log_prob_proposal_posterior
            )

        return log_prob_proposal_posterior

    def _log_prob_proposal_posterior_mog(
        self, theta: Tensor, x: Tensor, proposal: DirectPosterior
    ) -> Tensor:
        """Return log-probability of the proposal posterior for MoG proposal.
        For MoG proposals and MoG density estimators, this can be done in closed form
        and does not require atomic loss (i.e. there will be no leakage issues).
        Notation:
        m are mean vectors.
        prec are precision matrices.
        cov are covariance matrices.
        _p at the end indicates that it is the proposal.
        _d indicates that it is the density estimator.
        _pp indicates the proposal posterior.
        All tensors will have shapes (batch_dim, num_components, ...)
        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            proposal: Proposal distribution.
        Returns:
            Log-probability of the proposal posterior.
        """

        # Evaluate the proposal. MDNs do not have functionality to run the embedding_net
        # and then get the mixture_components (**without** calling log_prob()). Hence,
        # we call them separately here.
        encoded_x = proposal.posterior_estimator._embedding_net(
            proposal.default_x
        )
        dist = (
            proposal.posterior_estimator._distribution
        )  # defined to avoid ugly black formatting.
        logits_p, m_p, prec_p, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_p = logits_p - torch.logsumexp(
            logits_p, dim=-1, keepdim=True
        )

        # Evaluate the density estimator.
        encoded_x = self._neural_net._embedding_net(x)
        dist = (
            self._neural_net._distribution
        )  # defined to avoid black formatting.
        logits_d, m_d, prec_d, _, _ = dist.get_mixture_components(encoded_x)
        norm_logits_d = logits_d - torch.logsumexp(
            logits_d, dim=-1, keepdim=True
        )

        # z-score theta if it z-scoring had been requested.
        theta = self._maybe_z_score_theta(theta)

        # Compute the MoG parameters of the proposal posterior.
        (
            logits_pp,
            m_pp,
            prec_pp,
            cov_pp,
        ) = self._automatic_posterior_transformation(
            norm_logits_p, m_p, prec_p, norm_logits_d, m_d, prec_d
        )

        # Compute the log_prob of theta under the product.
        log_prob_proposal_posterior = utils.mog_log_prob(
            theta, logits_pp, m_pp, prec_pp
        )
        utils.assert_all_finite(
            log_prob_proposal_posterior,
            """the evaluation of the MoG proposal posterior. This is likely due to a 
            numerical instability in the training procedure. Please create an issue on Github.""",
        )

        return log_prob_proposal_posterior

    def _automatic_posterior_transformation(
        self,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        r"""Returns the MoG parameters of the proposal posterior.
        The proposal posterior is:
        $pp(\theta|x) = 1/Z * q(\theta|x) * prop(\theta) / p(\theta)$
        In words: proposal posterior = posterior estimate * proposal / prior.
        If the posterior estimate and the proposal are MoG and the prior is either
        Gaussian or uniform, we can solve this in closed-form. The is implemented in
        this function.
        This function implements Appendix A1 from Greenberg et al. 2019.
        We have to build L*K components. How do we do this?
        Example: proposal has two components, density estimator has three components.
        Let's call the two components of the proposal i,j and the three components
        of the density estimator x,y,z. We have to multiply every component of the
        proposal with every component of the density estimator. So, what we do is:
        1) for the proposal, build: i,i,i,j,j,j. Done with torch.repeat_interleave()
        2) for the density estimator, build: x,y,z,x,y,z. Done with torch.repeat()
        3) Multiply them with simple matrix operations.
        Args:
            logits_p: Component weight of each Gaussian of the proposal.
            means_p: Mean of each Gaussian of the proposal.
            precisions_p: Precision matrix of each Gaussian of the proposal.
            logits_d: Component weight for each Gaussian of the density estimator.
            means_d: Mean of each Gaussian of the density estimator.
            precisions_d: Precision matrix of each Gaussian of the density estimator.
        Returns: (Component weight, mean, precision matrix, covariance matrix) of each
            Gaussian of the proposal posterior. Has L*K terms (proposal has L terms,
            density estimator has K terms).
        """

        precisions_pp, covariances_pp = self._precisions_proposal_posterior(
            precisions_p, precisions_d
        )

        means_pp = self._means_proposal_posterior(
            covariances_pp, means_p, precisions_p, means_d, precisions_d
        )

        logits_pp = self._logits_proposal_posterior(
            means_pp,
            precisions_pp,
            covariances_pp,
            logits_p,
            means_p,
            precisions_p,
            logits_d,
            means_d,
            precisions_d,
        )

        return logits_pp, means_pp, precisions_pp, covariances_pp

    def _precisions_proposal_posterior(
        self, precisions_p: Tensor, precisions_d: Tensor
    ):
        """Return the precisions and covariances of the proposal posterior.
        Args:
            precisions_p: Precision matrices of the proposal distribution.
            precisions_d: Precision matrices of the density estimator.
        Returns: (Precisions, Covariances) of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        precisions_p_rep = precisions_p.repeat_interleave(num_comps_d, dim=1)
        precisions_d_rep = precisions_d.repeat(1, num_comps_p, 1, 1)

        precisions_pp = precisions_p_rep + precisions_d_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            precisions_pp -= self._maybe_z_scored_prior.precision_matrix

        covariances_pp = torch.inverse(precisions_pp)

        return precisions_pp, covariances_pp

    def _means_proposal_posterior(
        self,
        covariances_pp: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """Return the means of the proposal posterior.
        means_pp = C_ix * (P_i * m_i + P_x * m_x - P_o * m_o).
        Args:
            covariances_pp: Covariance matrices of the proposal posterior.
            means_p: Means of the proposal distribution.
            precisions_p: Precision matrices of the proposal distribution.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.
        Returns: Means of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # First, compute the product P_i * m_i and P_j * m_j
        prec_m_prod_p = batched_mixture_mv(precisions_p, means_p)
        prec_m_prod_d = batched_mixture_mv(precisions_d, means_d)

        # Repeat them to allow for matrix operations: same trick as for the precisions.
        prec_m_prod_p_rep = prec_m_prod_p.repeat_interleave(num_comps_d, dim=1)
        prec_m_prod_d_rep = prec_m_prod_d.repeat(1, num_comps_p, 1)

        # Means = C_ij * (P_i * m_i + P_x * m_x - P_o * m_o).
        summed_cov_m_prod_rep = prec_m_prod_p_rep + prec_m_prod_d_rep
        if isinstance(self._maybe_z_scored_prior, MultivariateNormal):
            summed_cov_m_prod_rep -= self.prec_m_prod_prior

        means_pp = batched_mixture_mv(covariances_pp, summed_cov_m_prod_rep)

        return means_pp

    @staticmethod
    def _logits_proposal_posterior(
        means_pp: Tensor,
        precisions_pp: Tensor,
        covariances_pp: Tensor,
        logits_p: Tensor,
        means_p: Tensor,
        precisions_p: Tensor,
        logits_d: Tensor,
        means_d: Tensor,
        precisions_d: Tensor,
    ):
        """Return the component weights (i.e. logits) of the proposal posterior.
        Args:
            means_pp: Means of the proposal posterior.
            precisions_pp: Precision matrices of the proposal posterior.
            covariances_pp: Covariance matrices of the proposal posterior.
            logits_p: Component weights (i.e. logits) of the proposal distribution.
            means_p: Means of the proposal distribution.
            precisions_p: Precision matrices of the proposal distribution.
            logits_d: Component weights (i.e. logits) of the density estimator.
            means_d: Means of the density estimator.
            precisions_d: Precision matrices of the density estimator.
        Returns: Component weights of the proposal posterior. L*K terms.
        """

        num_comps_p = precisions_p.shape[1]
        num_comps_d = precisions_d.shape[1]

        # Compute log(alpha_i * beta_j)
        logits_p_rep = logits_p.repeat_interleave(num_comps_d, dim=1)
        logits_d_rep = logits_d.repeat(1, num_comps_p)
        logit_factors = logits_p_rep + logits_d_rep

        # Compute sqrt(det()/(det()*det()))
        logdet_covariances_pp = torch.logdet(covariances_pp)
        logdet_covariances_p = -torch.logdet(precisions_p)
        logdet_covariances_d = -torch.logdet(precisions_d)

        # Repeat the proposal and density estimator terms such that there are LK terms.
        # Same trick as has been used above.
        logdet_covariances_p_rep = logdet_covariances_p.repeat_interleave(
            num_comps_d, dim=1
        )
        logdet_covariances_d_rep = logdet_covariances_d.repeat(1, num_comps_p)

        log_sqrt_det_ratio = 0.5 * (
            logdet_covariances_pp
            - (logdet_covariances_p_rep + logdet_covariances_d_rep)
        )

        # Compute for proposal, density estimator, and proposal posterior:
        # mu_i.T * P_i * mu_i
        exponent_p = batched_mixture_vmv(precisions_p, means_p)
        exponent_d = batched_mixture_vmv(precisions_d, means_d)
        exponent_pp = batched_mixture_vmv(precisions_pp, means_pp)

        # Extend proposal and density estimator exponents to get LK terms.
        exponent_p_rep = exponent_p.repeat_interleave(num_comps_d, dim=1)
        exponent_d_rep = exponent_d.repeat(1, num_comps_p)
        exponent = -0.5 * (exponent_p_rep + exponent_d_rep - exponent_pp)

        logits_pp = logit_factors + log_sqrt_det_ratio + exponent

        return logits_pp

    def _maybe_z_score_theta(self, theta: Tensor) -> Tensor:
        """Return potentially standardized theta if z-scoring was requested."""

        if self.z_score_theta:
            theta, _ = self._neural_net._transform(theta)

        return theta
