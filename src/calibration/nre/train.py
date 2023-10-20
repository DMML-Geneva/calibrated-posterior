r"""Utility program to train ratio estimators.

This program provides a whole range of utilities to
monitor and train ratio estimators in various ways.
All defined through command line arguments!

"""

import argparse
from copy import deepcopy
import hypothesis as h
import hypothesis.workflow as w
import json
import numpy as np
import os
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn

from hypothesis.train import RatioEstimatorTrainer
from hypothesis.util import load_module
from hypothesis.util.data import BaseNamedDataset
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NamedSubDataset
from tqdm import tqdm


# Globals
p_bottom = None
p_top = None


def none_or_str(value):
    if value == "None":
        return None
    return value


class GDStep(object):
    r"""Creates a callable that performs gradient descent (GD) optimization steps
    for parameters :math:`\phi` with respect to differentiable loss values.
    The callable takes a scalar loss :math:`l` as input, performs a step
    .. math:: \phi \gets \text{GD}(\phi, \nabla_{\!\phi} \, l)
    and returns the loss, detached from the computational graph. To prevent invalid
    parameters, steps are skipped if not-a-number (NaN) or infinite values are found
    in the gradient. This feature requires CPU-GPU synchronization, which could be a
    bottleneck for some applications.
    Arguments:
        optimizer: An optimizer instance (e.g. :class:`torch.optim.SGD`).
        clip: The norm at which the gradients are clipped. If :py:`None`,
            gradients are not clipped.
    """
    # Copied from: https://github.com/francois-rozet/lampe/blob/63e37ef5de077c0fe67c958e8c0c1420035b026c/lampe/utils.py#L12

    def __init__(self, optimizer: Optimizer, clip: float = None):
        self.optimizer = optimizer
        self.parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        self.clip = clip

    def __call__(self, loss: Tensor) -> Tensor:
        if loss.isfinite().all():
            self.optimizer.zero_grad()
            loss.backward()

            if self.clip is None:
                self.optimizer.step()
            else:
                norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if norm.isfinite():
                    self.optimizer.step()

        return loss.detach()


class Trainer(RatioEstimatorTrainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.step_optimizer = GDStep(self._optimizer, clip=1)

    def read_loss_history(self, path):
        if os.path.isfile(path + "/losses-test.npy"):
            self._losses_test = np.load(path + "/losses-test.npy").tolist()
        if os.path.isfile(path + "/losses-validation.npy"):
            self._losses_validate = np.load(
                path + "/losses-validation.npy",
            ).tolist()
        if os.path.isfile(path + "/losses-train.npy"):
            self._losses_train = np.load(path + "/losses-train.npy").tolist()

    def read_best_state_dict(self, path):
        weights = torch.load(path, map_location="cpu")
        self._state_dict_best = weights

    def train(self):
        assert self._dataset_train is not None
        self._estimator.train()
        loader = self._allocate_train_loader()
        losses = []
        total_batches = len(loader)
        for index, sample_joint in enumerate(loader):
            self.call_event(
                self.events.batch_train_start,
                batch_index=index,
                total_batches=total_batches,
            )
            for k, v in sample_joint.items():
                sample_joint[k] = v.to(self._accelerator, non_blocking=True)
            loss = self._criterion(**sample_joint)
            loss = self.step_optimizer(loss)
            loss = loss.item()
            losses.append(loss)
            self.call_event(
                self.events.batch_train_complete,
                batch_index=index,
                total_batches=total_batches,
                loss=loss,
            )
        expected_loss = np.mean(losses)

        return expected_loss

    @torch.no_grad()
    def _estimator_cpu_state_dict(self):
        # Check if we're training a Data Parallel model.
        self._estimator.eval()
        self._estimator = self._estimator.cpu()
        if isinstance(self._estimator, torch.nn.DataParallel):
            state_dict = deepcopy(self._estimator.module.state_dict())
        else:
            state_dict = deepcopy(self._estimator.state_dict())
        # Move back to the original device
        self._estimator = self._estimator.to(self.accelerator)

        return state_dict


def main(arguments):
    # Allocate the datasets
    dataset_train, dataset_validate, dataset_test = load_datasets(arguments)
    # Allocate the ratio estimator
    estimator = load_ratio_estimator(arguments)
    # Allocate the optimizer
    optimizer = load_optimizer(arguments, estimator)
    # Allocate the criterion
    criterion = load_criterion(arguments, estimator, dataset_train)
    # Allocate the trainer instance
    trainer = Trainer(
        accelerator=h.accelerator,
        batch_size=arguments.batch_size,
        criterion=criterion,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        dataset_validate=dataset_validate,
        epochs=arguments.epochs,
        optimizer=optimizer,
        pin_memory=arguments.pin_memory,
        show=arguments.show,
        shuffle=not arguments.dont_shuffle,
        workers=arguments.workers,
    )
    if arguments.read_checkpoint is not None:
        trainer.read_loss_history(arguments.read_checkpoint)
        trainer.read_best_state_dict(arguments.read_checkpoint + "/weights.th")
    # Add the hooks to the training object.
    add_hooks(arguments, trainer)
    # Start the optimization procedure
    trainer.fit()
    # Check if the path does not exists.
    if not os.path.exists(arguments.out):
        os.makedirs(arguments.out)
    # Save the generated results.
    if os.path.isdir(arguments.out):
        # Save the associated losses.
        if len(trainer.losses_test) > 0:
            np.save(arguments.out + "/losses-test.npy", trainer.losses_test)
        if len(trainer.losses_validate) > 0:
            np.save(
                arguments.out + "/losses-validation.npy",
                trainer.losses_validate,
            )
        if len(trainer.losses_train) > 0:
            np.save(arguments.out + "/losses-train.npy", trainer.losses_train)
        # Save the state dict of the best ratio estimator
        torch.save(trainer.best_state_dict, arguments.out + "/weights.th")
        torch.save(trainer.state_dict, arguments.out + "/weights-final.th")
        torch.save(
            trainer.optimizer.state_dict(), arguments.out + "/optimizer.th"
        )


def load_criterion(arguments, estimator, dataset_train):
    Criterion = load_module(arguments.criterion)
    kwargs = arguments.criterion_args
    kwargs["batch_size"] = arguments.batch_size
    kwargs["gamma"] = arguments.gamma
    kwargs["logits"] = arguments.logits
    kwargs["prior"] = getattr(load_module(arguments.prior), "Prior")()
    criterion = Criterion(estimator=estimator, **kwargs)

    return criterion


def load_datasets(arguments):
    # Load test set
    if arguments.data_test is not None:
        dataset_test = load_module(arguments.data_test)()
        assert isinstance(dataset_test, BaseNamedDataset)
    else:
        dataset_test = None

    # Load train set
    if arguments.data_train is not None:
        dataset_train = load_module(arguments.data_train)()
        assert isinstance(dataset_train, BaseNamedDataset)
    else:
        dataset_train = None

    # Load validation set
    if arguments.data_validate is not None:
        dataset_validate = load_module(arguments.data_validate)()
        assert isinstance(dataset_validate, BaseNamedDataset)
    elif dataset_train is not None:
        # Split train set into train and validation set
        n = len(dataset_train)
        indices = np.arange(n)
        np.random.shuffle(indices)
        validate_samples = int(n * arguments.validate_fraction)
        dataset_validate = NamedSubDataset(
            dataset_train, indices[:validate_samples]
        )
        dataset_train = NamedSubDataset(
            dataset_train, indices[validate_samples:]
        )
    else:
        dataset_validate = None

    return dataset_train, dataset_validate, dataset_test


def load_ratio_estimator(arguments):
    RatioEstimator = load_module(arguments.estimator)
    estimator = RatioEstimator()
    # Check if weights have been specified.
    if arguments.weights is not None:
        weights = torch.load(arguments.weights, map_location="cpu")
        estimator.load_state_dict(weights)
    elif arguments.read_checkpoint is not None:
        weights = torch.load(
            f"{arguments.read_checkpoint}/weights-final.th", map_location="cpu"
        )
        estimator.load_state_dict(weights)
    # Check if we are able to allocate a data parallel model.
    if torch.cuda.device_count() > 1 and arguments.data_parallel:
        estimator = torch.nn.DataParallel(estimator)
    estimator = estimator.to(h.accelerator)

    return estimator


def load_optimizer(arguments, estimator):
    optimizer = torch.optim.AdamW(
        estimator.parameters(),
        lr=arguments.lr,
        weight_decay=arguments.weight_decay,
    )
    if arguments.read_checkpoint is not None:
        optimizer_params = torch.load(
            f"{arguments.read_checkpoint}/optimizer.th", map_location="cpu"
        )
        optimizer.load_state_dict(optimizer_params)

    return optimizer


def add_hooks(arguments, trainer):
    # Add the learning rate scheduling hooks
    add_hooks_lr_scheduling(arguments, trainer)
    # Check if a custom hook method has been specified.
    if arguments.hooks is not None:
        hook_loader = load_module(arguments.hooks)
        hook_loader(arguments, trainer)


def add_hooks_lr_scheduling(arguments, trainer):
    # Check which learning rate scheduler has been specified
    if arguments.lrsched_on_plateau:
        add_hooks_lr_scheduling_on_plateau(arguments, trainer)
    elif arguments.lrsched_cyclic:
        add_hooks_lr_scheduling_cyclic(arguments, trainer)


def add_hooks_lr_scheduling_on_plateau(arguments, trainer):
    # Check if a test set is available, as the scheduler required a metric.
    if arguments.data_test is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, verbose=True
        )

        def schedule(trainer, **kwargs):
            scheduler.step(trainer.losses_test[-1])

        trainer.add_event_handler(trainer.events.epoch_complete, schedule)


def add_hooks_lr_scheduling_cyclic(arguments, trainer):
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        trainer.optimizer,
        cycle_momentum=False,
        base_lr=arguments.lrsched_cyclic_base_lr,
        max_lr=arguments.lrsched_cyclic_max_lr,
    )

    def schedule(trainer, **kwargs):
        scheduler.step()

    trainer.add_event_handler(trainer.events.batch_train_complete, schedule)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Criterion settings
    parser.add_argument(
        "--criterion",
        type=str,
        default="hypothesis.nn.ratio_estimation.BalancedCriterion",
        help="Optimization criterion (default: hypothesis.nn.ratio_estimation.BalancedCriterion).",
    )
    parser.add_argument(
        "--criterion-args",
        type=json.loads,
        default="{}",
        help="Additional criterion arguments (default: '{}').",
    )
    parser.add_argument(
        "--prior",
        type=str,
        default=None,
        help="Prior distribution (default: None).",
    )
    parser.add_argument(
        "--extent",
        type=str,
        default=None,
        help="Extent of support (default: None).",
    )
    # General settings
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Enable data-parallel training whenever multiple GPU's are available (default: false).",
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Disable the usage of GPU's (default: false).",
    )
    parser.add_argument(
        "--dont-shuffle",
        action="store_true",
        help="Do not shuffle the datasets (default: false).",
    )
    parser.add_argument(
        "--hooks",
        type=str,
        default=None,
        help="Method name (including module) to which adds custom hooks to the trainer (default: none).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=".",
        help="Output directory of the generated files (default: '.').",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Memory map and pipeline data loading to the GPU (default: false).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show progress of the training to stdout (default: false).",
    )
    # Optimization settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256).",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs (default: 1)."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=100.0,
        help="Hyper-parameter to force the calibration criterion (default: 100.0). This option will overwrite the corresponding setting specified in `--criterion-args`.",
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        help="Use the logit-trick for the minimization criterion (default: false).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent data loaders (default: 4).",
    )
    # Data settings
    parser.add_argument(
        "--data-test",
        type=str,
        default=None,
        help="Full classname of the testing dataset (default: none, optional).",
    )
    parser.add_argument(
        "--data-train",
        type=str,
        default=None,
        help="Full classname of the training dataset (default: none).",
    )
    parser.add_argument(
        "--data-validate",
        type=str,
        default=None,
        help="Full classname of the validation dataset (default: none, optional).",
    )
    parser.add_argument(
        "--validate-fraction",
        type=float,
        default=0.2,
        help="Fraction of the training set to use as validation set if validation set not provided (default: 0.2)",
    )
    # Ratio estimator settings
    parser.add_argument(
        "--estimator",
        type=str,
        default=None,
        help="Full classname of the ratio estimator (default: none).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path for weight initialization (default: none).",
    )
    # Learning rate scheduling (you can only allocate 1 learning rate scheduler, they will be allocated in the following order.)
    ## Learning rate scheduling on a plateau
    parser.add_argument(
        "--lrsched-on-plateau",
        action="store_true",
        help="Enables learning rate scheduling whenever a plateau has been detected (default: false).",
    )
    ## Cyclic learning rate scheduling
    parser.add_argument(
        "--lrsched-cyclic",
        action="store_true",
        help="Enables cyclic learning rate scheduling. Requires a test dataset to be specified (default: false).",
    )
    parser.add_argument(
        "--lrsched-cyclic-base-lr",
        type=float,
        default=None,
        help="Base learning rate of the scheduler (default: --lr / 10).",
    )
    parser.add_argument(
        "--lrsched-cyclic-max-lr",
        type=float,
        default=None,
        help="Maximum learning rate of the scheduler (default: --lr).",
    )
    ## Patch to checkpoint
    parser.add_argument(
        "--read-checkpoint",
        type=none_or_str,
        default=None,
        help="Path for checkpoint (default: none).",
    )
    # Parse the supplied arguments
    arguments, _ = parser.parse_known_args()

    # Set the default options of the cyclic learning rate scheduler
    if arguments.lrsched_cyclic_base_lr is None:
        arguments.lrsched_cyclic_base_lr = arguments.lr / 10
    if arguments.lrsched_cyclic_max_lr is None:
        arguments.lrsched_cyclic_max_lr = arguments.lr

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.disable_gpu:
        h.disable_gpu()
    main(arguments)
