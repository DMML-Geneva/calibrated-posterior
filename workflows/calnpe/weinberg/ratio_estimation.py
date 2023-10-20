r"""CASBI: Conservative Amortized Simulation-Based Inference

Contains the definition of the ratio estimators
and the corresponding utilities.
"""

# import NF
import glob
import hypothesis as h
import numpy as np
import torch
import cloudpickle as pickle
import os

from hypothesis.benchmark.weinberg import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset
from tqdm import tqdm

from sbi.inference.base import infer
from sbi.utils.get_nn_models import posterior_nn
from torch.utils.tensorboard.writer import SummaryWriter

### Utilities ##################################################################

prior = Prior()

extent = [  # I know, this isn't very nice :(
    prior.low.item(),
    prior.high.item(),
]


@torch.no_grad()
def load_estimator(query, reduce="ratio_mean"):
    paths = glob.glob(query)
    if len(paths) > 1:
        estimators = [load_estimator(p) for p in paths]
        r = RatioEstimatorEnsemble(estimators, reduce=reduce)
    else:
        path = paths[0]
        if "/flow" in path:
            r = FlowRatioEstimator()
        else:
            r = ClassifierRatioEstimator()
        r.load_state_dict(torch.load(path))
    r = r.to(h.accelerator)
    r.eval()

    return r


@torch.no_grad()
def compute_log_posterior(r, observable, resolution=100, flow_sbi=False):
    # Prepare grid
    epsilon = 0.00001
    inputs = torch.linspace(extent[0], extent[1] - epsilon, resolution).view(
        -1, 1
    )  # Account for half-open interval of uniform prior
    inputs = inputs.to(h.accelerator)

    if flow_sbi:
        observable = observable.to(h.accelerator)
        log_posterior = (
            r.log_prob(inputs, x=observable, norm_posterior=False)[0]
            .view(resolution)
            .cpu()
        )
        assert log_posterior.shape == (resolution,)
    else:
        log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
        observables = observable.repeat(resolution, 1).float()
        observables = observables.to(h.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        log_posterior = (
            (log_prior_probabilities + log_ratios).view(resolution).cpu()
        )

    return log_posterior


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False):
    inputs = inputs.to(h.accelerator)
    outputs = outputs.to(h.accelerator)

    if flow_sbi:
        log_posterior = torch.Tensor(
            [
                r.log_prob(theta, x=x, norm_posterior=False)
                for theta, x in zip(inputs, outputs)
            ]
        )
        return log_posterior.squeeze()
    else:
        log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
        log_prior = prior.log_prob(inputs)

        return (log_prior + log_ratios).squeeze()


@torch.no_grad()
def coverage(r, inputs, outputs, alphas=[0.05], flow_sbi=False):
    n = len(inputs)
    covered = [0 for _ in alphas]
    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index].squeeze().unsqueeze(0)
        observable = outputs[index].squeeze().unsqueeze(0)
        nominal = nominal.to(h.accelerator)
        observable = observable.to(h.accelerator)
        pdf = compute_log_posterior(r, observable, flow_sbi=flow_sbi).exp()
        nominal_pdf = compute_log_pdf(
            r, nominal, observable, flow_sbi=flow_sbi
        ).exp()

        for i, alpha in enumerate(alphas):
            level = highest_density_level(pdf, alpha)
            if nominal_pdf >= level:
                covered[i] += 1

    return [x / n for x in covered]


### Ratio estimators ###########################################################


class ClassifierRatioEstimator(BaseRatioEstimator):
    def __init__(self):
        random_variables = {"inputs": (1,), "outputs": (4,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [128] * 3
        r = Class(activation=activation, trunk=trunk)
        super(ClassifierRatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, **kwargs):
        return self._r.log_ratio(**kwargs)


class FlowRatioEstimator(BaseRatioEstimator):
    """This didn't work for them."""

    def __init__(self):
        denominator = "inputs|outputs"
        random_variables = {"inputs": (1,), "outputs": (4,)}
        super(FlowRatioEstimator, self).__init__(
            denominator=denominator, random_variables=random_variables
        )
        # Flow definition for now a simple conditionnal autoregressive affine
        conditioner_type = NF.AutoregressiveConditioner
        conditioner_args = {
            "in_size": np.prod(random_variables["inputs"]),
            "hidden": [128, 128, 128],
            "out_size": 2,
            "cond_in": np.prod(random_variables["outputs"]),
        }
        normalizer_type = NF.AffineNormalizer
        normalizer_args = {}
        nb_flow = 5
        self.flow = NF.buildFCNormalizingFlow(
            nb_flow,
            conditioner_type,
            conditioner_args,
            normalizer_type,
            normalizer_args,
        )
        self._prior = Prior()

    def log_ratio(self, inputs, outputs, **kwargs):
        b_size = inputs.shape[0]
        log_posterior, _ = self.flow.compute_ll(
            inputs.view(b_size, -1), outputs.view(b_size, -1)
        )
        log_prior = self._prior.log_prob(inputs)

        return log_posterior.view(-1, 1) - log_prior.view(-1, 1)


### Datasets ###################################################################


class DatasetJointTrain(NamedDataset):
    def __init__(self, n=None):
        inputs = np.load("weinberg/data/train/inputs.npy")
        outputs = np.load("weinberg/data/train/outputs.npy")
        if n is not None:
            indices = np.random.choice(
                np.arange(len(inputs)), n, replace=False
            )
            inputs = inputs[indices, :]
            outputs = outputs[indices, :]
        inputs = TensorDataset(torch.from_numpy(inputs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetJointTrain, self).__init__(inputs=inputs, outputs=outputs)


class DatasetJointTrain1024(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain1024, self).__init__(n=1024)


class DatasetJointTrain2048(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain2048, self).__init__(n=2048)


class DatasetJointTrain4096(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain4096, self).__init__(n=4096)


class DatasetJointTrain8192(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain8192, self).__init__(n=8192)


class DatasetJointTrain16384(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain16384, self).__init__(n=16384)


class DatasetJointTrain32768(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain32768, self).__init__(n=32768)


class DatasetJointTrain65536(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain65536, self).__init__(n=65536)


class DatasetJointTrain131072(DatasetJointTrain):
    def __init__(self):
        super(DatasetJointTrain131072, self).__init__(n=131072)


class DatasetJointValidate(NamedDataset):
    def __init__(self):
        inputs = NumpyDataset("weinberg/data/validate/inputs.npy")
        outputs = NumpyDataset("weinberg/data/validate/outputs.npy")
        super(DatasetJointValidate, self).__init__(
            inputs=inputs, outputs=outputs
        )


class DatasetJointTest(NamedDataset):
    def __init__(self):
        inputs = NumpyDataset("weinberg/data/test/inputs.npy")
        outputs = NumpyDataset("weinberg/data/test/outputs.npy")
        super(DatasetJointTest, self).__init__(inputs=inputs, outputs=outputs)


class DatasetJointValidateSmall(NamedDataset):
    def __init__(self):
        if not os.path.exists("weinberg/data/validate/inputs_small.npy"):
            inputs = np.load("weinberg/data/validate/inputs.npy")
            if inputs.shape[0] >= 10000:
                inputs = inputs[:10000]

            np.save("weinberg/data/validate/inputs_small.npy", inputs)

        if not os.path.exists("weinberg/data/validate/outputs_small.npy"):
            outputs = np.load("weinberg/data/validate/outputs.npy")
            if outputs.shape[0] >= 10000:
                outputs = outputs[:10000]

            np.save("weinberg/data/validate/outputs_small.npy", outputs)

        inputs = NumpyDataset("weinberg/data/validate/inputs_small.npy")
        outputs = NumpyDataset("weinberg/data/validate/outputs_small.npy")

        super(DatasetJointValidateSmall, self).__init__(
            inputs=inputs, outputs=outputs
        )


class DatasetJointTestSmall(NamedDataset):
    def __init__(self):
        if not os.path.exists("weinberg/data/test/inputs_small.npy"):
            inputs = np.load("weinberg/data/test/inputs.npy")
            if inputs.shape[0] >= 10000:
                inputs = inputs[:10000]

            np.save("weinberg/data/test/inputs_small.npy", inputs)

        if not os.path.exists("weinberg/data/test/outputs_small.npy"):
            outputs = np.load("weinberg/data/test/outputs.npy")
            if outputs.shape[0] >= 10000:
                outputs = outputs[:10000]

            np.save("weinberg/data/test/outputs_small.npy", outputs)

        inputs = NumpyDataset("weinberg/data/test/inputs_small.npy")
        outputs = NumpyDataset("weinberg/data/test/outputs_small.npy")

        super(DatasetJointTestSmall, self).__init__(
            inputs=inputs, outputs=outputs
        )
