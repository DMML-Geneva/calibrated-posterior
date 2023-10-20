r"""CASBI: Conservative Amortized Simulation-Based Inference

Contains the definition of the ratio estimators
and the corresponding utilities.
"""

import glob
import hypothesis as h
import numpy as np
import torch
import pickle
import os

from hypothesis.benchmark.lotka_volterra_small import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset
from tqdm import tqdm

### Utilities ##################################################################

prior = Prior()

extent = [  # I know, this isn't very nice :(
    prior.low[0].item(),
    prior.high[0].item(),
    prior.low[1].item(),
    prior.high[1].item(),
]


@torch.no_grad()
def load_estimator(query, reduce="ratio_mean"):
    paths = glob.glob(query)
    if len(paths) > 1:
        estimators = [load_estimator(p) for p in paths]
        r = RatioEstimatorEnsemble(estimators, reduce=reduce)
    else:
        if len(paths) == 0:
            raise ValueError("No paths found for the query:", query)
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
    p1 = torch.linspace(
        extent[0], extent[1] - epsilon, resolution
    )  # Account for half-open interval of uniform prior
    p2 = torch.linspace(
        extent[2], extent[3] - epsilon, resolution
    )  # Account for half-open interval of uniform prior
    p1 = p1.to(h.accelerator)
    p2 = p2.to(h.accelerator)
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)

    if flow_sbi:
        observable = observable.to(h.accelerator)
        log_posterior = (
            r.log_prob(inputs, x=observable, norm_posterior=False)[0]
            .view(resolution, resolution)
            .cpu()
        )
    else:
        log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
        observables = observable.repeat(resolution**2, 1, 1, 1).float()
        observables = observables.to(h.accelerator)
        log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
        log_posterior = (
            (log_prior_probabilities + log_ratios)
            .view(resolution, resolution)
            .cpu()
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

        return (log_prior + log_ratios).squeeze().cpu()


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
        random_variables = {"inputs": (2,), "outputs": (2002,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [128] * 3
        r = Class(activation=activation, trunk=trunk)
        super(ClassifierRatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, inputs, outputs, **kwargs):
        outputs = outputs.view(-1, 2002)
        return self._r.log_ratio(inputs=inputs, outputs=outputs, **kwargs)


### Datasets ###################################################################


class DatasetJointTrain(NamedDataset):
    def __init__(self, n=None):
        inputs = np.load("lotka_volterra/data/train/inputs.npy")
        outputs = np.load("lotka_volterra/data/train/outputs.npy")
        if (inputs.ndim != 2) or (outputs.ndim != 2):
            inputs = inputs.reshape(inputs.shape[0], -1)
            outputs = outputs.reshape(outputs.shape[0], -1)
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
        # inputs = NumpyDataset("lotka_volterra/data/validate/inputs.npy")
        # outputs = NumpyDataset("lotka_volterra/data/validate/outputs.npy")
        inputs = np.load("lotka_volterra/data/validate/inputs.npy")
        outputs = np.load("lotka_volterra/data/validate/outputs.npy")
        if (inputs.ndim != 2) or (outputs.ndim != 2):
            inputs = inputs.reshape(inputs.shape[0], -1)
            outputs = outputs.reshape(outputs.shape[0], -1)
        inputs = TensorDataset(torch.from_numpy(inputs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetJointValidate, self).__init__(
            inputs=inputs, outputs=outputs
        )


class DatasetJointTest(NamedDataset):
    def __init__(self):
        # inputs = NumpyDataset("lotka_volterra/data/test/inputs.npy")
        # outputs = NumpyDataset("lotka_volterra/data/test/outputs.npy")
        inputs = np.load("lotka_volterra/data/test/inputs.npy")
        outputs = np.load("lotka_volterra/data/test/outputs.npy")
        if (inputs.ndim != 2) or (outputs.ndim != 2):
            inputs = inputs.reshape(inputs.shape[0], -1)
            outputs = outputs.reshape(outputs.shape[0], -1)
        inputs = TensorDataset(torch.from_numpy(inputs))
        outputs = TensorDataset(torch.from_numpy(outputs))
        super(DatasetJointTest, self).__init__(inputs=inputs, outputs=outputs)


class DatasetJointValidateSmall(NamedDataset):
    def __init__(self):
        if not os.path.exists("lotka_volterra/data/validate/inputs_small.npy"):
            inputs = np.load("lotka_volterra/data/validate/inputs.npy")
            if inputs.shape[0] >= 10000:
                inputs = inputs[:10000]

            np.save("lotka_volterra/data/validate/inputs_small.npy", inputs)

        if not os.path.exists(
            "lotka_volterra/data/validate/outputs_small.npy"
        ):
            outputs = np.load("lotka_volterra/data/validate/outputs.npy")
            if outputs.shape[0] >= 10000:
                outputs = outputs[:10000]

            np.save("lotka_volterra/data/validate/outputs_small.npy", outputs)

        inputs = NumpyDataset("lotka_volterra/data/validate/inputs_small.npy")
        outputs = NumpyDataset(
            "lotka_volterra/data/validate/outputs_small.npy"
        )

        super(DatasetJointValidateSmall, self).__init__(
            inputs=inputs, outputs=outputs
        )


class DatasetJointTestSmall(NamedDataset):
    def __init__(self):
        if not os.path.exists("lotka_volterra/data/test/inputs_small.npy"):
            inputs = np.load("lotka_volterra/data/test/inputs.npy")
            if inputs.shape[0] >= 10000:
                inputs = inputs[:10000]

            np.save("lotka_volterra/data/test/inputs_small.npy", inputs)

        if not os.path.exists("lotka_volterra/data/test/outputs_small.npy"):
            outputs = np.load("lotka_volterra/data/test/outputs.npy")
            if outputs.shape[0] >= 10000:
                outputs = outputs[:10000]

            np.save("lotka_volterra/data/test/outputs_small.npy", outputs)

        inputs = NumpyDataset("lotka_volterra/data/test/inputs_small.npy")
        outputs = NumpyDataset("lotka_volterra/data/test/outputs_small.npy")

        super(DatasetJointTestSmall, self).__init__(
            inputs=inputs, outputs=outputs
        )
