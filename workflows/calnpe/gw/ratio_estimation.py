r"""CASBI: Conservative Amortized Simulation-Based Inference

Contains the definition of the ratio estimators
and the corresponding utilities.
"""

import glob
import torch
import numpy as np
import torch.nn as nn
import hypothesis as h

from tqdm import tqdm
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from hypothesis.stat import highest_density_level
from torch.utils.data import Dataset, TensorDataset
from hypothesis.benchmark.gravitational_waves import Prior
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble

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
def compute_log_posterior(
    r, observable, resolution=100, batch_size=64, flow_sbi=False
):
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
        log_posterior = torch.empty(resolution**2)

        for b in range(0, inputs.shape[0], batch_size):
            cur_inputs = inputs[b : b + batch_size]
            log_posterior[b : b + batch_size] = r.log_prob(
                cur_inputs, x=observable, norm_posterior=False
            )[0]

        log_posterior = log_posterior.view(resolution, resolution).cpu()

    else:
        log_prior_probabilities = prior.log_prob(inputs).flatten()

        log_ratios = torch.empty(resolution**2)

        for b in range(0, inputs.shape[0], batch_size):
            cur_inputs = inputs[b : b + batch_size]
            observables = observable.repeat(cur_inputs.shape[0], 1, 1).float()
            observables = observables.to(h.accelerator)
            log_ratios[b : b + batch_size] = r.log_ratio(
                inputs=cur_inputs, outputs=observables
            ).squeeze(1)

        log_prior_probabilities = log_prior_probabilities.cpu()
        log_ratios = log_ratios.cpu()

        log_posterior = (
            (log_prior_probabilities + log_ratios)
            .view(resolution, resolution)
            .cpu()
        )

    return log_posterior


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False, batch_size=32):
    inputs = inputs.to(h.accelerator)
    # outputs = outputs.to(h.accelerator)
    with torch.no_grad():
        if flow_sbi:
            log_posterior = torch.Tensor(
                [
                    r.log_prob(
                        theta.to(h.accelerator),
                        x=x.unsqueeze(0).to(h.accelerator),
                        norm_posterior=False,
                    )
                    for theta, x in zip(inputs, outputs)
                ]
            )
            return log_posterior.squeeze()
        else:
            log_ratios = torch.empty(len(inputs))

            for b in range(0, inputs.shape[0], batch_size):
                cur_inputs = inputs[b : b + batch_size]
                cur_outputs = outputs[b : b + batch_size]
                cur_outputs = cur_outputs.to(h.accelerator)
                log_ratios[b : b + batch_size] = r.log_ratio(
                    inputs=cur_inputs, outputs=cur_outputs
                ).squeeze(1)

            log_prior = prior.log_prob(inputs).flatten().detach().cpu()
            log_ratios = log_ratios.detach().cpu()

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
        super(ClassifierRatioEstimator, self).__init__(
            denominator="inputs|outputs",
            random_variables={"inputs": (2,), "outputs": (2, 8192)},
        )

        nb_channels = 16
        fc_layers = [nb_channels + 2, 128, 128, 128, 1]

        cnn = [
            nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1)
        ]

        for i in range(13):
            cnn.append(
                nn.Conv1d(
                    in_channels=nb_channels,
                    out_channels=nb_channels,
                    kernel_size=2,
                    dilation=2**i,
                )
            )
            cnn.append(nn.SELU())

        self.features = nn.Sequential(*cnn)
        fc = []
        for i in range(len(fc_layers) - 1):
            fc.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            fc.append(nn.SELU())

        fc.pop()
        self.fc = nn.Sequential(*fc)

        self.features.type(torch.float32)
        self.fc.type(torch.float32)

    def log_ratio(self, inputs, outputs, **kwargs):
        inputs = inputs.type(torch.float32)
        outputs = outputs.type(torch.float32)
        features = self.features(outputs).view(outputs.shape[0], -1)
        concat = torch.cat((features, inputs), 1)
        return self.fc(concat)


class FlowRatioEstimator(BaseRatioEstimator):
    # Need change
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
        inputs = np.load("gw/data/train/inputs.npy")
        outputs = np.load("gw/data/train/outputs.npy")
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
        inputs = NumpyDataset("gw/data/validate/inputs.npy")
        outputs = NumpyDataset("gw/data/validate/outputs.npy")
        super(DatasetJointValidate, self).__init__(
            inputs=inputs, outputs=outputs
        )


class DatasetJointTest(NamedDataset):
    def __init__(self):
        inputs = NumpyDataset("gw/data/test/inputs.npy")
        outputs = NumpyDataset("gw/data/test/outputs.npy")
        super(DatasetJointTest, self).__init__(inputs=inputs, outputs=outputs)
