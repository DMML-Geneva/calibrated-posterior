import glob
import hypothesis as h
import numpy as np
import torch
import os
import math
from matplotlib import pyplot as plt

from tqdm import tqdm

from hypothesis.benchmark.lotka_volterra_small import Prior
from hypothesis.nn import build_ratio_estimator
from hypothesis.nn.ratio_estimation import BaseRatioEstimator
from hypothesis.nn.ratio_estimation import RatioEstimatorEnsemble
from hypothesis.stat import highest_density_level
from hypothesis.util.data import NamedDataset
from hypothesis.util.data import NumpyDataset
from torch.utils.data import TensorDataset
import torch.nn as nn


prior = Prior()


extent = [  # I know, this isn't very nice :(
    prior.low[0].item(),
    prior.high[0].item(),
    prior.low[1].item(),
    prior.high[1].item(),
]


@torch.no_grad()
def load_estimator(query):
    paths = glob.glob(query)
    if len(paths) == 1:
        estimator = RatioEstimator()
        estimator.load_state_dict(torch.load(query))
    else:
        estimators = []
        for path in paths:
            estimators.append(load_estimator(path))
        estimator = RatioEstimatorEnsemble(estimators, reduce="ratio_mean")
    estimator = estimator.to(h.accelerator)
    estimator.eval()

    return estimator


@torch.no_grad()
def compute_log_posterior(r, observable, resolution=100):
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
    log_prior_probabilities = prior.log_prob(inputs).view(-1, 1)
    observables = observable.repeat(resolution**2, 1, 1).float()
    observables = observables.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=observables)
    log_posterior = (
        (log_prior_probabilities + log_ratios)
        .view(resolution, resolution)
        .cpu()
    )

    return log_posterior, p1.cpu(), p2.cpu()


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs, flow_sbi=False):
    inputs = inputs.to(h.accelerator)
    outputs = outputs.to(h.accelerator)
    log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
    log_prior = prior.log_prob(inputs)

    return (log_prior + log_ratios).squeeze()


def plot_posterior(p1, p2, pdf, nominal, mean_1, mean_2, index, outputdir):
    p1 = p1.cpu()
    p2 = p2.cpu()
    pdf = pdf.cpu()
    nominal = nominal.cpu()
    mean_1 = mean_1.cpu()
    mean_2 = mean_2.cpu()
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    plt.pcolormesh(
        g1, g2, pdf, antialiased=True, edgecolors="face", shading="auto"
    )
    plt.set_cmap("viridis_r")
    plt.colorbar()
    plt.plot(nominal[0, 0], nominal[0, 1], "*", color="k")
    plt.hlines(mean_2, extent[0], extent[1])
    plt.vlines(mean_1, extent[2], extent[3])

    plt.savefig(outputdir + "/posterior_{}.pdf".format(index))
    plt.close()


class RatioEstimator(BaseRatioEstimator):
    def __init__(self):
        super(RatioEstimator, self).__init__(
            denominator="inputs|outputs",
            random_variables={"inputs": (2,), "outputs": (2002,)},
        )

        nb_channels = 8
        nb_conv_layers = 8
        shrink_every = 2
        final_shape = 1001
        for i in range(nb_conv_layers):
            if i % shrink_every == 0:
                final_shape = math.floor((final_shape - 1) / 2 + 1)
            else:
                final_shape = final_shape

        print("final shape = {}".format(final_shape))
        fc_layers = [2 + nb_channels * final_shape, 128, 128, 128, 1]

        cnn = [
            nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1)
        ]

        for i in range(nb_conv_layers):
            if i % shrink_every == 0:
                stride = 2
            else:
                stride = 1

            cnn.append(
                nn.Conv1d(
                    in_channels=nb_channels,
                    out_channels=nb_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            cnn.append(nn.SELU())
            cnn.append(nn.MaxPool1d(3, stride=stride, padding=1))

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
        outputs = outputs / 100
        outputs = outputs.permute((0, 2, 1))
        features = self.features(outputs).view(outputs.shape[0], -1)
        concat = torch.cat((features, inputs), 1)
        return self.fc(concat)


"""


class RatioEstimator(BaseRatioEstimator):

    def __init__(self):
        random_variables = {"inputs": (2,), "outputs": (2002,)}
        Class = build_ratio_estimator("mlp", random_variables)
        activation = torch.nn.SELU
        trunk = [128] * 3
        r = Class(activation=activation, trunk=trunk)
        super(RatioEstimator, self).__init__(r=r)
        self._r = r

    def log_ratio(self, inputs, outputs, **kwargs):
        outputs = outputs / 100
        return self._r.log_ratio(inputs=inputs, outputs=outputs, **kwargs)
"""


class DatasetJointTrain(NamedDataset):
    def __init__(self, n=None):
        inputs = np.load("lotka_volterra/data/train/inputs.npy")
        outputs = np.load("lotka_volterra/data/train/outputs.npy")
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
        inputs = NumpyDataset("lotka_volterra/data/validate/inputs.npy")
        outputs = NumpyDataset("lotka_volterra/data/validate/outputs.npy")
        super(DatasetJointValidate, self).__init__(
            inputs=inputs, outputs=outputs
        )


class DatasetJointTest(NamedDataset):
    def __init__(self):
        inputs = NumpyDataset("lotka_volterra/data/test/inputs.npy")
        outputs = NumpyDataset("lotka_volterra/data/test/outputs.npy")
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
