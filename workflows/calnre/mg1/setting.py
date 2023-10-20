import torch

from hypothesis.benchmark.mg1 import Prior as _Prior
from hypothesis.benchmark.mg1 import Simulator


class Prior(torch.nn.Module):
    def __init__(
        self,
        prior=_Prior(),
    ):
        super(Prior, self).__init__()
        self.low = torch.nn.Parameter(prior.low, requires_grad=False)
        self.high = torch.nn.Parameter(prior.high, requires_grad=False)

    def log_prob(self, x):
        return torch.distributions.Uniform(
            low=self.low, high=self.high, validate_args=False
        ).log_prob(x)

    def sample(self, sample_shape):
        return torch.distributions.Uniform(
            low=self.low, high=self.high, validate_args=False
        ).sample(sample_shape)

    def forward(self, x):
        return self.log_prob(x)


memory = "8GB"
ngpus = 1
batched = True
