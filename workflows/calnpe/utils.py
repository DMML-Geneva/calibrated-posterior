from collections import OrderedDict

import torch
from torch import nn
from torch.nn.functional import threshold


class OppositeThreshold(nn.Module):
    r"""Thresholds each element of the input Tensor.

    OpositeThreshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x < \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = OppositeThreshold(0.1, 20)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ["threshold", "value", "inplace"]

    threshold: float
    value: float
    inplace: bool

    def __init__(
        self, threshold: float, value: float = None, inplace: bool = False
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value if value is not None else threshold
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return -threshold(-input, -self.threshold, -self.value, self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return "threshold={}, value={}{}".format(
            self.threshold, self.value, inplace_str
        )


def build_embedding(problem):
    """ "
    Embedding network for flow conditioning.

    Args:
        problem (str) -- Which problem is run. Options: 'gw' |
            'lotka_volterra' | 'mg1' | 'slcp' | 'spatialsir' | 'weinberg'

    Returns:
        nn.Sequential
    """
    assert problem in [
        "gw",
        "lotka_volterra",
        "mg1",
        "slcp",
        "spatialsir",
        "weinberg",
    ]

    if problem == "gw":
        nb_channels = 16
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
        cnn.append(nn.Flatten())
        return nn.Sequential(*cnn)
    elif problem == "lotka_volterra":
        hidden = 64
        latent = 10
        return torch.nn.Sequential(
            torch.nn.Linear(2002, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, latent),
        )
    elif problem == "mg1":
        hidden = 64
        latent = 10
        return torch.nn.Sequential(
            OppositeThreshold(threshold=200),
            torch.nn.Linear(5, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, latent),
        )
    elif problem == "slcp":
        hidden = 64
        latent = 10
        return torch.nn.Sequential(
            torch.nn.Linear(8, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, latent),
        )
    elif problem == "spatialsir":
        hidden = 64
        latent = 10
        return torch.nn.Sequential(
            torch.nn.Linear(7500, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, latent),
        )
    elif problem == "weinberg":
        hidden = 64
        latent = 10
        return torch.nn.Sequential(
            torch.nn.Linear(20, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SELU(),
            torch.nn.Linear(hidden, latent),
        )


def load_checkpoint(model, checkpoint, load_cp_continue, optimizer=None):
    """
    Load an existing checkpoint of the model to continue training or for
    evaluation.

    :param torch.nn.Module model: Model loaded to continue training or for
        evaluation
    :param checkpoint: Checkpoint for continuing to train or for
        evaluation
    :type checkpoint: dict or OrderedDict
    :param bool load_cp_continue: Whether the checkpoint is loaded to continue
        training
    :param optimizer: Optimizer that was used
    :type optimizer: torch.optim or None
    """
    if isinstance(checkpoint, OrderedDict):
        model.load_state_dict(state_dict=checkpoint)
    else:
        model.load_state_dict(state_dict=checkpoint["state_dict"])

    if load_cp_continue:
        assert optimizer is not None, (
            f"When checkpoint is loaded to continue training, optimizer "
            f"cannot  be {optimizer}"
        )
        optimizer.load_state_dict(state_dict=checkpoint["optimizer"])
    print("--> The checkpoint of the model is being loaded")
