"""PyTorch utils."""

from typing import Literal, cast
import torch

ActivationLiteral = Literal[
    'relu', 'leakyrelu', 'selu', 'sigmoid', 'softplus',
    'tanh', 'identity', 'linear', 'none'
]


def get_activation(
        activation: ActivationLiteral | None
) -> torch.nn.Module:
    """Get PyTorch activation function by string query.

    Args:
        activation (str, None): Activation function, one of `relu`, `leakyrelu`, `selu`, `sigmoid`, `softplus`,
            `tanh`, `identity` (aka `linear`, `none`, or None).

    Raises:
        ValueError: if activation not found.

    Returns:
        PyTorch activation or None: activation function
    """

    if activation is None:
        return torch.nn.Identity()

    activation_str = cast(str, activation).lower()

    # Normalization aliases
    alias_map = {
        'none': 'identity',
        'linear': 'identity',
    }
    activation_str = alias_map.get(activation_str, activation_str)

    activations = {
        'relu': torch.nn.ReLU,
        'leakyrelu': torch.nn.LeakyReLU,
        'selu': torch.nn.SELU,
        'sigmoid': torch.nn.Sigmoid,
        'softplus': torch.nn.Softplus,
        'tanh': torch.nn.Tanh,
        'identity': torch.nn.Identity,
    }

    try:
        return activations[activation_str]()
    except KeyError:
        raise ValueError(
            f"Unknown activation '{activation}'. Available options: " + ", ".join(activations.keys() | alias_map.keys())
        )
