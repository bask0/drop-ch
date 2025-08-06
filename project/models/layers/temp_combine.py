"""TemporalCombine Layer to combine dynamic and static features."""

import torch
from torch import Tensor


class TemporalCombine(torch.nn.Module):
    """
    Combine dynamic and static features.

    Shapes:
        x: The dynamic inputs, shape (batch, sequence, num_dynamic)
        s: The static inputs, shape (batch, num_static)

        output: A tensor of combined dynamic and static features, shape (batch, sequence, num_dynamic + num_static).

    """

    def __init__(self) -> None:
        """Initialize TemporalCombine layer."""
        super().__init__()

    def get_expand_arg(self, x: Tensor, s: Tensor) -> list[int]:
        """
        Compute the expanded shape for a tensor `s` to match `x` in all but last dimension.

        Args:
            x (Tensor): The input reference tensor.
            s (Tensor): The tensor to be expanded.

        Returns:
            list[int]: A list representing the expanded shape.
        """
        expand_size = list(x.shape)
        expand_size[-1] = s.shape[-1]

        return expand_size

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """
        Concatenates the input tensor `x` with the expanded tensor `s` along the last dimension.

        Args:
            x (Tensor): The main input tensor.
            s (Tensor): The secondary input tensor to be expanded and concatenated.

        Returns:
            Tensor: The concatenated tensor resulting from combining `x` and the expanded `s`.
        """
        s = s.unsqueeze(1).expand(*self.get_expand_arg(x=x, s=s))

        xs = torch.cat((x, s), dim=-1)

        return xs
