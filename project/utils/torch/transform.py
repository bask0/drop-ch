"""Transform layer for PyTorch."""

from typing import Callable
import torch
from torch import Tensor


class TransformLayer(torch.nn.Module):
    """
    Transform layer for PyTorch.

    Turn functions into layers.
    """

    def __init__(self, transform_fun: Callable, name: str = "Transform()") -> None:
        """Transform layer, applies `transform_fun`.

        Example:
            >>> import torch
            >>> reshape_fun = lambda x: x.flatten(start_dim=1)
            >>> flatten_layer = Transform(reshape_fun)
            >>> flatten_layer(torch.ones(2, 3, 4)).shape
            torch.Size([2, 12])

        Args:
            transform_fun (Callable): thetransform function.
        """
        super(TransformLayer, self).__init__()

        self.transform_fun = transform_fun
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the transformation function to the input tensor.

        Args:
            x (Tensor): Input tensor to be transformed.

        Returns:
            Tensor: Transformed tensor after applying the transformation function.
        """
        return self.transform_fun(x)

    def __repr__(self) -> str:
        """
        Return a string representation of the object, typically its name attribute.

        Returns:
            str: The name of the object.
        """
        s = f"{self.name}"
        return s
