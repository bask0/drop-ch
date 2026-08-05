"""Defines the EvaporationModel, a torch module with training routing for evaporation prediction."""

import torch
from torch import Tensor

from project.utils.unit_conversion import convert_mmd_to_m3s


class EvaporationModel(torch.nn.Module):
    """A simple evaporation model based on temperature, area, and water frequency.

    The model computes evaporation in cubic meters per second (m³/s) based on the
    temperature (T), area (in km²), and water frequency. The evaporation rate is
    calculated using a formula that incorporates an exponential factor derived from
    the temperature, a scaling factor for area, and a frequency adjustment.

    """

    def __init__(self, disable: bool = False, return_m3s: bool = True):
        """Initialize the EvaporationModel.

        Args:
            disable (bool): If True, the model will return zero evaporation regardless of inputs.
            return_m3s (bool): If True, the model returns evaporation in cubic meters per second (m³/s), else in
                millimeters per day (mm/d).

        """
        super().__init__()

        self.disable = disable
        self.return_m3s = return_m3s
        self.alpha = torch.nn.Parameter(torch.log(torch.tensor((4.0))))  # Leads to 4mm/d at 20C
        self.beta = torch.nn.Parameter(torch.tensor(1.5))

    def forward(self, T: Tensor, area: Tensor, water_freq: Tensor, subset: Tensor | None) -> Tensor:
        """Compute evaporation based on temperature, area, and water frequency.

        Args:
            T (Tensor): Temperature in degrees Celsius, shape (batch_size, time_steps).
            area (Tensor): Area in square kilometers, shape (batch_size,).
            water_freq (Tensor): Water frequency, shape (batch_size,).
            subset (Tensor | None): Optional subset of indices to apply the model to.

        Returns:
            Tensor: Evaporation in cubic meters per second (m³/s), shape (batch_size, time_steps).
        """

        if subset is not None:
            T = T[subset]
            area = area[subset]
            water_freq = water_freq[subset]

        if self.disable:
            return torch.zeros_like(T)

        alpha_exp = torch.exp(self.alpha)  # Exponentiate alpha to ensure it's positive
        beta_softplus = torch.nn.functional.softplus(self.beta)  # Ensure beta is positive
        T_20 = torch.clamp(T / 20, min=1e-2)  # Use reference of 20°C, set negative to a small positive value.
        e_mmd = alpha_exp * T_20**beta_softplus  # Evaporation in mm/d

        if not self.return_m3s:
            return e_mmd

        # TODO: Check if this conversion was correct.
        # e_m3s = e_mmd / 1000 * area_m2 * water_freq[:, None, None] / (24 * 60 * 60)  # Convert mm/d to m³/s

        e_m3s = convert_mmd_to_m3s(e_mmd, area[:, None, None] * water_freq[:, None, None])  # Convert mm/d to m³/s

        return e_m3s
