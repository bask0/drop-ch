"""Defines an LSTM with pre-fusion of static inputs."""

import torch
from torch import Tensor


class EncodingModule(torch.nn.Module):
    """Implements an encoding module based on Conv1D layers.

    Each hidden layer consists of a linear, dropout and an activation layer. The last layer does not have
    dropout. The hidden size is `hidden_size_factor` times the encoding size (num_enc).

    Shapes:
        - Input tensor: (B, L, I)
        - Output tensor: (B, L, E)
        ...where B is the batch size, I the input size, E the encoding size, and L the sequence lengths.

    """

    def __init__(
            self,
            num_in: int,
            num_enc: int,
            num_layers: int,
            dropout: float,
            hidden_size_factor: int = 1,
            activation: torch.nn.Module | None = None,
            activation_last: torch.nn.Module | None = None) -> None:
        """Initialize EncodingModule.

        Args:
            num_in (int): the input channel size (C_in).
            num_enc (int): the encoding size (C_out).
            num_layers (int): the number of layers. Note that num_layers=1 is a linear transformation with
                activation_last applied afterwards.
            dropout (float): the dropout applied to all but the output layer.
            unsqueeze_input (bool): whether to unsqueeze the input.
            hidden_size_factor (int): the hidden size as a factor of the encoding size (`num_enc`);
                `hidden_size = hidden_size_factor * num_enc`. The default is 1.
            activation (torch.nn.Module, optional): The activation of hidden layers.
                Defaults to None -> torch.nn.Sigmoid().
            activation_last (torch.nn.Module, optional): the activation of the output layer.
                Defaults to None -> torch.nn.Identity().
        """

        super().__init__()

        num_hidden = num_enc * hidden_size_factor

        if activation is None:
            activation = torch.nn.Sigmoid()
        if activation_last is None:
            activation_last = torch.nn.Identity()

        layers: list[torch.nn.Module] = []

        for layer in range(num_layers):
            is_first = layer == 0
            is_last = layer == num_layers - 1

            conv = torch.nn.Linear(
                in_features=num_in if is_first else num_hidden,
                out_features=num_enc if is_last else num_hidden,
            )
            layers.append(conv)

            if is_last:
                layers.append(activation_last)
            else:
                layers.append(torch.nn.Dropout(dropout))
                layers.append(activation)

        self.encoding = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Model forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, ..., I), where B is the batch size and I is the input size.

        Returns:
            Tensor: Encoded tensor of shape (B, ..., E), where E is the encoding size.
        """
        return self.encoding(x)


class VariationalDropout(torch.nn.Module):
    """Apply the same dropout mask across the temporal dimension.

    See https://arxiv.org/abs/1512.05287 for more details.

    CODE SOURCE: https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
    AUTHOR: Keita Kurita
    ORIGINAL LICENSE: MIT License

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

    """

    def __init__(self, dropout: float, batch_first: bool = False) -> None:
        """Initialize VariationalDropout.

        Args:
            dropout (float): Dropout rate to apply.
            batch_first (bool): If True, input tensors are expected to have shape (B, T, C) where
                B is the batch size, T is the sequence length, and C is the number of features.
                If False, input tensors are expected to have shape (T, B, C).
        """

        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) or (T, B, C) if `batch_first=False`.

        Returns:
            torch.Tensor: Output tensor with the same shape as input, with dropout applied.
        """

        if not self.training or self.dropout <= 0.:
            return x

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)

        x = x * m / (1 - self.dropout)

        return x


class LSTM(torch.nn.Module):
    """
    LSTM module with static and dynamic input encodings.

    This module uses separate encoders for static and dynamic inputs, processes the
    temporally encoded dynamic inputs with an LSTM, and outputs predictions through
    a fully connected layer with an optional activation.

    The static encoding is broadcast and added to the dynamic encoding at each timestep and follows the
    approach `prefusion with encoding` described in this paper:
    https://hess.copernicus.org/articles/29/1061/2025/

    """

    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            hidden_size: int = 32,
            lstm_layers: int = 1,
            output_size: int = 1,
            enc_dropout: float = 0.0,
            variational_dropout: float = 0.0,
            **kwargs) -> None:
        """
        Initialize the LSTM model.

        Args:
            num_static_in (int): Number of static input features.
            num_dynamic_in (int): Number of dynamic input features (per timestep).
            hidden_size (int): Size of hidden representations throughout the model.
            lstm_layers (int): Number of stacked LSTM layers.
            output_size (int): Output feature size per timestep.
            enc_dropout (float): Dropout rate used in both encoding modules.
            variational_dropout (float): Dropout rate used on dynamic encodings (variational).
        """

        super().__init__(**kwargs)

        self.model_type = 'LSTM'

        # Static input encoding: encodes static features into hidden representation
        self.static_encoding = EncodingModule(
            num_in=num_static_in,
            num_enc=hidden_size,
            num_layers=2,
            dropout=0,  # Done separately.
            activation=torch.nn.ReLU(),
            activation_last=torch.nn.Sigmoid()
        )

        self.static_dropout = torch.nn.Dropout(
            p=enc_dropout
        )

        # Dynamic input encoding: encodes dynamic inputs for each timestep
        self.dynamic_encoding = EncodingModule(
            num_in=num_dynamic_in,
            num_enc=hidden_size,
            num_layers=2,
            dropout=0,  # Done separately.
            activation=torch.nn.ReLU(),
            activation_last=torch.nn.Tanh()
        )

        self.variational_dropout = VariationalDropout(
            dropout=variational_dropout,
            batch_first=True
        )

        # Temporal modeling with LSTM
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Output projection layer
        self.out_layer = torch.nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

        # Optional activation function applied to outputs
        self.output_activation = torch.nn.Identity()

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Dynamic input tensor of shape (B, S, D), where
                B = batch size, S = sequence length, D = dynamic feature count.
            s (Tensor): Static input tensor of shape (B, C), where
                B = batch size, C = static feature count.

        Returns:
            Tensor: Output predictions of shape (B, S, O), where
                B = batch size, S = sequence length, O = output size (usually 1).
        """

        # Encode dynamic inputs: (B, S, D) -> (B, S, E)
        d_enc = self.dynamic_encoding(x)
        d_enc = self.variational_dropout(d_enc)

        if s is not None:
            # Encode static inputs and broadcast: (B, C) -> (B, 1, E)
            s_enc = self.static_encoding(s).unsqueeze(1)
            s_enc = self.static_dropout(s_enc)

            # Combine static and dynamic encodings with broadcasting
            enc = d_enc + s_enc

        # Pass through LSTM: (B, S, E) -> (B, S, E)
        out, _ = self.lstm(enc)

        # Apply dropout and output projection: (B, S, E) -> (B, S, O)
        out = self.out_layer(out)

        # Apply final activation
        return self.output_activation(out)
