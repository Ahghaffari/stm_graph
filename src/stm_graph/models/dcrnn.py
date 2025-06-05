import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN as DCRNNCell
from .base_model import BaseGNN


class EncoderDCRNN(nn.Module):
    """Encoder for the DCRNN model."""

    def __init__(self, in_channels, hidden_channels, num_layers=2, K=3, dropout=0.0):
        super(EncoderDCRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DCRNNCell(in_channels, hidden_channels, K))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(DCRNNCell(hidden_channels, hidden_channels, K))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index, edge_weight=None, hidden_states=None):
        """
        Forward pass through the encoder.
        """
        batch_size, num_nodes, seq_len, _ = x_seq.shape

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        for t in range(seq_len):
            x_t = x_seq[:, :, t, :]  # [batch_size, num_nodes, in_channels]

            for layer_idx in range(self.num_layers):
                layer_hidden_states = []
                for b in range(batch_size):
                    h = None
                    if hidden_states[layer_idx] is not None:
                        h = hidden_states[layer_idx][b]

                    h_new = self.dcrnn_cells[layer_idx](
                        x_t[b], edge_index, edge_weight, h
                    )

                    if layer_idx < self.num_layers - 1:
                        h_new = self.dropout_layer(h_new)

                    layer_hidden_states.append(h_new)

                hidden_states[layer_idx] = torch.stack(layer_hidden_states)

                if layer_idx < self.num_layers - 1:
                    x_t = hidden_states[layer_idx]

        return hidden_states


class DecoderDCRNN(nn.Module):
    """Decoder for the DCRNN model."""

    def __init__(self, hidden_channels, out_channels, num_layers=2, K=3, dropout=0.0):
        super(DecoderDCRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.dcrnn_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.dcrnn_cells.append(DCRNNCell(hidden_channels, hidden_channels, K))

        self.dropout_layer = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        edge_index,
        edge_weight=None,
        hidden_states=None,
        target_len=1,
        teacher_forcing_ratio=0.5,
        targets=None,
        go_input=None,
    ):
        """
        Forward pass through the decoder with scheduled sampling.
        """
        if hidden_states is None:
            raise ValueError("Hidden states from encoder are required for decoder")

        batch_size = hidden_states[0].shape[0]
        num_nodes = hidden_states[0].shape[1]
        device = hidden_states[0].device

        if go_input is None:
            current_input = torch.zeros(
                batch_size, num_nodes, self.hidden_channels, device=device
            )
        else:
            current_input = go_input

        for layer_idx in range(self.num_layers):
            layer_hidden_states = []
            for b in range(batch_size):
                h = hidden_states[layer_idx][b]

                h_new = self.dcrnn_cells[layer_idx](
                    current_input[b], edge_index, edge_weight, h
                )

                if layer_idx < self.num_layers - 1:
                    h_new = self.dropout_layer(h_new)

                layer_hidden_states.append(h_new)

            hidden_states[layer_idx] = torch.stack(layer_hidden_states)

            if layer_idx < self.num_layers - 1:
                current_input = hidden_states[layer_idx]

        decoder_output = hidden_states[-1]
        step_output = self.projection(decoder_output)

        # Return prediction with shape [batch_size, num_nodes, out_channels]
        return step_output


class RecurrentGCN_DCRNN(BaseGNN):
    """
    Diffusion Convolutional Recurrent Neural Network model
    """

    def __init__(self, in_channels, out_channels, hidden_dim=64, K=3, dropout=0.2):
        """
        Simplified initialization to match your original model
        """
        super(RecurrentGCN_DCRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_dim
        self.num_layers = 2

        self.encoder = EncoderDCRNN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=self.num_layers,
            K=K,
            dropout=dropout,
        )

        self.decoder = DecoderDCRNN(
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            num_layers=self.num_layers,
            K=K,
            dropout=dropout,
        )

        self.go_input = None
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Forward pass through the DCRNN model.

        Returns:
            (h_enc, y_pred): Tuple of encoder's final hidden state and prediction
                            where y_pred has shape [batch_size, num_nodes, out_channels]
        """
        if x.dim() == 3:  # [batch_size, num_nodes, in_channels]
            x = x.unsqueeze(
                2
            )  # Add time dimension [batch_size, num_nodes, 1, in_channels]

        encoder_hidden_states = self.encoder(x, edge_index, edge_weight)

        y_pred = self.decoder(
            edge_index=edge_index,
            edge_weight=edge_weight,
            hidden_states=encoder_hidden_states,
            target_len=1,
            teacher_forcing_ratio=0.0,
            targets=None,
            go_input=None,
        )

        return encoder_hidden_states[-1], y_pred
