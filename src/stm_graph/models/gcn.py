import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .base_model import BaseGNN


class GCN(BaseGNN):
    """
    Graph Convolutional Network with support for both static and temporal data.
    Can process both 3D and 4D inputs.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.2,
        temporal_pooling="last",
    ):
        """
        Initialize GCN model.

        Args:
            in_channels: Number of input features
            hidden_channels: Size of hidden layers
            out_channels: Number of output features
            dropout: Dropout probability
            temporal_pooling: Method to pool temporal dimension ('last', 'mean', 'max')
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.temporal_pooling = temporal_pooling

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Forward pass through GCN.

        Args:
            x: Input features [batch_size, num_nodes, in_channels] or
               [batch_size, num_nodes, time_steps, in_channels]
            edge_index: Edge indices
            edge_weight: Optional edge weights
            h: Hidden state (not used, included for API compatibility)

        Returns:
            (h, out): Tuple of hidden state (None) and output
        """
        # Handle 4D input (with temporal dimension)
        if x.dim() == 4:  # [batch_size, num_nodes, time_steps, in_channels]
            batch_size, num_nodes, time_steps, _ = x.shape
            outputs = []

            for t in range(time_steps):
                x_t = x[:, :, t, :]  # [batch_size, num_nodes, in_channels]

                batch_out = []
                for b in range(batch_size):
                    node_feats = x_t[b]  # [num_nodes, in_channels]

                    out = F.relu(self.conv1(node_feats, edge_index, edge_weight))
                    out = self.dropout(out)
                    out = self.conv2(out, edge_index, edge_weight)
                    batch_out.append(out)

                batch_out = torch.stack(
                    batch_out
                )  # [batch_size, num_nodes, out_channels]
                outputs.append(batch_out)

            outputs = torch.stack(
                outputs, dim=2
            )  # [batch_size, num_nodes, time_steps, out_channels]

            if self.temporal_pooling == "last":
                return h, outputs[:, :, -1, :]  # Return last time step
            elif self.temporal_pooling == "mean":
                return h, torch.mean(outputs, dim=2)  # Average over time
            elif self.temporal_pooling == "max":
                return h, torch.max(outputs, dim=2)[0]  # Max over time
            else:
                return h, outputs[:, :, -1, :]  # Default to last

        # Handle 3D input (without temporal dimension)
        elif x.dim() == 3:  # [batch_size, num_nodes, in_channels]
            batch_size, num_nodes, _ = x.shape
            batch_out = []

            for b in range(batch_size):
                node_feats = x[b]  # [num_nodes, in_channels]

                out = F.relu(self.conv1(node_feats, edge_index, edge_weight))
                out = self.dropout(out)
                out = self.conv2(out, edge_index, edge_weight)
                batch_out.append(out)

            batch_out = torch.stack(batch_out)  # [batch_size, num_nodes, out_channels]
            return h, batch_out

        # Handle 2D input (single sample, no batch dimension)
        else:  # [num_nodes, in_channels]
            out = F.relu(self.conv1(x, edge_index, edge_weight))
            out = self.dropout(out)
            out = self.conv2(out, edge_index, edge_weight)
            return h, out.unsqueeze(0)
