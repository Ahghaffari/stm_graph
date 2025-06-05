import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch_geometric.nn import ChebConv
from .base_model import BaseGNN


class TemporalConvLayer(nn.Module):
    """Temporal convolutional layer with gating mechanism"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        # Create two parallel convolutional pathways for gating
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2)
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_nodes, channels]
        """
        # For temporal convolution: [batch_size, channels, num_nodes, time_steps]
        x_input = x.permute(0, 3, 2, 1)
        x_conv1 = self.conv1(x_input)
        x_conv2 = self.conv2(x_input)
        x_glu = x_conv1 * torch.sigmoid(x_conv2)
        x_bn = self.bn(x_glu)
        x_out = x_bn.permute(0, 3, 2, 1)

        return x_out


class SpatialGraphConvLayer(nn.Module):
    """Spatial graph convolutional layer"""

    def __init__(self, in_channels, out_channels, K=3):
        super(SpatialGraphConvLayer, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ChebConv(in_channels, out_channels, K)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_nodes, channels]
            edge_index: Edge indices of the graph
        """
        batch_size, time_steps, num_nodes, channels = x.shape
        output = []
        for t in range(time_steps):
            batch_t = []
            for b in range(batch_size):
                # Graph convolution on [num_nodes, channels]
                node_feats = x[b, t]
                out_feats = self.conv(node_feats, edge_index, edge_weight)
                out_feats = self.bn(out_feats)
                out_feats = F.relu(out_feats)
                batch_t.append(out_feats)
            time_output = torch.stack(batch_t, dim=0)
            output.append(time_output)

        # Stack all time steps: [batch_size, time_steps, num_nodes, out_channels]
        output = torch.stack(output, dim=1)

        return output


class STConvBlock(nn.Module):
    """Spatio-temporal convolutional block (T-G-T-N-D structure)"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size=3,
        K=3,
        dropout=0.2,
    ):
        super(STConvBlock, self).__init__()
        self.temp_conv1 = TemporalConvLayer(in_channels, hidden_channels, kernel_size)
        self.graph_conv = SpatialGraphConvLayer(hidden_channels, hidden_channels, K)
        self.temp_conv2 = TemporalConvLayer(hidden_channels, out_channels, kernel_size)
        self.layer_norm = nn.LayerNorm([out_channels])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_nodes, in_channels]
            edge_index: Edge indices of the graph
        """
        x = self.temp_conv1(x)
        x = self.graph_conv(x, edge_index, edge_weight)
        x = self.temp_conv2(x)

        batch_size, time_steps, num_nodes, channels = x.shape
        x = x.reshape(-1, channels)
        x = self.layer_norm(x)
        x = x.reshape(batch_size, time_steps, num_nodes, channels)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):
    """Output block for final prediction (T-N-F-F structure)"""

    def __init__(
        self, in_channels, hidden_channels, out_channels, kernel_size=3, dropout=0.2
    ):
        super(OutputBlock, self).__init__()
        self.temp_conv = TemporalConvLayer(in_channels, hidden_channels, kernel_size)
        self.layer_norm = nn.LayerNorm([hidden_channels])
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_nodes, in_channels]
        """

        x = self.temp_conv(x)
        batch_size, time_steps, num_nodes, channels = x.shape
        x = x.reshape(-1, channels)
        x = self.layer_norm(x)
        x = x.reshape(batch_size, time_steps, num_nodes, channels)
        x = x[:, -1]  # [batch_size, num_nodes, channels]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class STGCN(BaseGNN):
    def __init__(
        self,
        num_nodes,
        in_channels,
        hidden_channels=64,
        out_channels=1,
        K=3,
        kernel_size=3,
        dropout=0.2,
        num_st_blocks=2,
        history_window=3,
    ):
        """
        Spatio-Temporal Graph Convolutional Network

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features (1 for regression, >1 for classification)
            K: Order of Chebyshev polynomials
            kernel_size: Size of temporal convolution kernel
            dropout: Dropout rate
            num_st_blocks: Number of ST-Conv blocks
            history_window: Number of time steps in history (for reshaping 3D input)
        """
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.history_window = history_window
        self.in_channels = in_channels

        self.st_blocks = nn.ModuleList()

        self.st_blocks.append(
            STConvBlock(
                in_channels, hidden_channels, hidden_channels, kernel_size, K, dropout
            )
        )

        for _ in range(num_st_blocks - 1):
            self.st_blocks.append(
                STConvBlock(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    K,
                    dropout,
                )
            )

        self.output_block = OutputBlock(
            hidden_channels, hidden_channels, out_channels, kernel_size, dropout
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights with Glorot initialization"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:
                    init.xavier_uniform_(param)
                else:
                    init.uniform_(param)
            elif "bias" in name:
                init.constant_(param, 0)

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Forward pass

        Args:
            x: Input features with shape [batch_size, num_nodes, time_steps, in_channels]
            edge_index: Graph edge indices with shape [2, num_edges]
            edge_weight: Optional edge weights
            h: Not used, included for compatibility with interface

        Returns:
            Output predictions with shape [batch_size, num_nodes, out_channels]
        """
        input_dim = x.dim()

        if input_dim != 4:
            raise ValueError(
                f"Expected 4D input, got {input_dim}D tensor of shape {x.shape}"
            )

        # Permute dimensions for STGCN: [batch_size, time_steps, num_nodes, in_channels]
        x = x.permute(0, 2, 1, 3)

        for block in self.st_blocks:
            x = block(x, edge_index, edge_weight)

        out = self.output_block(x)  # [batch_size, num_nodes, out_channels]

        return h, out
