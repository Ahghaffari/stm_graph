import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from .base_model import BaseGNN


class RecurrentGCN_AGCRN(BaseGNN):
    """
    Adaptive Graph Convolutional Recurrent Network.
    """

    def __init__(
        self,
        num_nodes,
        in_channels,
        hidden_dim=64,
        k=3,
        embedding_dimensions=8,
        out_channels=1,
    ):
        """
        Initialize AGCRN model.

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features per node
            hidden_dim: Size of hidden layers
            k: Number of Chebyshev filter taps
            embedding_dimensions: Size of node embedding
            out_channels: Number of output features
        """
        super(RecurrentGCN_AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.K = k
        self.embedding_dimensions = embedding_dimensions

        # Single shared node embeddings for both layers
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embedding_dimensions)
        )

        # Create two stacked AGCRN layers
        self.agcrn1 = AGCRN(
            number_of_nodes=self.num_nodes,
            in_channels=self.in_channels,
            out_channels=self.hidden_dim,
            K=self.K,
            embedding_dimensions=self.embedding_dimensions,
        )

        self.agcrn2 = AGCRN(
            number_of_nodes=self.num_nodes,
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            K=self.K,
            embedding_dimensions=self.embedding_dimensions,
        )

        # Final projection layer
        self.linear = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Forward pass through the AGCRN model.

        Args:
            x: Input features [batch_size, num_nodes, in_channels] or
               [batch_size, num_nodes, time_steps, in_channels]
            edge_index: Edge indices (not used in AGCRN but kept for API consistency)
            edge_weight: Optional edge weights (not used in AGCRN but kept for API consistency)
            h: Hidden state from previous step - expected to be a tensor not a tuple

        Returns:
            (h2, y_out): Single combined hidden state (h2) and output prediction
        """
        h1, h2 = None, None
        if h is not None:
            batch_size = h.size(0)
            num_nodes = h.size(1)
            h_size = self.hidden_dim
            h1 = torch.zeros(batch_size, num_nodes, h_size, device=h.device)
            h2 = h.clone()

        # Handle 4D input (with temporal dimension)
        if x.dim() == 4:
            batch_size, num_nodes, time_steps, _ = x.shape

            if num_nodes != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

            if h1 is None:
                h1 = torch.zeros(
                    batch_size, num_nodes, self.hidden_dim, device=x.device
                )
            if h2 is None:
                h2 = torch.zeros(
                    batch_size, num_nodes, self.hidden_dim, device=x.device
                )

            for t in range(time_steps):
                x_t = x[:, :, t, :]  # [batch_size, num_nodes, in_channels]

                h1 = self.agcrn1(x_t, self.node_embeddings, h1)

                h2 = self.agcrn2(h1, self.node_embeddings, h2)

            y_out = self.linear(h2)

            # Return only second layer's hidden state for training loop compatibility
            return h2, y_out

        # Handle 3D input (batch dimension but no temporal dimension)
        elif x.dim() == 3:  # [batch_size, num_nodes, in_channels]
            batch_size, num_nodes, _ = x.shape

            if num_nodes != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

            if h1 is None:
                h1 = torch.zeros(
                    batch_size, num_nodes, self.hidden_dim, device=x.device
                )
            if h2 is None:
                h2 = torch.zeros(
                    batch_size, num_nodes, self.hidden_dim, device=x.device
                )

            h1 = self.agcrn1(x, self.node_embeddings, h1)
            h2 = self.agcrn2(h1, self.node_embeddings, h2)
            y_out = self.linear(h2)

            return h2, y_out

        # Handle 2D input (single sample, no batch dimension)
        else:  # [num_nodes, in_channels]
            x = x.unsqueeze(0)  # [1, num_nodes, in_channels]

            if h1 is None:
                h1 = torch.zeros(1, self.num_nodes, self.hidden_dim, device=x.device)
            if h2 is None:
                h2 = torch.zeros(1, self.num_nodes, self.hidden_dim, device=x.device)

            h1 = self.agcrn1(x, self.node_embeddings, h1)
            h2 = self.agcrn2(h1, self.node_embeddings, h2)

            y_out = self.linear(h2)

            return h2.squeeze(0), y_out.squeeze(0)
