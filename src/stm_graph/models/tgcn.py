from torch_geometric_temporal.nn.recurrent import TGCN2
from .base_model import BaseGNN
import torch
import torch.nn as nn


class TGCN(BaseGNN):
    def __init__(self, in_channels, out_channels, batch_size=1):
        super().__init__()
        self.tgcn = TGCN2(in_channels, out_channels, batch_size=batch_size)
        self.out_channels = out_channels
        self.batch_size = batch_size

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Forward pass through the TGCN model.

        Args:
            x: Input features [batch_size, num_nodes, in_channels]
            edge_index: Edge indices
            edge_weight: Optional edge weights
            h: Hidden state from previous step

        Returns:
            (h, out): Updated hidden state and output
        """

        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, num_nodes, in_channels]

        batch_size, num_nodes = x.size(0), x.size(1)

        if h is not None and (h.size(0) != batch_size or h.size(1) != num_nodes):
            if batch_size < h.size(0):
                h = h[:batch_size]
            else:
                h = torch.zeros(
                    batch_size, num_nodes, self.out_channels, device=x.device
                )

        h = self.tgcn(x, edge_index, edge_weight, h)

        return h, h
