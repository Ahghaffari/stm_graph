import torch.nn as nn


class BaseGNN(nn.Module):
    """Base class for all GNN models"""

    def __init__(self):
        super(BaseGNN, self).__init__()

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Standard interface for GNN forward pass.

        Args:
            x: Input features [batch_size, num_nodes, in_channels] or
               [batch_size, num_nodes, time_steps, in_channels]
            edge_index: Edge indices
            edge_weight: Optional edge weights
            h: Hidden state from previous step (for recurrent models)

        Returns:
            (h, out): Tuple of updated hidden state and output
        """
        raise NotImplementedError("Subclasses must implement forward method")
