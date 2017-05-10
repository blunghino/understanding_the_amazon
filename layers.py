"""
custom pytorch layers
"""
import torch.nn as nn


class Flatten(nn.Module):
    """
    flatten a 4d tensor into a 2d tensor

    from cs231n assignment 2
    """
    def forward(self, x):
        N, C, H, W = x.size() 
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)  