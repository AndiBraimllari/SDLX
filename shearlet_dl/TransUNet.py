import torch
import torch.nn as nn
import torch.nn.functional as F


# this file contains the TransUNet architecture


class TransUNet(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2102.04306.pdf
    """

    def __init__(self, num_of_subbands):
        super().__init__()

    def forward(self, x):
        return x
