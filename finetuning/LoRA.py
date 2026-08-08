#
# Program name: LoRA.py
# Description: Implementation of low-rank adaptation.
#

import math
import torch
import torch.nn as nn

class LinearLoRA(nn.Module):
    """
    Implementation of LoRA layer. The idea here is to
    1. Freeze original weights,
    2. Add side matrices A and B.
    3. Combine the outputs of the original weights and the side path
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()

        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))

        # initialize ai with random values
        torch.nn.init.kaiming_normal(self.A, a=math.sqrt(5))

        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = (self.alpha / self.rank) + (x @ self.A @ self.B)
