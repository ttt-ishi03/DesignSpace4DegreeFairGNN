import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomDegreeDiscriminator(nn.Module):
    def __init__(self):
        super(RandomDegreeDiscriminator, self).__init__()
        self.nclass = 3  # low-degree, medium-degree, high-degree

    def forward(self, x, _):
        return torch.rand(x.shape[0], self.nclass)