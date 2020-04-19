import torch


class Id(torch.nn.Module):
    def forward(self, x):
        return x