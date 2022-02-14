import torch
from torch import nn


class APIPreferLinearNet(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_layer1: int,
        dim_layer2: int,
        dim_layer3: int,
    ):
        super(APIPreferLinearNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=dim_layer1),
            nn.ReLU(),
            nn.Linear(in_features=dim_layer1, out_features=dim_layer2),
            nn.ReLU(),
            nn.Linear(in_features=dim_layer2, out_features=dim_layer3),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=dim_layer3, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        out = self.hidden(x)
        out = self.output(out)
        return out
