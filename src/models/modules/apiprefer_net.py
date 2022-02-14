from torch import nn


class APIPreferNet(nn.Module):
    def __init__(self, hparams: dict):
        super(APIPreferNet, self).__init__()
