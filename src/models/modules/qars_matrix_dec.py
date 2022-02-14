from torch import nn


class MatrixDec(nn.Module):
    def __init__(self, hparams: dict):
        super(MatrixDec, self).__init__()

