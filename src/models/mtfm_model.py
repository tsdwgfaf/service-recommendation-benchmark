from typing import Any, Optional
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from models.modules.sc_net import SCNet
from utils import *


class MTFMModel(LightningModule):

    def __init__(
        self,
        text_len: int,
        embedding_size: int,
        num_api: int,
        conv_kernel_size_tuple: List[int],
        conv_num_kernel: int,
        feature_dim: int,
        top_k_list: List[int],
        lr: float,
        weight_decay: float = 1e-5,
    ):
        super(MTFMModel, self).__init__()
        self.save_hyperparameters()
        self.sc_net = SCNet(self.hparams)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> Any:
        return self.sc_net(x)

    def step(self, batch: Any):
        x, target = batch
        pred = self.forward(x)
        loss = self.criterion(pred, target.float())
        return loss, pred, target

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, pred, target = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', self.acc(pred, target), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        return {
            'loss': loss,
            'preds': pred,
            'targets': target
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        return {
            'loss': loss,
            'preds': pred,
            'targets': target
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
