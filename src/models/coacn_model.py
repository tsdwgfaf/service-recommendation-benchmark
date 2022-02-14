from typing import Any, Optional, List

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from models.modules.coacn_net import COACNNet
from utils.loss_function import BPRLoss


class COACNModel(LightningModule):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_name_mashup_embedding: str,
        data_name_domain_embedding: str,
        data_name_api_embedding: str,
        data_name_inter_matrix: str,
        len_text: int,
        dim_word_embedding: int,
        dim_embedding: int,

        hp_beta: float,
        hp_lambda: float,
        hp_num_gcn_layer: int,
        hp_weight_gcn_layer: List[int],
        lr: float,
        weight_decay: float = 1e-5,
    ):
        super(COACNModel, self).__init__()
        self.save_hyperparameters()
        self.coacn_net = COACNNet(self.hparams)
        self.criterion = BPRLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> Any:
        return self.coacn_net(x)

    def step(self, batch: Any):
        x, target = batch
        pred = self.forward(x)
        loss = self.criterion(pred, target.float())
        return loss, pred, target

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, pred, target = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', self.acc(pred, target), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        self.log('val/acc', self.acc(pred, target), on_step=False, on_epoch=True, prog_bar=False)
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
