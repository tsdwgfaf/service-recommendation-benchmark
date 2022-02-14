import os.path

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from typing import Any, Optional


class GloveBaseLineModel(LightningModule):
    def __init__(
        self,
        data_dir,
        data_name,
        api_feature_name: str,
        dim_mashup_feature,
        dim_api_feature,
        lr: float,
        weight_decay: float,
    ):
        super(GloveBaseLineModel, self).__init__()
        self.save_hyperparameters()

        dataset_dir = os.path.join(data_dir, data_name)
        self.api_feature = torch.load(os.path.join(dataset_dir, api_feature_name))  # (num_api, dim_api_feature)
        self.num_api = self.api_feature.size(0)

        # self.linear = nn.Sequential(
        #     nn.Linear(
        #         in_features=dim_mashup_feature + dim_api_feature,
        #         out_features=1
        #     ),
        # )
        self.linear = nn.Linear(dim_mashup_feature + dim_api_feature, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> Any:
        batch_size = x.size(0)
        v_m = x  # (batch_size, dim_mashup_feature)
        # v = torch.cat((
        #     v_m.unsqueeze(dim=1).repeat_interleave(self.num_api, dim=1),
        #     self.api_feature.unsqueeze(dim=0).repeat_interleave(batch_size, dim=0)
        # ), dim=2)
        # pred = self.linear(v).squeeze(2)
        # return pred

        pred = []
        for v_a in self.api_feature:
            v = torch.cat((v_m, v_a.unsqueeze(dim=0).repeat_interleave(batch_size, dim=0)), dim=1)
            pred.append(self.linear(v))
        pred = torch.cat(pred, dim=1)
        return pred

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
