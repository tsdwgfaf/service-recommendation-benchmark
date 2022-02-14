from typing import Any, Optional

import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from utils import *
from utils.loss_function import T2L2Loss


class T2L2Model(LightningModule):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        dim_word: int,
        num_token: int,
        dim_mashup: int,
        dim_service: int,
        lr: float,
        weight_decay: float,
        pre_model: str,
    ):
        super(T2L2Model, self).__init__()
        self.pre_model = pre_model
        self.save_hyperparameters()

        dataset_dir = os.path.join(data_dir, data_name)
        service_embedding_path = os.path.join(dataset_dir, 'api_embedding_word2vec.pt')
        self.service_embedding: torch.Tensor = torch.load(service_embedding_path).cuda()
        num_service = self.service_embedding.size(0)
        self.service_embedding = self.service_embedding.view(num_service, -1)

        if pre_model == 'GloVe':
            self.mashup_linear = nn.Linear(in_features=num_token * dim_word, out_features=dim_mashup)
            self.service_linear = nn.Linear(in_features=num_token * dim_word, out_features=dim_service)

        self.linear1 = nn.Linear(in_features=dim_mashup, out_features=dim_service)
        self.linear2 = nn.Linear(in_features=dim_service, out_features=dim_service)
        self.sigmoid = nn.Sigmoid()
        self.linear3 = nn.Linear(in_features=2 * dim_service, out_features=1)

        self.criterion = T2L2Loss(dim=num_service)
        self.acc = torchmetrics.Accuracy()

    # def forward(self, x: torch.Tensor) -> Any:
    #     return self.sc_net(x)
    def forward(self, x: torch.Tensor, step: str) -> Any:
        batch_size = x.size(0)
        v_m = x.view(batch_size, -1)
        service = self.service_embedding
        if self.pre_model == 'GloVe':
            v_m = self.mashup_linear(v_m)
            service = self.service_linear(service)
        v_m = self.linear1(v_m)
        if step == 'train':
            v_m = self.linear2(v_m)  # (batch_size, dim_service)

        pred = []
        for v_s in service:  # (dim_service,)
            s = v_s + v_m
            y_head = self.linear3(torch.cat((v_m, s), dim=1))
            pred.append(y_head)
        pred = torch.cat(pred, dim=1)
        return pred

    # def step(self, batch: Any):
    #     x, target = batch
    #     pred = self.forward(x)
    #     loss = self.criterion(pred, target)
    #     return loss, pred, target

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, target = batch
        pred = self.forward(x, step='train')
        loss = self.criterion(pred, target)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', self.acc(pred, target), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, target = batch
        pred = self.forward(x, step='val')
        loss = self.criterion(pred, target)
        return {
            'loss': loss,
            'preds': pred,
            'targets': target
        }

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, target = batch
        pred = self.forward(x, step='test')
        loss = self.criterion(pred, target)
        return {
            'loss': loss,
            'preds': pred,
            'targets': target
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
