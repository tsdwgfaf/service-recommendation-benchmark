import random
from typing import Any, Optional

import gensim
import torch.optim
from gensim import similarities
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from utils import *
from utils.loss_function import QARSLoss


class QARSModel(LightningModule):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        model_dir: str,
        model_name: str,
        dim_feature: int,
        hp_lam_alpha: float,
        hp_lam_beta: float,
        num_sims_api: int,
        lr: float,
    ):
        super(QARSModel, self).__init__()
        self.save_hyperparameters()

        dataset_dir = os.path.join(data_dir, data_name)
        mashup_des_path = os.path.join(dataset_dir, 'mashup_description.json')
        api_des_path = os.path.join(dataset_dir, 'api_description.json')
        model_path = os.path.join(os.path.join(os.path.join(data_dir, model_dir), model_name), 'model.wv')

        self.inter = torch.load(os.path.join(dataset_dir, 'user_item_interaction_matrix.pt'))

        with open(mashup_des_path, 'r', encoding='utf-8') as f:
            mashup_description = json.load(f)
        with open(api_des_path, 'r', encoding='utf-8') as f:
            api_description = json.load(f)
        dic = gensim.corpora.Dictionary(mashup_description + api_description)
        self.mashup_corpus = [dic.doc2bow(doc) for doc in mashup_description]
        self.api_corpus = [dic.doc2bow(doc) for doc in api_description]

        self.num_mashup = len(mashup_description)
        self.num_api = len(api_description)
        self.mashup_index = [i for i in range(self.num_mashup)]
        self.api_index = [i for i in range(self.num_api)]

        self.hdp_model: gensim.models.HdpModel = gensim.models.hdpmodel.HdpModel.load(model_path)
        num_topics = self.hdp_model.get_topics().shape[0]
        self.sims = similarities.MatrixSimilarity(self.hdp_model[self.api_corpus], num_features=num_topics)

        self.mashup_rectangular = nn.Parameter((torch.rand((self.num_mashup, dim_feature)) + 0.5).cuda())
        self.api_rectangular = nn.Parameter((torch.rand((self.num_api, dim_feature)) + 0.5).cuda())
        # self.mashup_rectangular = nn.Parameter(torch.rand((self.num_mashup, dim_feature)), requires_grad=True)
        # self.api_rectangular = nn.Parameter(torch.rand((self.num_api, dim_feature)), requires_grad=True)
        # self.w1 = nn.Parameter(torch.rand((num_mashup, num_api)), requires_grad=True)
        # self.w2 = nn.Parameter(torch.rand((num_mashup, num_api)), requires_grad=True)

        self.criterion = QARSLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> Any:
        return torch.matmul(self.mashup_rectangular, self.api_rectangular.t())

    def step(self, batch: Any):
        x = batch.squeeze()
        pred = self.forward(x)
        loss = self.criterion(pred, x.float())
        return loss, pred, x

    def predict(self):
        cand_mashups = random.sample(self.mashup_index, 64)
        pred = []
        for ma in cand_mashups:
            corpus = self.mashup_corpus[ma]
            api_sims = sorted(enumerate(self.sims[self.hdp_model[corpus]]), key=lambda item: -item[1])
            cand_apis = [api_sims[i][0] for i in range(self.hparams.num_sims_api)]
            v_m = self.mashup_rectangular[ma].unsqueeze(0)
            v_i = self.api_rectangular.unsqueeze(2)
            p = torch.matmul(v_m, v_i).squeeze()  # (num_api, )
            useless_apis = torch.tensor(list(set(self.api_index).difference(set(cand_apis))), dtype=torch.int64, device='cuda:0')
            pred.append(p.scatter(0, useless_apis, 0.).unsqueeze(0))
        pred = torch.cat(pred, dim=0)
        target = torch.cat([self.inter[i].unsqueeze(0) for i in cand_mashups], dim=0)
        return pred, target.int()

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, pred, target = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log('train/acc', self.acc(pred, target), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        # real_pred, real_target = self.predict()

        # return {
        #     'loss': loss,
        #     'preds': real_pred,
        #     'targets': real_target
        # }
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, pred, target = self.step(batch)
        # real_pred, real_target = self.predict()

        # return {
        #     'loss': loss,
        #     'preds': real_pred,
        #     'targets': real_target
        # }
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        # return torch.optim.Adam(
        #     params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        # )
        return torch.optim.Adam([
            {'params': self.mashup_rectangular, 'weight_decay': self.hparams.hp_lam_alpha},
            {'params': self.api_rectangular, 'weight_decay': self.hparams.hp_lam_beta}
        ], lr=self.hparams.lr)
