import json
import os.path
import queue
from typing import Any, Optional

import gensim.models
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from models.modules.apiprefer_linear_net import APIPreferLinearNet
from collections import Counter


class APIPreferModel(LightningModule):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        api_description_data_name: str,
        mashup_description_data_name: str,
        interaction_data_name: str,
        model_dir: str,
        model_name: str,
        num_similar_mashup: int,
        hp_epsilon: float,
        hp_eta: float,
        lr: float,
        weight_decay: float = 0.001,
    ):
        super(APIPreferModel, self).__init__()
        self.save_hyperparameters()

        num_topic = 200
        dim_polling = 10
        self.num_similar_mashup = 200

        dataset_dir = os.path.join(data_dir, data_name)
        api_path = os.path.join(dataset_dir, api_description_data_name)
        mashup_path = os.path.join(dataset_dir, mashup_description_data_name)
        inter_path = os.path.join(dataset_dir, interaction_data_name)
        model_path = os.path.join(os.path.join(data_dir, os.path.join(model_dir, model_name)), 'model.wv')

        self.inter: torch.Tensor = torch.load(inter_path)
        self.lda_model: gensim.models.ldamodel.LdaModel = gensim.models.ldamodel.LdaModel.load(model_path)

        with open(api_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        dic = gensim.corpora.Dictionary(corpus)
        doc_bow = [dic.doc2bow(doc) for doc in corpus]
        api_topics = list(self.lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        self.num_api = len(api_topics)
        self.api_embedding = torch.zeros((self.num_api, num_topic), dtype=torch.float32)  # (num_api, num_topic)
        for i in range(self.num_api):
            topic = api_topics[i]
            for j in range(len(topic)):
                self.api_embedding[i][j] = float(topic[j][1])

        with open(mashup_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        dic = gensim.corpora.Dictionary(corpus)
        doc_bow = [dic.doc2bow(doc) for doc in corpus]
        api_topics = list(self.lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        self.num_mashup = len(api_topics)
        self.mashup_embedding = torch.zeros((self.num_mashup, num_topic),
                                            dtype=torch.float32)  # (num_mashup, num_topic)
        for i in range(self.num_mashup):
            topic = api_topics[i]
            for j in range(len(topic)):
                self.mashup_embedding[i][j] = float(topic[j][1])

        self.mashup_embedding = self.mashup_embedding.cuda()
        self.api_embedding = self.api_embedding.cuda()

        self.weight = nn.Parameter(torch.rand(num_topic, num_topic), requires_grad=True)
        self.polling = nn.MaxPool2d(kernel_size=10)
        self.fc_net = APIPreferLinearNet(
            dim_in=int((num_topic / dim_polling) ** 2) + num_topic * 2,
            dim_layer1=200,
            dim_layer2=100,
            dim_layer3=20,
        )

        self.similarity = nn.CosineSimilarity(dim=2, eps=1e-8)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> Any:
        batch_size = x.size(0)

        # package_stat = torch.zeros((batch_size, self.num_api, self.num_api), dtype=torch.float32)
        # for k in range(batch_size):
        #     t_m = x[k]  # (num_topic, )
        #     for i in range(self.num_api):
        #         print(i)
        #         t_a1 = self.api_embedding[i]
        #         for j in range(i + 1):
        #             t_a2 = self.api_embedding[j]
        #             m_a = t_a1.unsqueeze(dim=1).matmul(t_a2.unsqueeze(dim=0))
        #             m_tf = m_a * self.weight
        #             m_tf = self.polling(m_tf.unsqueeze(dim=0))
        #             t_if = m_tf.squeeze().view(-1)
        #             t_a = torch.where(t_a1 > t_a2, t_a1, t_a2)
        #             t_cat = torch.cat((t_m, t_if, t_a), dim=0)
        #             stat = self.fc_net(t_cat).item()
        #             package_stat[k][i][j] = stat
        #             package_stat[k][j][i] = stat

        t_m = x
        # package_stat = torch.rand((batch_size, self.num_api, self.num_api))
        package_stat = []
        for i in range(self.num_api):
            t_a1 = self.api_embedding[i]  # (num_topic, )
            pack_stat = []
            for j in range(i + 1):
                t_a2 = self.api_embedding[j]
                m_a = t_a1.unsqueeze(dim=1).matmul(t_a2.unsqueeze(dim=0))
                m_tf = m_a * self.weight
                m_tf = self.polling(m_tf.unsqueeze(dim=0))
                t_if = m_tf.squeeze().view(-1)
                t_a = torch.where(t_a1 > t_a2, t_a1, t_a2)
                t_cat = torch.cat((t_m, t_if.repeat(batch_size, 1), t_a.repeat(batch_size, 1)), dim=1)
                stat = self.fc_net(t_cat)  # (batch_size, 1)
                pack_stat.append(stat.unsqueeze(dim=2))
            pack_stat.extend([torch.zeros((batch_size, 1, 1)).cuda() for i in range(self.num_api - i - 1)])
            package_stat.append(torch.cat(pack_stat, dim=2))  # (batch_size, 1, num_api)
        package_stat = torch.cat(package_stat, dim=1)
        for k in range(batch_size):
            for i in range(self.num_api):
                for j in range(i + 1, self.num_api):
                    package_stat[k][i][j] = package_stat[k][j][i]



        # select and sort the similar mashups
        sim = self.similarity(self.mashup_embedding.unsqueeze(dim=0), x.unsqueeze(dim=1))  # (batch_size, num_mashup)
        sim_index = sim.sort(dim=1)[1]
        sim_index = sim_index[:, :self.hparams.num_similar_mashup]
        pred = []
        for k in range(len(sim_index)):
            each_index = sim_index[k]
            pack_stat = package_stat[k]
            candidate_apis = []
            for mashup_idx in each_index:
                related_apis = self.inter[mashup_idx]
                for idx in range(len(related_apis)):
                    if related_apis[idx] == 1:
                        candidate_apis.append(idx)
            coun = Counter(candidate_apis)
            sum_call = len(candidate_apis)
            api_stat = {}
            for key, value in coun.items():
                api_stat[key] = float(value) / float(sum_call)
            can_api_sort = sorted(coun.items(), key=lambda d: d[1], reverse=True)
            api_index = [can_api_sort[i][0] for i in range(len(can_api_sort))]
            # create graph for candidate apis
            num_can_api = len(can_api_sort)
            g = torch.zeros((num_can_api, num_can_api), dtype=torch.int32)  # (num_can_api, num_can_api)
            for i in range(num_can_api):
                for j in range(i + 1):
                    idx1 = api_index[i]
                    idx2 = api_index[2]
                    if pack_stat[idx1][idx2] > max(api_stat[idx1], api_stat[idx2], self.hparams.hp_epsilon):
                        g[i][j] = 1
                        g[j][i] = 1

            # discover the api seed
            api_seed = None
            for i in range(num_can_api):
                if g[i].sum() > 0:
                    api_seed = i
                    break
                if api_stat[api_index[i]] > self.hparams.hp_eta:
                    api_seed = i
                    break

            if api_seed is None:
                pred.append(torch.zeros((1, self.num_api), dtype=torch.float32))
                continue
            # discover the fully connected sub-graph
            q = queue.Queue(num_can_api)
            q.put(api_seed)
            effective_apis = []
            is_visit = [False for i in range(num_can_api)]
            is_visit[api_seed] = True
            while not q.empty():
                cur = q.get()
                is_effective = True
                for eff_api in effective_apis:
                    if g[cur][eff_api] == 0:
                        is_effective = False
                if is_effective:
                    effective_apis.append(cur)
                for i in range(num_can_api):
                    if g[cur][i] == 1 and not is_visit[i]:
                        is_visit[i] = True
                        q.put(i)
            effective_apis = list(map(lambda d: api_index[d], effective_apis))
            pred.append(
                torch.zeros((self.num_api,), dtype=torch.float32)
                    .scatter(0, torch.tensor(effective_apis, dtype=torch.int64), 1).unsqueeze(dim=0)
            )

        return torch.cat(pred, dim=0).cuda()

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


if __name__ == '__main__':
    a = torch.tensor([1.11])
    print(a.item())