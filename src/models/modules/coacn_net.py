import json
import os

import numpy as np
import torch
from torch import nn


class COACNNet(nn.Module):
    def __init__(self, hparams: dict):
        super(COACNNet, self).__init__()
        self.dim_word_embedding = hparams['dim_word_embedding']
        self.dim_embedding = hparams['dim_embedding']
        # self.num_domain_token = hparams['num_domain_token']
        self.beta = hparams['hp_beta']
        self.num_gcn_layer = hparams['hp_num_gcn_layer']
        self.weight_gcn_layer = hparams['hp_weight_gcn_layer']

        data_dir = os.path.join(hparams['data_dir'], hparams['data_name'])
        data_path_mashup = os.path.join(data_dir, hparams['data_name_mashup_embedding'])
        data_path_domain = os.path.join(data_dir, hparams['data_name_domain_embedding'])
        data_path_api = os.path.join(data_dir, hparams['data_name_api_embedding'])

        self.mashup_embedding = torch.load(data_path_mashup)
        self.mashup_embedding = torch.transpose(self.mashup_embedding, 1, 2)
        # (num_mashup, dim_word_embedding, num_mashup_token)
        self.num_mashup = self.mashup_embedding.size(0)
        self.num_mashup_token = self.mashup_embedding.size(2)
        self.mashup_embedding = self.mashup_embedding.contiguous().view(self.num_mashup, -1)
        self.mashup_embedding = self.mashup_embedding.cuda()

        self.domain_embedding = torch.load(data_path_domain)
        self.domain_embedding = torch.transpose(self.domain_embedding, 1, 2)
        # (num_domain, dim_word_embedding, num_mashup_token)
        self.num_domain = self.domain_embedding.size(0)
        self.num_domain_token = self.domain_embedding.size(2)
        self.domain_embedding = self.domain_embedding.contiguous().view(self.num_domain, -1)
        self.domain_embedding = self.domain_embedding.cuda()

        self.api_embedding = torch.load(data_path_api)
        self.api_embedding = torch.transpose(self.api_embedding, 1, 2)
        # (num_api, dim_word_embedding, num_api_token)
        self.num_api = self.api_embedding.size(0)
        self.num_api_token = self.api_embedding.size(2)
        self.api_embedding = self.api_embedding.contiguous().view(self.num_api, -1)
        self.api_embedding = self.api_embedding.cuda()

        data_path_inter_matrix = os.path.join(data_dir, hparams['data_name_inter_matrix'])
        self.inter_matrix = torch.load(data_path_inter_matrix)

        A_1 = torch.cat((torch.zeros((self.num_mashup, self.num_mashup), dtype=torch.float32), self.inter_matrix),
                        dim=1)
        A_2 = torch.cat((torch.transpose(self.inter_matrix, 0, 1), torch.zeros((self.num_api, self.num_api))), dim=1)
        self.A = torch.cat((A_1, A_2), dim=0)  # (num_mashup + num_api, num_mashup + num_api)
        self.A = self.A.cuda()

        # self.D = torch.diag(torch.sum(self.A, dim=1))
        # D_pow = torch.pow(self.D, -0.5)
        D = torch.sum(self.A, dim=1)
        for i in range(len(D)):
            if (D[i] - 0.) < 1e-6:
                D[i] = 1.
        D = torch.diag(torch.pow(D, -0.5))
        self.A_head = torch.matmul(torch.matmul(D, self.A), D)
        self.A_head = self.A_head.cuda()

        self.sde_fc = nn.Sequential(
            nn.Linear(in_features=self.dim_word_embedding * self.num_mashup_token, out_features=self.dim_embedding),
            nn.BatchNorm1d(self.dim_embedding),
            nn.LeakyReLU(),
        )
        self.sde_fc_value = nn.Sequential(
            nn.Linear(in_features=self.dim_word_embedding * self.num_domain_token, out_features=self.dim_embedding),
            nn.BatchNorm1d(self.dim_embedding),
            nn.LeakyReLU(),
        )
        self.sde_fc_key = nn.Sequential(
            nn.Linear(in_features=self.dim_word_embedding * self.num_domain_token, out_features=self.dim_embedding),
            nn.BatchNorm1d(self.dim_embedding),
            nn.LeakyReLU(),
        )
        self.sie_fc = nn.Sequential(
            nn.Linear(in_features=self.dim_word_embedding * self.num_api_token, out_features=self.dim_embedding),
            nn.BatchNorm1d(self.dim_embedding),
            nn.LeakyReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # (batch_size, dim_word_embedding, len_text)
        batch_size = x.size(0)

        # Service Domain Enhancement
        v_mi = x.view(batch_size, -1)
        v_mi = self.sde_fc(v_mi)
        v_mi = torch.unsqueeze(v_mi, 1)  # (size_batch, 1, dim_embedding)
        v_mi = torch.unsqueeze(v_mi, 1)  # (size_batch, 1, 1, dim_embedding)
        v_value = self.sde_fc_value(self.domain_embedding)  # (num_domain, dim_embedding)
        v_key = self.sde_fc_key(self.domain_embedding)  # (num_domain, dim_embedding)
        v_value = torch.unsqueeze(v_value, 2)
        v_key = torch.unsqueeze(v_key, 2)  # (num_domain, dim_embedding, 1)

        al_matrix = torch.matmul(v_mi, v_key)
        al_matrix = torch.squeeze(al_matrix)  # (size_batch, num_domain)
        alpha_sum = torch.sum(al_matrix, dim=1)  # (size_batch, )
        alpha = torch.div(al_matrix, torch.unsqueeze(alpha_sum, 1))  # (size_batch, num_domain)
        alpha = torch.unsqueeze(torch.unsqueeze(alpha, 2), 2)
        s_m = torch.mul(alpha, v_value)
        s_m = torch.squeeze(s_m)  # (size_batch, num_domain, dim_embedding)
        s_m = torch.sum(s_m, dim=1)  # (size_batch, dim_embedding)
        v_mi = torch.squeeze(v_mi)
        z_m = (1 - self.beta) * s_m + self.beta * v_mi  # (size_batch, dim_embedding)

        # Structured Information Extraction
        v_m = self.sde_fc(self.mashup_embedding)
        v_s = self.sie_fc(self.api_embedding)

        X_k = torch.cat((v_m, v_s), dim=0)  # (num_mashup + num_api, dim_embedding)
        X = self.weight_gcn_layer[0] * X_k
        for i in range(1, self.num_gcn_layer):
            X_k = torch.matmul(self.A_head, X_k)
            X = X + self.weight_gcn_layer[i] * X_k
        O = X[-self.num_api:]  # (num_api, dim_embedding)
        z_m = torch.unsqueeze(z_m, dim=1)
        z_m = torch.unsqueeze(z_m, dim=1)
        O = torch.unsqueeze(O, dim=2)
        pred = torch.matmul(z_m, O)
        pred = torch.squeeze(pred)
        pred = self.sigmoid(pred)
        return pred


if __name__ == '__main__':
    with open('../../../data/api_mashup/mashup_related_api.json', 'r', encoding='utf-8') as f:
        apis = json.load(f)
    print(len(apis))
