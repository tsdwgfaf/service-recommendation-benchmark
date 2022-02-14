import random

import torch
from torch import nn


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bpr_cost = -torch.mean(torch.log(self.sigmoid(pred - target)))
        # l2_cost = sum(torch.norm(weight, 2) for weight in self.params)
        # print(l2_cost)
        return bpr_cost
        # return bpr_cost + self.p_lambda * l2_cost


class QARSLoss(nn.Module):

    def __init__(self):
        super(QARSLoss, self).__init__()
        # self.w = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = target - pred
        loss = torch.sum(loss.pow(2.))
        return loss


class T2L2Loss(nn.Module):

    def __init__(self, dim: int):
        super(T2L2Loss, self).__init__()
        self.index = set(range(dim))
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        batch_size = pred.size(0)
        loss = torch.zeros((1,), dtype=torch.float32, requires_grad=True).cuda()
        for i in range(batch_size):
            pr = pred[i]
            ta = target[i]
            positive_index = ta.nonzero().squeeze(1).tolist()
            num_positive = len(positive_index)
            num_negative = num_positive * 6
            negative_index = list(self.index.difference(set(positive_index)))
            negative_index = random.sample(negative_index, num_negative)
            pred_index = positive_index + negative_index
            new_pred = torch.tensor([pr[idx] for idx in pred_index], dtype=torch.float32).cuda()
            new_target = torch.tensor([1. for idx in range(num_positive)] + [0. for idx in range(num_negative)],
                                      dtype=torch.float32).cuda()
            loss += self.loss(new_pred, new_target)
        return loss
