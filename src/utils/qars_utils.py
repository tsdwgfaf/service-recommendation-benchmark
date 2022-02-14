import json
import random

import gensim
import torch
from gensim import similarities

from models.qars_model import QARSModel


def predict(
    data_dir: str,
    num_mashup: int,
    mashup_des_path: str,
    api_des_path: str,
    hdp_model_path: str,
    checkpoint_path: str,
    interaction_matrix_path: str,
    num_sims_api: int,
):
    inter = torch.load(interaction_matrix_path)
    with open(mashup_des_path, 'r', encoding='utf-8') as f:
        mashup_description = json.load(f)
    with open(api_des_path, 'r', encoding='utf-8') as f:
        api_description = json.load(f)

    num_mashup = len(mashup_description)
    num_api = len(api_description)

    hdp_model: gensim.models.HdpModel = gensim.models.hdpmodel.HdpModel.load(hdp_model_path)
    num_topics = hdp_model.get_topics().shape[0]

    dic = gensim.corpora.Dictionary(mashup_description + api_description)
    mashup_corpus = [dic.doc2bow(doc) for doc in mashup_description]
    api_corpus = [dic.doc2bow(doc) for doc in api_description]

    sims = similarities.MatrixSimilarity(hdp_model[api_corpus], num_features=num_topics)

    mashup_index = [i for i in range(num_mashup)]
    api_index = [i for i in range(num_api)]
    cand_mashups = random.sample(mashup_index, 64)

    model = QARSModel(
        data_dir=data_dir,
        data_name='api_mashup',
        model_dir='pre_model',
        model_name='hdp',
        dim_feature=100,
        num_sims_api=50,
        hp_lam_alpha=0.75,
        hp_lam_beta=0.75,
        lr=0.001
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    pred = []
    for ma in cand_mashups:
        corpus = mashup_corpus[ma]
        api_sims = sorted(enumerate(sims[hdp_model[corpus]]), key=lambda item: -item[1])
        cand_apis = [api_sims[i][0] for i in range(num_sims_api)]
        v_m = self.mashup_rectangular[ma].unsqueeze(0)
        v_i = self.api_rectangular.unsqueeze(2)
        p = torch.matmul(v_m, v_i).squeeze()  # (num_api, )
        useless_apis = torch.tensor(list(set(api_index).difference(set(cand_apis))), dtype=torch.int64,
                                    device='cuda:0')
        pred.append(p.scatter(0, useless_apis, 0.).unsqueeze(0))
    pred = torch.cat(pred, dim=0)
    target = torch.cat([inter[i].unsqueeze(0) for i in cand_mashups], dim=0)
    return pred, target.int()

if __name__ == '__main__':

