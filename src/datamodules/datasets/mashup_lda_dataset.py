import json
import os

import gensim.models.ldamodel
import torch
from torch.utils.data import Dataset


class MashupLdaDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        model_dir: str,
        model_name: str,
    ):
        num_topic = 200

        data_path = os.path.join(data_dir, data_name)
        model_path = os.path.join(os.path.join(data_dir, model_dir), model_name)
        lda_path = os.path.join(model_path, 'model.wv')
        mashup_path = os.path.join(data_path, 'mashup_description.json')
        mashup_title_path = os.path.join(data_path, 'mashup_title.json')
        mashup_related_api_path = os.path.join(data_path, 'mashup_related_api.json')
        api_title_path = os.path.join(data_path, 'api_title.json')

        with open(mashup_title_path, 'r', encoding='utf-8') as f:
            self.mashup_titles = json.load(f)
        with open(mashup_related_api_path, 'r', encoding='utf-8') as f:
            self.mashup_related_apis = json.load(f)
        with open(api_title_path, 'r', encoding='utf-8') as f:
            self.api_titles = json.load(f)

        lda_model: gensim.models.ldamodel.LdaModel = gensim.models.ldamodel.LdaModel.load(lda_path)
        with open(mashup_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        dic = gensim.corpora.Dictionary(corpus)
        doc_bow = [dic.doc2bow(doc) for doc in corpus]
        topics = list(lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        self.num_mashup = len(topics)
        self.mashup_embedding = torch.zeros((self.num_mashup, num_topic), dtype=torch.float32)
        for i in range(self.num_mashup):
            topic = topics[i]
            for j in range(len(topic)):
                self.mashup_embedding[i][j] = float(topic[j][1])

    def __len__(self):
        return self.num_mashup

    def __getitem__(self, idx):
        mashup_related_apis = self.mashup_related_apis[idx]
        index = torch.tensor([[self.api_titles.index(api_title) for api_title in mashup_related_apis]], dtype=torch.int64)
        label = torch.zeros(size=(1, len(self.api_titles)), dtype=torch.int64).scatter(1, index, 1)
        label = torch.squeeze(label, dim=0)
        x = self.mashup_embedding[idx]
        return x, label
