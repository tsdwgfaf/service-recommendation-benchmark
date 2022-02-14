import json
import os.path
import torch
from torch.utils.data import Dataset


class MashupWord2VecDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        transpose: bool = False,
    ):
        self.transpose = transpose
        dataset_dir = os.path.join(data_dir, data_name)
        mashup_title_path = os.path.join(dataset_dir, 'mashup_title.json')
        mashup_description_path = os.path.join(dataset_dir, 'mashup_embedding_word2vec.pt')
        mashup_related_api_path = os.path.join(dataset_dir, 'mashup_related_api.json')
        api_title_path = os.path.join(dataset_dir, 'api_title.json')
        with open(mashup_title_path, 'r', encoding='utf-8') as f:
            self.mashup_titles = json.load(f)
        with open(mashup_related_api_path, 'r', encoding='utf-8') as f:
            self.mashup_related_apis = json.load(f)
        with open(api_title_path, 'r', encoding='utf-8') as f:
            self.api_titles = json.load(f)
        self.mashup_embedding = torch.load(mashup_description_path)

    def __len__(self):
        return len(self.mashup_titles)

    def __getitem__(self, idx):
        x = self.mashup_embedding[idx]
        mashup_related_apis = self.mashup_related_apis[idx]
        index = torch.tensor([[self.api_titles.index(api_title) for api_title in mashup_related_apis]],
                             dtype=torch.int64)
        label = torch.zeros(size=(1, len(self.api_titles)), dtype=torch.int64).scatter(1, index, 1)
        label = torch.squeeze(label, dim=0)
        # 返回词向量构成的description, shape=(词数, 300) 和 多分类的label, shape=(1, api总数), 包含某api则对应位置为1, 其余为0
        if self.transpose:
            return torch.transpose(x, 0, 1), label
        return x, label
