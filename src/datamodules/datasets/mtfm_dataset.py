import json
import os.path
import torch
from torch.utils.data import Dataset


class MTFMMashupDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        transpose: bool = False,
    ):
        self.transpose = transpose
        dataset_dir = os.path.join(data_dir, data_name)
        mashup_title_path = os.path.join(dataset_dir, 'mashup_title.json')
        mashup_description_path = os.path.join(dataset_dir, 'mashup_description_word2vec.json')
        mashup_related_api_path = os.path.join(dataset_dir, 'mashup_related_api.json')
        api_title_path = os.path.join(dataset_dir, 'api_title.json')
        with open(mashup_title_path, 'r', encoding='utf-8') as f:
            self.mashup_title_list = json.load(f)
        with open(mashup_description_path, 'r', encoding='utf-8') as f:
            self.mashup_description_list = json.load(f)
        with open(mashup_related_api_path, 'r', encoding='utf-8') as f:
            self.mashup_related_api_list = json.load(f)
        with open(api_title_path, 'r', encoding='utf-8') as f:
            self.api_title_list = json.load(f)

    def __len__(self):
        return len(self.mashup_title_list)

    def __getitem__(self, idx):
        mashup_description = self.mashup_description_list[idx]
        mashup_related_apis = self.mashup_related_api_list[idx]
        index = torch.LongTensor([[self.api_title_list.index(api_title) for api_title in mashup_related_apis]])
        label = torch.zeros(size=(1, len(self.api_title_list)), dtype=torch.int64).scatter(1, index, 1)
        label = torch.squeeze(label, dim=0)
        # 返回词向量构成的description, shape=(词数, 300) 和 多分类的label, shape=(1, api总数), 包含某api则对应位置为1, 其余为0
        x = torch.tensor(mashup_description, dtype=torch.float32)
        if self.transpose:
            return torch.transpose(x, 0, 1), label
        return x, label


if __name__ == '__main__':
    out = torch.tensor([[0.7, 0.1, 0.4], [1, 0.03, 0.4]])
    pp = torch.clone(out)
    pp = pp.map_(pp, lambda x, *y: 1 if x > 0.5 else 0)
    pp = pp.int()
    print(out)
    print(pp)
