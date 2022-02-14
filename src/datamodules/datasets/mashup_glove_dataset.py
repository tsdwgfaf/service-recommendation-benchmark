import json
import os.path
import torch
from torch.utils.data import Dataset


class MashupGloveDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        mashup_title_name: str,
        mashup_feature_name: str,
        related_api_name: str,
        api_title_name: str,
    ):
        dataset_dir = os.path.join(data_dir, data_name)
        feature_path = os.path.join(dataset_dir, mashup_feature_name)
        related_apis_path = os.path.join(dataset_dir, related_api_name)
        mashup_title_path = os.path.join(dataset_dir, mashup_title_name)
        api_title_path = os.path.join(dataset_dir, api_title_name)

        with open(mashup_title_path, 'r', encoding='utf-8') as f:
            self.mashup_titles = json.load(f)
        with open(related_apis_path, 'r', encoding='utf-8') as f:
            self.mashup_related_apis = json.load(f)
        with open(api_title_path, 'r', encoding='utf-8') as f:
            self.api_titles = json.load(f)
        self.mashup_feature = torch.load(feature_path)

    def __len__(self):
        return len(self.mashup_titles)

    def __getitem__(self, idx):
        x = self.mashup_feature[idx]
        mashup_related_apis = self.mashup_related_apis[idx]
        index = torch.tensor([[self.api_titles.index(api_title) for api_title in mashup_related_apis]],
                             dtype=torch.int64)
        label = torch.zeros(size=(1, len(self.api_titles)), dtype=torch.int64).scatter(1, index, 1)
        label = torch.squeeze(label, dim=0)
        return x, label

if __name__ == '__main__':
    dataset = MashupGloveDataset(
        data_dir='../../../data',
        data_name='api_mashup',
        mashup_title_name='mashup_title.json',
        mashup_feature_name='mashup_feature.pt',
        related_api_name='mashup_related_api.json',
        api_title_name='api_title.json'
    )