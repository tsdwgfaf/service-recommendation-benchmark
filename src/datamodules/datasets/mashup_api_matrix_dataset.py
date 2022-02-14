import torch
from torch.utils.data import Dataset
import os
import numpy as np


class MashupApiMatrixDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
    ):
        dataset_dir = os.path.join(data_dir, data_name)
        self.mashup_api_matrix = torch.load(os.path.join(dataset_dir, 'user_item_interaction_matrix.pt'))

    def __len__(self):
        return self.mashup_api_matrix.size(0)

    def __getitem__(self, idx):
        return self.mashup_api_matrix
