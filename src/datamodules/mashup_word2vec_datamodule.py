from typing import Optional, List
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from datamodules.datasets.mashup_word2vec_dataset import MashupWord2VecDataset


class MashupWord2VecDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        train_val_test_split: List[int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        transpose: bool = False,
    ):
        super(MashupWord2VecDataModule, self).__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transpose = transpose

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = MashupWord2VecDataset(
            data_dir=self.data_dir,
            data_name='api_mashup',
            transpose=self.transpose,
        )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
