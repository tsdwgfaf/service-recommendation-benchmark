from typing import Optional, List
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from datamodules.datasets.mashup_glove_dataset import MashupGloveDataset


class MashupGloveDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        mashup_title_name: str,
        mashup_feature_name: str,
        related_api_name: str,
        api_title_name: str,
        train_val_test_split: List[int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super(MashupGloveDataModule, self).__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.mashup_title_name = mashup_title_name
        self.mashup_feature_name = mashup_feature_name
        self.related_api_name = related_api_name
        self.api_title_name = api_title_name
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = MashupGloveDataset(
            data_dir=self.data_dir,
            data_name=self.data_name,
            mashup_title_name=self.mashup_title_name,
            mashup_feature_name=self.mashup_feature_name,
            related_api_name=self.related_api_name,
            api_title_name=self.api_title_name,
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
