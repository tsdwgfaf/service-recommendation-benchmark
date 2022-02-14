from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from datamodules.datasets.mashup_api_matrix_dataset import MashupApiMatrixDataset


class MashupApiMatrixDatamodule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super(MashupApiMatrixDatamodule, self).__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = MashupApiMatrixDataset(
            data_dir=self.data_dir,
            data_name=self.data_name,
        )
        self.data_train = dataset
        self.data_val = dataset
        self.data_test = dataset

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
