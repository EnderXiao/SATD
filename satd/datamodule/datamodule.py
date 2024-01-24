import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pytorch_lightning as pl
import torch
from satd.datamodule.dataset import HYBTr_Dataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from satd.utils.utils import (load_config)
from satd.datamodule.gen_symbols_struct_dict import vocab
from satd.datamodule.dataset import get_dataset, get_test_dataset

class CROHMEDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        config_path: str,
        zipfile_path:
        str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        test_year: str = "2014",
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        test_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
        image_resize: bool = True,
        image_width: int = 1600,
        image_height: int = 320,
        image_channel: int = 1,
        dropout: bool = True,
        train_image_path: str = 'data/train_image.pkl',
        train_label_path: str = 'data/train_label.pkl',
        eval_image_path: str = 'data/test_image.pkl',
        eval_label_path: str = 'data/test_label.pkl',
        test_image_path: str = 'data/test_image.pkl',
        test_label_path: str = 'data/test_label.pkl',
        word_path: str = 'data/word.txt',
        workers: int = 0,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.config_path = config_path
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug
        print(f"Load data from: {self.zipfile_path}")
        print("with batch size = ", self.train_batch_size)

    def setup(self, stage: Optional[str] = None) -> None:
        params = load_config(self.config_path)
        if stage == "fit" or stage is None:
            self.train_dataset, self.eval_dataset, self.train_dataLoader, self.eval_dataLoader = get_dataset(
                params)
        if stage == "test" or stage is None:
            self.test_dataset, self.test_dataLoader = get_test_dataset(
                params)

    def train_dataloader(self):
        return self.train_dataLoader

    def val_dataloader(self):
        return self.eval_dataLoader

    def test_dataloader(self):
        return self.test_dataLoader
