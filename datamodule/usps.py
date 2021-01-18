import os, requests

import pytorch_lightning as pl
from torchvision.datasets import USPS
from torchvision import transforms

from torch.utils.data import DataLoader, random_split

class USPSDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "USPS", batch_size=100, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir=data_dir
        self.num_workers=num_workers

    def setup(self, stage=None):
        self.usps_test = USPS(self.data_dir, train=False, transform=transforms.ToTensor(), download=True)
        usps_full = USPS(self.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        self.usps_train, self.usps_val = random_split(usps_full, [6000, 1291])

    def train_dataloader(self):
        return DataLoader(self.usps_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.usps_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.usps_test, batch_size=self.batch_size, num_workers=self.num_workers)