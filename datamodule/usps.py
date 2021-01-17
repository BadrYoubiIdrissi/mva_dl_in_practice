import pytorch_lightning as pl
from torchvision.datasets import USPS
from torchvision import transforms

from torch.utils.data import DataLoader, random_split

class USPSDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = PATH, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.usps_test = USPS(self.data_dir, train=False, transform=transforms.ToTensor())
        usps_full = USPS(self.data_dir, train=True, transform=transforms.ToTensor())
        self.usps_train, self.usps_val = random_split(usps_full, [6000, 1291])

    def train_dataloader(self):
        return DataLoader(self.usps_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.usps_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.usps_test, batch_size=self.batch_size)