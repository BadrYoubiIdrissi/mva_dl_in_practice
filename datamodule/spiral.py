import pytorch_lightning as pl
import torch
from torchvision.datasets import USPS
from torchvision import transforms

from torch.utils.data import TensorDataset, DataLoader, random_split

class SpiralDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, train_size, val_size, test_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def spiral(self, phi):
        x = (phi+1)*torch.cos(phi)
        y = phi*torch.sin(phi)
        return torch.cat((x, y), dim=1)

    def generate_data(self, num_data):
        angles = torch.empty((num_data, 1)).uniform_(1, 15)
        data = self.spiral(angles)
        # add some noise to the data
        data += torch.empty((num_data, 2)).normal_(0.0, 0.4)
        labels = torch.zeros((num_data,), dtype=torch.int)
        # flip half of the points to create two classes
        data[num_data//2:,:] *= -1
        labels[num_data//2:] = 1
        return TensorDataset(data, labels)

    def setup(self, stage=None):
        self.train = self.generate_data(self.train_size)
        self.val = self.generate_data(self.val_size)
        self.test = self.generate_data(self.test_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

