import pytorch_lightning as pl
from torchvision.datasets import USPS
from torchvision import transforms

class ColorizedUSPSDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = PATH, batch_size, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def colorize_dataset(dataset):
        # array of colors
        COLORS = torch.tensor([
            [1.0, 0.0, 0.0], # 0 RED
            [0.0, 1.0, 0.0], # 1 GREEN
            [0.0, 0.0, 1.0], # 2 BLUE
            [1.0, 1.0, 0.0], # 3 YELLOW
            [1.0, 0.0, 1.0], # 4 MAGENTA
        ])
        N = len(dataset)
        images = torch.tensor(dataset.data/255).view(N, 1, 16, 16)
        labels = torch.tensor(dataset.targets).view(N, 1)
        color_labels = torch.randint(0, 5, (N,))
        colorized_images = images * COLORS[color_labels, :].view(N,3,1,1)
        full_labels = torch.cat((labels, color_labels.view(N, 1)), dim=1)
        return TensorDataset(colorized_images, full_labels)

    def setup(self, stage=None):
        self.usps_test = USPS(self.data_dir, train=False)
        usps_full = USPS(self.data_dir, train=True)
        self.usps_train, self.usps_val = random_split(usps_full, [6000, 1291])

    def train_dataloader(self):
        return DataLoader(self.usps_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.usps_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.usps_test, batch_size=self.batch_size, num_workers=self.num_workers)
