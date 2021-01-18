import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from hydra.utils import instantiate

class CustomCNNModel(pl.LightningModule):

    def __init__(self, input_width, input_channels, output_size, cnn_layers, kernel_sizes, activ_fn, criterion, lr, optimizer):
        super().__init__()
        
        self.build_model(input_width, input_channels, output_size, cnn_layers, kernel_sizes, activ_fn, criterion=="mse")
        self.build_criterion(criterion)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.lr = lr
        self.optimizer = optimizer

    def build_criterion(self, criterion):
        losses = {
                    "mse" : nn.MSELoss(),
                    "ce" : nn.CrossEntropyLoss()
                }
        self.criterion = losses[criterion]

    def build_model(self, input_width, input_channels, output_size, cnn_layers, kernel_sizes, activ_fn, add_sigmoid):
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        self.activ_fn = activations[activ_fn]
        cnn_layers = [input_channels]+list(cnn_layers)
        self.convs = nn.ModuleList([nn.Conv2d(cnn_layers[i], cnn_layers[i+1], kernel_sizes[i]) for i in range(len(kernel_sizes))])
        self.pool = nn.MaxPool2d(2, 2)
        self.output_size = output_size

        self.featuremap_size = input_width
        for size in kernel_sizes:
            self.featuremap_size = (self.featuremap_size - (size - 1))//2
        self.featuremap_output_size=self.featuremap_size*self.featuremap_size*cnn_layers[-1]
        self.fc = nn.Linear(self.featuremap_output_size, self.output_size)

    def forward(self, x):
        for conv_layer in self.convs:
            x = self.pool(self.activ_fn(conv_layer(x)))
        x = x.view(-1, self.featuremap_output_size)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD
        }
        optimizer = optimizers[self.optimizer]
        return optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)