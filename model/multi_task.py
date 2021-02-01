import torch
from torch import nn

import pytorch_lightning as pl

class MultiheadModel(pl.LightningModule):

    def __init__(self, input_width, input_channels, output_size, cnn_layers, layers_color, layers_digit, kernel_sizes, activ_fn, criterion, lr, optimizer):
        super().__init__()
        
        self.build_model(input_width, input_channels, output_size, cnn_layers, layers_color, layers_digit, kernel_sizes, activ_fn, criterion=="mse")
        self.build_criterion(criterion)

        self.train_acc_color = pl.metrics.Accuracy()
        self.val_acc_color = pl.metrics.Accuracy()
        self.test_acc_color = pl.metrics.Accuracy()
        
        self.train_acc_digit = pl.metrics.Accuracy()
        self.val_acc_digit = pl.metrics.Accuracy()
        self.test_acc_digit = pl.metrics.Accuracy()

        self.lr = lr
        self.optimizer = optimizer

    def build_criterion(self, criterion):
        losses = {
                    "mse" : nn.MSELoss(),
                    "ce" : nn.CrossEntropyLoss()
                }
        self.criterion = losses[criterion]

    def build_model(self, input_width, input_channels, output_size, cnn_layers, layers_color, layers_digit, kernel_sizes, activ_fn, add_sigmoid):
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

        n_nodes_color = [self.featuremap_output_size] + list(layers_color) + [5]
        n_nodes_digit = [self.featuremap_output_size] + list(layers_digit) + [10]
        self.fc_color = nn.ModuleList([nn.Linear(n_nodes_color[i], n_nodes_color[i+1]) for i in range(len(n_nodes_color) - 1)])
        self.fc_digit = nn.ModuleList([nn.Linear(n_nodes_digit[i], n_nodes_digit[i+1]) for i in range(len(n_nodes_digit) - 1)])

    def forward(self, x):
        for conv_layer in self.convs:
            x = self.pool(self.activ_fn(conv_layer(x)))
        x = x.view(-1, self.featuremap_output_size)
        color, digit = x, x
        for color_layer in self.fc_color[:-1]:
            color = self.activ_fn(color_layer(color))
        color = self.fc_color[-1](color)
        for digit_layer in self.fc_digit[:-1]:
            digit = self.activ_fn(digit_layer(digit))
        digit = self.fc_digit[-1](digit)
        return color, digit

    def configure_optimizers(self):
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD
        }
        optimizer = optimizers[self.optimizer]
        return optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        color, digit = self(x)
        loss_color = self.criterion(color, y[:,1])
        loss_digit = self.criterion(digit, y[:,0])
        loss = loss_color + loss_digit
        self.train_acc_color(color, y[:,1])
        self.train_acc_digit(digit, y[:,0])
        self.log("train_loss_color", loss_color, on_epoch=True, on_step=True)
        self.log("train_loss_digit", loss_digit, on_epoch=True, on_step=True)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("train_acc_color", self.train_acc_color, on_epoch=True, on_step=True)
        self.log("train_acc_digit", self.train_acc_digit, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        color, digit = self(x)
        loss_color = self.criterion(color, y[:,1])
        loss_digit = self.criterion(digit, y[:,0])
        loss = loss_color + loss_digit
        self.val_acc_color(color, y[:,1])
        self.val_acc_digit(digit, y[:,0])
        self.log("val_loss_color", loss_color, on_epoch=True, on_step=True)
        self.log("val_loss_digit", loss_digit, on_epoch=True, on_step=True)
        self.log("val_loss", loss, on_epoch=True, on_step=True)
        self.log("val_acc_color", self.val_acc_color, on_epoch=True, on_step=True)
        self.log("val_acc_digit", self.val_acc_digit, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        color, digit = self(x)
        loss_color = self.criterion(color, y[:,1])
        loss_digit = self.criterion(digit, y[:,0])
        loss = loss_color + loss_digit
        self.test_acc_color(color, y[:,1])
        self.test_acc_digit(digit, y[:,0])
        self.log("test_loss_color", loss_color, on_epoch=True, on_step=True)
        self.log("test_loss_digit", loss_digit, on_epoch=True, on_step=True)
        self.log("test_loss", loss, on_epoch=True, on_step=True)
        self.log("test_acc_color", self.test_acc_color, on_epoch=True, on_step=True)
        self.log("test_acc_digit", self.test_acc_digit, on_epoch=True, on_step=True)