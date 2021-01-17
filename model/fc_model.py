import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from hydra.utils import instantiate

class CustomFCModel(pl.LightningModule):

    def __init__(self, input_size, output_size, neurons_per_layer, nb_layers, activ_fn, criterion, optimizer):
        super().__init__()
        
        self.build_model(input_size, output_size, neurons_per_layer, nb_layers, activ_fn, criterion=="mse")
        self.build_criterion(criterion)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.optimizer = optimizer

    def build_criterion(self, criterion):
        losses = {
                    "mse" : nn.MSELoss(),
                    "bce" : nn.BCEWithLogitsLoss()
                }
        self.criterion = losses[criterion]

    def build_model(self, input_size, output_size, neurons_per_layer, nb_layers, activ_fn, add_sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        n_nodes = self.build_layer_size_list(neurons_per_layer, nb_layers)
        self.linears = nn.ModuleList([nn.Linear(n_nodes[i], n_nodes[i+1]) for i in range(len(n_nodes) - 1)])
        self.activ_fns = self.get_activ_fun_from_layer_sizes(activ_fn, n_nodes, add_sigmoid)
    
    def get_activ_fun_from_layer_sizes(self, activ, sizes, add_sigmoid=True):
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        return [activations[activ] for i in range(len(sizes)-2)]+[nn.Sigmoid() if add_sigmoid else lambda x: x]
    
    def build_layer_size_list(self, nb_neurons, nb_layers):
        return [self.input_size] + [nb_neurons for i in range(nb_layers)] + [self.output_size]

    def configure_optimizers(self):
        return instantiate(self.optimizer, self.parameters())

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.linears):
            x = self.activ_fns[i](layer(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # import pdb; pdb.set_trace()
        y_hat = self(x).view(-1)
        loss = self.criterion(y_hat, y.float())
        self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.criterion(y_hat, y.float())
        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.criterion(y_hat, y.float())
        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)