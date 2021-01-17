import os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import hydra

@hydra.main(config_path='config', config_name='config')
def app(cfg):
    # init model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    logger = hydra.utils.instantiate(cfg.logger) 
    trainer = pl.Trainer(logger=logger, **cfg.trainer)

    trainer.fit(model, datamodule)

    trainer.test(datamodule=datamodule)

if __name__ == "__main__":
    app()