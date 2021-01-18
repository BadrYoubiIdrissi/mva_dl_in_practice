import os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

@hydra.main(config_path='config', config_name='config')
def app(cfg):
    # init model
    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    logger = hydra.utils.instantiate(cfg.logger) 
    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.checkpoint)
    earl_callback = EarlyStopping(**cfg.callbacks.early_stopping)
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], **cfg.trainer)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # lr_finder = trainer.tuner.lr_find(model, datamodule)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # plt.savefig("lr_tuning.png")

    trainer.fit(model, datamodule)
    
    trainer.test(datamodule=datamodule)

if __name__ == "__main__":
    app()