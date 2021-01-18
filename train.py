import os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path='config', config_name='config')
def app(cfg):
    # init model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    logger = hydra.utils.instantiate(cfg.logger) 
    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.checkpoint)
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], **cfg.trainer)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer.fit(model, datamodule)

    trainer.test(datamodule=datamodule)

if __name__ == "__main__":
    app()