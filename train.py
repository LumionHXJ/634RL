from datetime import datetime
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from alphazero_pl import AlphaZeroNet


def main(gpus=1):
    pl.seed_everything(42)
    model = AlphaZeroNet.load_from_checkpoint('checkpoints/epoch=549.ckpt',
                                              board_height=3, 
                                              board_width=3, 
                                              k=3, 
                                              search_limits=1000)
    ckpt_hook = ModelCheckpoint(dirpath='./checkpoints/',
                                filename='{epoch}',
                                every_n_epochs=50)
    logger = TensorBoardLogger('logs', 
                               name=datetime.now().strftime("%Y%m%d-%H%M%S"))
    trainer = pl.Trainer(max_epochs=1500,
                         callbacks=[ckpt_hook],
                         gpus=gpus,
                         logger=logger,
                         log_every_n_steps=1)
    trainer.fit(model)

main()