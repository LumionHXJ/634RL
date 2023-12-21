from datetime import datetime
import argparse

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from alphazero_pl import AlphaZeroNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('height', type=int,  
                        help='Height of board.')
    parser.add_argument('width', type=int, 
                        help='Width of board.')
    parser.add_argument('k', type=int, 
                        help='Game rule: k stones in a row.')
    parser.add_argument('search_limits', type=int, 
                        help='Number of searches for MCTS.')   
    parser.add_argument('epochs', type=int, 
                        help='Number of traning epochs.')
    
    parser.add_argument('--gpus', '-g', type=int, default=0, 
                        required=False, 
                        help='Number of gpus used.')
    parser.add_argument('--ckpt', '-c', type=str, default=None, 
                        required=False, 
                        help='Resume from checkpoints.')   
    
    args = parser.parse_args()
    return args

def main():
    # mustn't fix seed in ddp mode, will generate same data!
    # pl.seed_everything(42)
    args = parse_args()

    model = AlphaZeroNet(board_height=args.height, 
                         board_width=args.width, 
                         k=args.k, 
                         search_limits=args.search_limits)
    ckpt_hook = ModelCheckpoint(dirpath='./checkpoints/',
                                filename='{epoch}',
                                every_n_epochs=10,
                                save_last=True)
    logger = TensorBoardLogger('logs', 
                               name=datetime.now().strftime("%Y%m%d-%H%M%S"))
    ddp_plugin = DDPPlugin(find_unused_parameters=False)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         callbacks=[ckpt_hook],
                         plugins=[ddp_plugin],
                         gpus=args.gpus,
                         logger=logger,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=20,
                         reload_dataloaders_every_n_epochs=1)
    trainer.fit(model, ckpt_path=args.ckpt)

main()