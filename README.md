# Gomoku by 634s

## Installation

```bash
conda create -n botzone python=3.6
conda activate botzone
pip install -r requirements.txt
```

## Training

```bash
python gameplay.py [--black BLACK] [--white WHITE] height width k
```

- BLACK and WHITE 
  - omit represents human player
  - checkpoint of model
  - (h, w, k) for gomoku game setttings

## GamePlay

```bash
python train.py [-h] [--gpus GPUS] [--ckpt CKPT]
                height width k search_limits epochs
```

- positional arguments:
	- height                Height of board.
	- width                 Width of board.
	- k                     Game rule: k stones in a row.
	- search_limits         Number of searches for MCTS.
	- epochs                Number of traning epochs.

- optional arguments:
	- --gpus GPUS, -g GPUS  Number of gpus used.
	- --ckpt CKPT, -c CKPT  Resume from checkpoints.
- Bugs: can't save checkpoints while training.