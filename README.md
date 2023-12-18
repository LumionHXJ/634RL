
[toc]

# Gomoku by 634s

## Installation

```bash
conda create -n botzone python=3.6
conda activate botzone
pip install -r requirements.txt
```

## Training

training by modifying arguments in `train.py`

## GamePlay

```bash
python gameplay.py [--black BLACK] [--white WHITE] height width k
```

- BLACK and WHITE 
  - omit represents human player
  - checkpoint of model
  - (h, w, k) for gomoku game setttings

