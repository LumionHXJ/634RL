import argparse

import pytorch_lightning as pl
import torch

from alphazero_pl import AlphaZeroNet
from mcts import MCTS, Gomoku, Node

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('height', type=int,  
                        help='Height of board.')
    parser.add_argument('width', type=int, 
                        help='Width of board.')
    parser.add_argument('k', type=int, 
                        help='Game rule: k stones in a row.')

    parser.add_argument('--black', '-b', type=str, default=None, 
                        required=False, 
                        help='Black Stone Player, none for human player.')
    parser.add_argument('--white', '-w', type=str, default=None, 
                        required=False, 
                        help='White Stone Player, none for human player.')
    

    args = parser.parse_args()
    return args


def tuple2loc(x, y, width):
    return x * width + y


def gameplay(mcts, gomoku, similate=None):
    step = 0
    print(gomoku)
    while True:
        if isinstance(mcts[step % 2], MCTS):
            mcts[step % 2].search(gomoku)
            action_probs = mcts[step % 2].action_probs()
            print(action_probs)         
            loc = mcts[step % 2].sample_action(action_probs)
        else:
            while(True):
                try:
                    x, y = tuple(map(int, input().split(' ')))
                except:
                    print("Invalid input.")
                    continue
                if not 0 <= x < gomoku.height or not 0 <= y < gomoku.width:
                    print('Must place stone on board!')
                    continue
                if torch.any(gomoku.board[:, x, y]):
                    print('Must place on empty grid!')
                    continue
                break
            loc = tuple2loc(x, y, gomoku.width)

        end = gomoku.step(loc, check_terminate=True)

        print(f"Player {step % 2} place on {gomoku.loc2tuple(loc)}")
        print("_" * 30)        
        print(gomoku)

        if end == Gomoku.TIE:
            print("Game End! Result: TIE!")
            break 
        elif end == Gomoku.WIN:
            print(f"Game End! Result: Player {step % 2} WIN!")
            break
        
        for tree in mcts:
            if isinstance(tree, MCTS):
                tree.root = tree.root.childs.get(loc, Node(None, loc))  # step forward
        step += 1
    

def main():
    args = parse_args()

    mcts = []
    if args.black:
        model = AlphaZeroNet.load_from_checkpoint(args.black,
                                                  map_location='cuda',
                                                  board_height=args.height, 
                                                  board_width=args.width,
                                                  k=args.k)
        mcts.append(MCTS(model, selfplay=False, time_budget=3.0))
    else:
        mcts.append(None)
    if args.white:
        model = AlphaZeroNet.load_from_checkpoint(args.white, 
                                                  map_location='cuda',
                                                  board_height=args.height, 
                                                  board_width=args.width,
                                                  k=args.k)
        mcts.append(MCTS(model, selfplay=False, time_budget=3.0))
    else:
        mcts.append(None)
    
    gomoku = Gomoku(args.height, args.width, args.k)
    gomoku.init_game()
    
    gameplay(mcts, gomoku)
    

main()