import copy
import time
from math import sqrt

import torch
from torch.distributions import Categorical, Dirichlet

class Gomoku:
    BLACK_STONE = 0
    WHITE_STONE = 1
    WIN = -1
    TIE = 0
    NOT_DETERMINED = 1
    DIRECTION = [(1, 0), (0, 1), (1, 1), (1, -1)]
    BLACK_SYMBOL = 'X'
    WHITE_SYMBOL = 'O'
    EMPTY_SYMBOL = '_'

    def __init__(self, height=15, width=15, k=5) -> None:
        """Generic Gomoku form playing on board (h, w).
        Winning when k stones in a <row>."""
        self.height = height
        self.width = width
        self.k = k
    
    def init_game(self):
        """Initialize game board, current player."""
        self.board = torch.zeros((2, self.height, self.width), 
                                 dtype=bool)
        # black is init player 
        self.to_place = Gomoku.BLACK_STONE
        self.last_move = None

    def loc2tuple(self, loc):
        """loc(int) to i-j style grid"""
        return loc // self.width, loc % self.width
    
    def step(self, loc, check_terminate=True):
        """Player wanna place a stone on board[player, x, y].
        
        Where player is save in `self.to_place`.
        Tuple (x,y) is computed through `self.loc2tuple`.

        Args:
            player(int): 0 for init player(black), 1 for second one.
            loc(tuple): ->(x,y), ranging from [0, h-1] / [0, w-1], 
            plz check the range before feeding in if needed.
            check_terminate(bool): checking if game is terminated.
        
        Return:
            Return whether terminate if `check_terminate=True`, 
            else return False.
        """
        x, y = self.loc2tuple(loc)
        assert not torch.any(self.board[:, x, y]) # promising (x,y) is empty
        self.board[self.to_place, x, y] = True # place a stone
        self.last_move = (x, y) # memorize last move
        
        ret = Gomoku.NOT_DETERMINED

        if check_terminate:
            ret = self.check(x, y)
        
        # change current player
        self.to_place = Gomoku.BLACK_STONE \
            if self.to_place == Gomoku.WHITE_STONE else Gomoku.WHITE_STONE 
        return ret

    def check(self, x, y):
        """Checking whether current player win the game by placing on (x,y).
        Must be called before changing player!"""
        for dx, dy in Gomoku.DIRECTION:
            count = 1
            for i in range(1, self.k):
                if 0 <= x + i*dx < self.height \
                    and 0 <= y + i*dy < self.width \
                        and self.board[self.to_place, x + i*dx, y + i*dy]:
                    count += 1
                else:
                    break # not continued in current direction
            
            #  trying opposite direction
            for i in range(1, self.k):
                if 0 <= x - i*dx < self.height \
                    and 0 <= y - i*dy < self.width \
                        and self.board[self.to_place, x - i*dx, y - i*dy]:
                    count += 1
                else:
                    break
            
            # if there are more than k connected stones
            if count >= self.k:
                return Gomoku.WIN
        
        # all grid has a stone
        if self.board.any(dim=0).all():            
            return Gomoku.TIE
        
        return Gomoku.NOT_DETERMINED
    
    def to_model_input(self):
        """Return Tensor in (4, h, w).
        4 represents player, opponent, last move, colour."""
        input = torch.zeros((4, self.height, self.width), 
                            dtype=torch.float32)
        
        if self.to_place == 0: # black side
            input[:2] = self.board.float()
        
        else: # white side
            input[:2] = self.board[[1, 0]].float()
            input[3] = Gomoku.WHITE_STONE
        
        if self.last_move is not None: # not first move
            input[2, self.last_move[0], self.last_move[1]] = 1
        
        return input
        
    def generate_mask(self):
        """Generating mask that block player to place on the grid 
        that already have a stone.

        Returns:
            mask(tensor): shape like (h*w, )
        """
        return self.board.any(dim=0).flatten()
    
    def __str__(self):
        ret = ''
        for i in range(self.height):
            for j in range(self.width):
                if self.board[Gomoku.BLACK_STONE, i, j]:
                    ret += Gomoku.BLACK_SYMBOL
                elif self.board[Gomoku.WHITE_STONE, i, j]:
                    ret += Gomoku.WHITE_SYMBOL
                else:
                    ret += Gomoku.EMPTY_SYMBOL
            ret += '\n'
        return ret

class Node:
    """Numerical value adapted from alphazero_gomoku: https://github.com/junxiaosong/AlphaZero_Gomoku
    """
    c = 5 # balance ratio for UCB
    def __init__(self, parent=None, loc=None):
        """
        Args:
            parent(Node or None)
            loc(int): parent node placing on `loc` resulted in child.
        """
        self.parent = parent
        self.loc = loc
        self.childs = {} # dict like x * height + y: childNode
        self.nsa = None # h * w, represents for nsa
        self.qsa = None # h * w, represents for qsa
        self.psa = None # psa, output of DNN
        self.n = 0 # total visits: n = sum(nsa) + 1(1 for first visit: the expansion)
    
    def select(self):
        """Selecting childs node with max ucb.
        Constructing childs if child hasn't been expanded.
        """
        ucb = self.qsa + Node.c * self.psa / (1 + self.nsa) * sqrt(self.n)
        loc = ucb.argmax().item() 
        if self.childs.get(loc, None) is None:
            self.childs[loc] = Node(self, loc)
        return self.childs[loc], loc
    
class MCTS:
    """Implementation of MCTS for Alphazero"""
    def __init__(self, 
                 model, 
                 selfplay=True,
                 time_budget=5.0,
                 search_limits=1e3,
                 temperature=1) -> None:
        """
        Args:
            model: DNN for state evaluation.
            selfplay: for selfplay training, generating data by encouraging exploration.
            time_budget: search time limits
            search_limits: using in selfplay, having deterministic outcome by setting seed.
            temperature: computing action probs.
        """
        self.model = model
        self.root = Node(None, None)
        self.selfplay = selfplay
        self.time_budget = time_budget
        self.search_limits = search_limits
        self.temperature = temperature # TODO: annealing temperature?
    
    def search(self, gomoku):
        """Search through MCTS, start with state `gomoku`."""
        time_tik = time.time()
        self.root.parent = None
        if self.selfplay:
            for i in range(self.search_limits):
                node, value = self.tree_policy(self.root, copy.deepcopy(gomoku))
                self.backup(node, value) 
        else:
            while time.time() - time_tik < self.time_budget:
                node, value = self.tree_policy(self.root, copy.deepcopy(gomoku))
                self.backup(node, value)  
        return     
    
    def tree_policy(self, node, gomuku: Gomoku):
        """Recursively select node in MCTS.
        Returns the node and value of the selected node.
        """
        # case 1: expand current node by DNN if first visit
        if node.n == 0:
            input = gomuku.to_model_input().unsqueeze(0)
            p, v = self.model(input.to(self.model.device)) # mask & softmax inside model
            
            p, v = p.view(-1).cpu(), v.view(-1).cpu()[0] # p: (h*w, ), v:(1, )

            # adding Dirichlet noise for root prior
            node.psa = p
            if self.selfplay and node.parent is None:
                mask = gomuku.generate_mask()
                node.psa[~mask] = 0.75 * p[~mask] + \
                    0.25 * Dirichlet(0.03 * torch.ones_like(p[~mask])).sample()
                
            # init value of qsa must equal to minimal, it'll be washed after updating!
            node.qsa = torch.ones_like(p) * -1 
            node.nsa = torch.zeros_like(p)
            node.n = 1 # first visit
            return node, v
        
        # TODO: pruning deterministic cases

        # case 2: node already expanded
        child, loc = node.select()
        res = gomuku.step(loc, check_terminate=True) # TODO: always check if game is ended?
        if res <= 0: # game has terminated: TIE or child LOSE
            return child, res 
        return self.tree_policy(child, gomuku)

    def backup(self, node, value):
        """Backup value from node to root."""
        while node.parent is not None:
            loc = node.loc
            node = node.parent
            value = -value # turning into opponent value, ranging from [-1,1]

            # updating parent node
            node.nsa[loc] += 1
            node.qsa[loc] += (value - node.qsa[loc]) / node.nsa[loc]
            node.n += 1
    
    def action_probs(self):
        """Action probs propto n(s,a)^{1/tau}
        """
        probs = self.root.nsa ** (1 / self.temperature)
        return probs / probs.sum()      
    
    def sample_action(self, probs):
        """Sample action by `probs`, probs might be return from `self.action_probs`.
        Returns in loc-style.
        """
        if self.selfplay:
            return Categorical(probs=probs).sample().item()
        else:
            return probs.argmax().item()
