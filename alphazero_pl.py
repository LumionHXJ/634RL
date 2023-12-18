import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader, Dataset

from mcts import MCTS, Gomoku, Node
from data_utils import GomukuDataset

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=kernel_size,  
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=kernel_size, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class AlphaZeroNet(pl.LightningModule):
    def __init__(self, 
                 board_height, 
                 board_width,
                 k, 
                 input_channels=4, 
                 num_resblock=3,
                 lamb=1,
                 search_limits=1000):
        super().__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.k = k
        self.search_limits = search_limits

        self.backbone = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                                      *[ResBlock(32, 32, 3) for i in range(num_resblock)])

        self.policy_conv = nn.Sequential(nn.Conv2d(32, 4, kernel_size=1),
                                         nn.BatchNorm2d(4),
                                         nn.ReLU())
        self.policy_proj = nn.Sequential(nn.Linear(4*board_width*board_height, 
                                                   board_width*board_height)) # n, h*w
        
        self.softmax = nn.Softmax(dim=1)

        self.value_conv = nn.Sequential(nn.Conv2d(32, 2, kernel_size=1),
                                        nn.BatchNorm2d(2),
                                        nn.ReLU())
        self.value_proj = nn.Sequential(nn.Linear(2*board_width*board_height, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1),
                                      nn.Tanh())
        
        self.policy_loss = nn.KLDivLoss() # mean not batch mean
        self.value_loss = nn.MSELoss()
        
        self.lamb = lamb
        
    def generate_policy_mask(self, state):
        """Generate mask for policy, True for masking
        Return:
            mask(tensor): shape like n, h, w    
        """
        mask = state[:, :2].any(dim=1) # n, h*w
        return mask.view(-1, self.board_width*self.board_height)
    
    def forward(self, state, mask=None):
        """
        Returns:
            policy(tensor): probs after softmax + mask(not log-softmax so log is needed before
                NLL), shape like (n, h*w).
            value(tensor): evaluation of state, shape like (n, 1)
        """
        if mask is None:
            mask = self.generate_policy_mask(state)
        x = self.backbone(state)

        x_policy = self.policy_conv(x)
        policy = self.policy_proj(x_policy.view(-1, 4*self.board_width*self.board_height))
        policy = policy.masked_fill_(mask, float('-inf')) # n, h*w
        policy = self.softmax(policy) # output softmax as prob

        x_value = self.value_conv(x)
        value = self.value_proj(x_value.view(-1, 2*self.board_width*self.board_height))

        return policy, value
    
    def training_step(self, batch_input, batch_index):
        """
        Args:
            `batch_input` can be decomposed to `state`, `policy_target` and `value_target`.
            `state`: tensor with shape (n, 4, h, w)
            `policy_target`: tensor with shape (n, h*w)
            `value_target`: tensor with shape (n, 1)
        """
        state, policy_target, value_target = batch_input
        mask = self.generate_policy_mask(state)

        policy_pred, value_pred = self(state, mask)
        
        policy_pred = torch.log(policy_pred[~mask]) # will be flatten to 1D tensor
        policy_loss = self.policy_loss(policy_pred, policy_target[~mask]) * self.lamb
        value_loss = self.value_loss(value_pred, value_target)

        self.log('train_policy_loss', policy_loss)
        self.log('train_value_loss', value_loss)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr)
        return {'loss': value_loss + policy_loss}
    '''    
    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)
    '''
    def configure_optimizers(self):
        optimizer =  optim.AdamW(self.parameters(), lr=1e-2, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    @torch.no_grad() # generating selfplay data need no grad.
    def selfplay(self, iter=1):
        data_states = []
        data_probs = []
        data_values = []
        for _ in range(iter):

            # init game
            mcts = MCTS(self, 
                        selfplay=True, 
                        search_limits=self.search_limits)
            gomoku = Gomoku(self.board_height, self.board_width, self.k)
            gomoku.init_game()
            step = 0    

            # simulating game til end
            while True:
                mcts.search(gomoku)
                action_probs = mcts.action_probs()
                
                data_states.append(gomoku.to_model_input()) # before place a stone
                data_probs.append(action_probs) # logits
                
                loc = mcts.sample_action(action_probs)
                end = gomoku.step(loc, check_terminate=True)
                step += 1

                if end == Gomoku.TIE:
                    data_value = [torch.tensor([0], dtype=torch.float32) for _ in range(step)]
                    data_values.extend(data_value)
                    break 
                elif end == Gomoku.WIN:
                    data_value = [torch.tensor([1], dtype=torch.float32) if _%2==0 \
                                else torch.tensor([-1], dtype=torch.float32) for _ in range(step)]
                    data_value = data_value[::-1]
                    data_values.extend(data_value)
                    break
                
                mcts.root = mcts.root.childs.get(loc, Node(None, loc))  # step forward
                # TODO: clear cache?
        
        return GomukuDataset(data_states, data_probs, data_values)
    
    def train_dataloader(self):
        dataset = self.selfplay()
        return DataLoader(dataset, 
                          batch_size=len(dataset),
                          num_workers=8) 
    
    def val_dataloader(self):
        dataset = self.selfplay(iter=10)
        return DataLoader(dataset, 
                          batch_size=len(dataset),
                          num_workers=8) 
