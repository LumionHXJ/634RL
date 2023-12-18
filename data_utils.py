from torch.utils.data import Dataset


class GomukuDataset(Dataset):
    """Gomoku data, can be generate by selfplay or download from botzone.
    """
    def __init__(self, 
                 states=None, 
                 probs=None, 
                 values=None, 
                 offline_data_dir=None):
        """
        Args:
            states, probs, values are generate by selfplay.
            offline_data_dir save data download from botzone.
            states(list of tensor)
        """
        self.states = states # n, 4, h, w
        self.probs = probs # n, h*w
        self.values = values # n,
    
    def __getitem__(self, i):
        return self.states[i], self.probs[i], self.values[i]
    
    def __len__(self):
        return len(self.values)
