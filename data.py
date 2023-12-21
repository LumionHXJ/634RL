from torch.utils.data import Dataset


class GomukuDataset(Dataset):
    """Gomoku data, can be generate by selfplay or download from botzone(TODO).
    """
    def __init__(self, 
                 states=None, 
                 probs=None, 
                 values=None):
        """
        Args:
            states, probs, values are generate by selfplay, both are list of tensor.
        """
        self.states = states # n, (4, h, w)
        self.probs = probs # n, (h*w, )
        self.values = values # n, (1,)
    
    @classmethod
    def merge(cls, dataset_list):
        states = []
        probs = []
        values = []
        for dataset in dataset_list:
            states.extend(dataset.states)
            probs.extend(dataset.probs)
            values.extend(dataset.values)
        return GomukuDataset(states, probs, values)
  
    def __getitem__(self, i):
        return self.states[i], self.probs[i], self.values[i]
    
    def __len__(self):
        return len(self.values)