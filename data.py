import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encoded, block_size=128):
        self.data = encoded
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x), torch.tensor(y)
