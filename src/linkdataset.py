from torch.utils.data import Dataset

class LinkDataset(Dataset):
    def __init__(self, data):
        super(LinkDataset, self).__init__()
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)