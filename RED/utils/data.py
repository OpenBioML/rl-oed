class RT3DDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size()

    def __getitem__(self, idx):
        return self.data[idx]

class DRPGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.size()

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
