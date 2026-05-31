import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, itc_path):
        super().__init__()
        self.itc = pd.read_csv(itc_path)

    def __len__(self):
        return len(self.itc)

    def __getitem__(self, idx):
        return self.itc.iloc[idx].values

    @property
    def labels(self):
        return self.itc.iloc[:, -1].values

    def itc_collate_fn(self, batch):
        drug1 = []
        drug2 = []
        label = []
        for data in batch:
            drug1.append(data[0])
            drug2.append(data[1])
            label.append(data[2])
        return torch.tensor(drug1), torch.tensor(drug2), torch.tensor(label)
