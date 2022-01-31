from enum import Enum
from torch.utils.data import Dataset

WaveNumberType = Enum('WaveNumberType', 'XY GKM')


class WaveNumber(Dataset):
    def __init__(self, type: WaveNumberType):
        if type == WaveNumberType.XY:

        ks = []

        self.ks =
        ...

    def __len__(self):
        return len(self.ks)

    def __getitem__(self, idx):
        return self.ks[idx]
