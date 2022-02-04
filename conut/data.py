from enum import Enum
import numpy as np
import torch
from torch.utils.data import Dataset
from conut import MechanicalGrapheneLattice


WaveNumberType = Enum('WaveNumberType', 'XY GKM')


class WaveNumber(Dataset):
    def __init__(
            self,
            wn_type: WaveNumberType,
            lattice: MechanicalGrapheneLattice,
            precision=1e-1) -> None:
        self.precision = precision
        ks = []
        idcs = []
        if wn_type == WaveNumberType.XY:
            self.kxs = torch.arange(-lattice.xw, lattice.xw, precision)
            self.kys = torch.arange(-lattice.yw, lattice.yw, precision)
            for y, ky in enumerate(self.kys):
                for x, kx in enumerate(self.kxs):
                    ks.append([[kx], [ky]])
                    idcs.append([y, x])
            self.shape = (len(self.kys), len(self.kxs))
        elif wn_type == WaveNumberType.GKM:
            self.num_gk = int(np.linalg.norm(lattice.GK.cpu()) / precision)
            self.num_km = int(np.linalg.norm(lattice.KM.cpu()) / precision)
            self.num_mg = int(np.linalg.norm(
                lattice.MG.cpu()) / precision) + 1
            self.kxs = np.hstack([
                [lattice.gk.cpu()[0, 0] * precision * i for i in range(self.num_gk)],
                [lattice.km.cpu()[0, 0] * precision * i
                    + lattice.GK.cpu()[0, 0]
                    for i in range(self.num_km)],
                [lattice.mg.cpu()[0, 0] * precision * i
                    + lattice.GK.cpu()[0, 0]
                    + lattice.KM.cpu()[0, 0]
                    for i in range(self.num_mg)]
            ])
            self.kys = np.hstack([
                [lattice.gk.cpu()[1, 0] * precision * i for i in range(self.num_gk)],
                [lattice.km.cpu()[1, 0] * precision * i
                    + lattice.GK.cpu()[1, 0]
                    for i in range(self.num_km)],
                [lattice.mg.cpu()[1, 0] * precision * i
                    + lattice.GK.cpu()[1, 0]
                    + lattice.KM.cpu()[1, 0]
                    for i in range(self.num_mg)]
            ])
            for i, (ky, kx) in enumerate(zip(self.kys, self.kxs)):
                ks.append([[kx], [ky]])
                idcs.append(i)
            self.shape = (len(self.kys))
        else:
            raise AttributeError()
        self.ks = torch.tensor(ks, dtype=torch.cdouble)
        self.idcs = torch.tensor(idcs)

    def __len__(self):
        return len(self.ks)

    def __getitem__(self, idx):
        return self.idcs[idx], self.ks[idx]
