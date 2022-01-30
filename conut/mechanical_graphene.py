import torch
import torch.nn as nn
import numpy as np
from conut.util import σ


class MechanicalGrapheneLattice:
    def __init__(self, l: float) -> None:
        """
        :param κ:
        :param M:
        :param l:
        :param precision:
        """
        # Width of x, y
        self.xw = np.pi / np.sqrt(3) * 2
        self.yw = np.pi / 3 * 4

        x = torch.tensor([[1.], [0.]])  # x hat
        y = torch.tensor([[0.], [1.]])  # y hat

        # Sublattice vector
        self.a1 = np.sqrt(3) * l * x
        self.a2 = (np.sqrt(3) * x + 3 * y) * l / 2.

        self.r1 = 1 / 3 * (self.a1 + self.a2)
        self.r2 = 1 / 3 * (-2 * self.a1 + self.a2)
        self.r3 = 1 / 3 * (self.a1 - 2 * self.a2)
        self.r1h = self.r1 / torch.norm(self.r1)
        self.r2h = self.r2 / torch.norm(self.r2)
        self.r3h = self.r3 / torch.norm(self.r3)
        self.r11 = self.r1h * self.r1h.conj().T
        self.r22 = self.r2h * self.r2h.conj().T
        self.r33 = self.r3h * self.r3h.conj().T


class MGBulkH(nn.Module):
    def __init__(self, lattice: MechanicalGrapheneLattice, ω0: float, perturbation=False) -> None:
        super(MGBulkH, self).__init__()
        self.lat = lattice
        self.ω0 = ω0
        self.perturbation = perturbation

    def forward(self, k: torch.tensor, Ω: float):
        r11, r22, r33 = self.lat.r11, self.lat.r22, self.lat.r33
        K1 = torch.exp(1.j * k.conj().T.mm(self.lat.a1))
        K2 = torch.exp(1.j * k.conj().T.mm(self.lat.a2))

        if self.perturbation:
            H = self.ω0**2 * np.vstack([
                np.hstack([r11 + r22 + r33 - 2 * Ω * σ.y / self.ω0**2,
                           -(r11 + K1.conj() * r22 + K2.conj() * r33)]),
                np.hstack([-(r11 + K1 * r22 + K2 * r33),
                           r11 + r22 + r33 - 2 * Ω * σ.y / self.ω0**2]),
            ])
            return H
        L11 = torch.vstack([
            torch.hstack(
                [r11 + r22 + r33, -(r11 + K1.conj() * r22 + K2.conj() * r33)]),
            torch.hstack([-(r11 + K1 * r22 + K2 * r33), r11 + r22 + r33])
        ])
        L12 = torch.vstack([
            torch.hstack([-2 * Ω * σ.y, torch.zeros((2, 2))]),
            torch.hstack([torch.zeros(2, 2), -2 * Ω * σ.y])
        ])
        L = self.ω0**2 * torch.vstack([
            torch.hstack([L11, L12]),
            torch.hstack([torch.zeros(4, 4), torch.eye(4)])
        ])
        M = torch.vstack([
            torch.hstack([torch.zeros(4, 4), torch.eye(4)]),
            torch.hstack([torch.eye(4), torch.zeros(4, 4)])
        ]).to(torch.cdouble)

        H = torch.inverse(M).mm(L)
        return H


class MechanicalGrapheneBulk(nn.Module):
    def __init__(self, κ: float, m: float, l: float, precision=1e-1, perturbation=False) -> None:
        """
        :param κ:
        :param M:
        :param l:
        :param precision:
        """
        super(MechanicalGrapheneBulk, self).__init__()
        ω0 = np.sqrt(κ / m)
        self.lat = MechanicalGrapheneLattice(l)
        self.kxs = torch.arange(-self.lat.xw, self.lat.xw, precision)
        self.kys = torch.arange(-self.lat.yw, self.lat.yw, precision)
        self.h = MGBulkH(self.lat, ω0, perturbation=perturbation)

        self.evals_all = torch.zeros(
            (len(self.kys), len(self.kxs), 8), dtype=torch.cdouble)
        self.evecs_all = torch.zeros(
            (len(self.kys), len(self.kxs), 8, 8), dtype=torch.cdouble)

    def forward(self):
        for y, ky in enumerate(self.kys):
            for x, kx in enumerate(self.kxs):
                k = torch.tensor([[kx], [ky]])
                evals, evecs = torch.linalg.eig(self.h(k, 0.))
                idcs = torch.argsort(evals.real)
                evals, evecs = evals[idcs], evecs[idcs]
                self.evals_all[y, x] = evals
                self.evecs_all[y, x] = evecs
        return self.evals_all, self.evecs_all


class MechanicalGrapheneRibbon:
    def __init__(self, C: float, M: float, l: float, precision=1e-1) -> None:
        """

        :param C:
        :param M:
        :param l:
        :param precision:
        """

        pass

    def h(self):
        ...
