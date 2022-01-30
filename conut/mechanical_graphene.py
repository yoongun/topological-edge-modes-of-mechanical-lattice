import torch
import torch.nn as nn
import numpy as np
from conut.util import σ


class MechanicalGrapheneLattice:
    def __init__(self, l: float, α: float) -> None:
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

        # Sublattice vectors
        self.a1 = np.sqrt(3) * l * x
        self.a2 = (np.sqrt(3) * x + 3 * y) * l / 2.

        # Translation vectors
        self.r1 = 1 / 3 * (self.a1 + self.a2)
        self.r2 = 1 / 3 * (-2 * self.a1 + self.a2)
        self.r3 = 1 / 3 * (self.a1 - 2 * self.a2)
        self.r1h = self.r1 / torch.norm(self.r1)
        self.r2h = self.r2 / torch.norm(self.r2)
        self.r3h = self.r3 / torch.norm(self.r3)
        self.r11 = self.r1h * self.r1h.conj().T
        self.r22 = self.r2h * self.r2h.conj().T
        self.r33 = self.r3h * self.r3h.conj().T

        # Manipulation of Dirac Cones in Mechanical
        # Graphene Toshikaze Kariyado & Yasuhiro Hatsugai (2015)
        z = torch.tensor([[0, 1], [-1, 0]])
        r1_ = z.mm(self.r1h)
        r2_ = z.mm(self.r2h)
        r3_ = z.mm(self.r3h)
        r11_ = r1_ * r1_.conj().T
        r22_ = r2_ * r2_.conj().T
        r33_ = r3_ * r3_.conj().T
        self.γ1 = (1 - α) * r11_ + self.r11
        self.γ2 = (1 - α) * r22_ + self.r22
        self.γ3 = (1 - α) * r33_ + self.r33

        # Dispersion relation G1 -> K -> M -> G2
        b1 = 2 * np.pi / np.sqrt(3) / l * (x - y / np.sqrt(3))
        b2 = 4 * np.pi / 3 / l * y
        self.G = torch.tensor([[0.], [0.]])
        self.K = (b1 / 2 + b2 / 4) / np.cos(np.pi / 6)**2
        self.M = b1 / 2

        self.GK = (self.K - self.G)
        self.KM = (self.M - self.K)
        self.MG = (self.G - self.M)
        self.gk = self.GK / torch.norm(self.GK)
        self.km = self.KM / torch.norm(self.KM)
        self.mg = self.MG / torch.norm(self.MG)


class MGBulkH(nn.Module):
    def __init__(self, lattice: MechanicalGrapheneLattice, ω0: float, α: float, perturbation=False) -> None:
        super(MGBulkH, self).__init__()
        self.lat = lattice
        self.ω0 = ω0
        self.α = α
        self.perturbation = perturbation

    def forward(self, k: torch.tensor, Ω: float):
        r11, r22, r33 = self.lat.r11, self.lat.r22, self.lat.r33
        γ1, γ2, γ3 = self.lat.γ1, self.lat.γ2, self.lat.γ3
        K1 = torch.exp(1.j * k.conj().T.mm(self.lat.a1))
        K2 = torch.exp(1.j * k.conj().T.mm(self.lat.a2))

        if self.perturbation:
            H = self.ω0**2 * np.vstack([
                np.hstack([(r11 + r22 + r33) * (2 - self.α) - 2 * Ω * σ.y / self.ω0**2,
                           -(γ1 + K1.conj() * γ2 + K2.conj() * γ3)]),
                np.hstack([-(γ1 + K1 * γ2 + K2 * γ3),
                           (r11 + r22 + r33) * (2 - self.α) - 2 * Ω * σ.y / self.ω0**2]),
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
    def __init__(self, κ: float, α: float, m: float, l: float, Ω=0., precision=1e-1, perturbation=False) -> None:
        """
        :param κ:
        :param M:
        :param l:
        :param precision:
        """
        super(MechanicalGrapheneBulk, self).__init__()
        ω0 = np.sqrt(κ / m)
        self.lat = MechanicalGrapheneLattice(l, α, precision)
        self.kxs = torch.arange(-self.lat.xw, self.lat.xw, precision)
        self.kys = torch.arange(-self.lat.yw, self.lat.yw, precision)
        num_gk = int(np.linalg.norm(self.lat.GK) / precision)
        num_km = int(np.linalg.norm(self.lat.KM) / precision)
        num_mg = int(np.linalg.norm(self.lat.MG) / precision) + 1
        self.kxs_gkmg = np.hstack([
            [self.lat.gk[0, 0] * precision * i for i in range(num_gk)],
            [self.lat.km[0, 0] * precision * i + self.lat.GK[0, 0]
                for i in range(num_km)],
            [self.lat.mg[0, 0] * precision * i + self.lat.GK[0, 0] + self.lat.KM[0, 0]
                for i in range(num_mg)]
        ])
        self.kys_gkmg = np.hstack([
            [self.lat.gk[1, 0] * precision * i for i in range(num_gk)],
            [self.lat.km[1, 0] * precision * i + self.lat.GK[1, 0]
                for i in range(num_km)],
            [self.lat.mg[1, 0] * precision * i + self.lat.GK[1, 0] + self.lat.KM[1, 0]
                for i in range(num_mg)]
        ])
        self.h = MGBulkH(self.lat, ω0, α, perturbation=perturbation)
        self.Ω = Ω
        self.perturbation = perturbation

        self.evals_all = torch.zeros(
            (len(self.kys), len(self.kxs), 8), dtype=torch.cdouble)
        self.evecs_all = torch.zeros(
            (len(self.kys), len(self.kxs), 8, 8), dtype=torch.cdouble)

    def forward(self, gkmg=False):
        if gkmg:
            evals_all = []
            evecs_all = []
            for ky, kx in zip(self.kys_gkmg, self.kxs_gkmg):
                k = torch.tensor([[kx], [ky]])
                evals, evecs = torch.linalg.eig(self.h(k, self.Ω))
                idcs = torch.argsort(evals.real)
                evals, evecs = evals[idcs], evecs[idcs]
                self.evals_all[y, x] = evals
                self.evecs_all[y, x] = evecs

        for y, ky in enumerate(self.kys):
            for x, kx in enumerate(self.kxs):
                k = torch.tensor([[kx], [ky]])
                evals, evecs = torch.linalg.eig(self.h(k, self.Ω))
                idcs = torch.argsort(evals.real)
                evals, evecs = evals[idcs], evecs[idcs]
                self.evals_all[y, x] = evals
                self.evecs_all[y, x] = evecs
        if self.perturbation:
            self.evals_all = torch.sqrt(self.evecs_all)
        return self.evals_all.real, self.evecs_all

    def dispersion(self, gkmg=False):
        evals, evecs = self.forward(gkmg)
        return evals.numpy(), evecs.numpy()


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
