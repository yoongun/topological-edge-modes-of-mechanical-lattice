from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from conut.util import σ


Mode = Enum('Mode', ['Bulk', 'Ribbon'])


class MechanicalGrapheneLattice:
    def __init__(self, l: float, α: float) -> None:
        """
        :param l:
        """
        # Width of x, y in brillouin zone.
        self.xw = np.pi / np.sqrt(3) * 2 / l
        self.yw = np.pi / 3 * 4 / l

        x = torch.tensor([[1.], [0.]], dtype=torch.cdouble)  # x hat
        y = torch.tensor([[0.], [1.]], dtype=torch.cdouble)  # y hat

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
        z = torch.tensor([[0, 1], [-1, 0]], dtype=torch.cdouble)
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
        self.b1 = 2 * np.pi / np.sqrt(3) / l * (x - y / np.sqrt(3))
        self.b2 = 4 * np.pi / 3 / l * y
        self.G = torch.tensor([[0.], [0.]])
        self.K = (self.b1 / 2 + self.b2 / 4) / np.cos(np.pi / 6)**2
        self.M = self.b1 / 2

        self.GK = (self.K - self.G)
        self.KM = (self.M - self.K)
        self.MG = (self.G - self.M)
        self.gk = self.GK / torch.norm(self.GK)
        self.km = self.KM / torch.norm(self.KM)
        self.mg = self.MG / torch.norm(self.MG)


class BulkH(nn.Module):
    def __init__(self, lattice: MechanicalGrapheneLattice, ω0: float, α: float, Ω: float, perturbation=False) -> None:
        super(BulkH, self).__init__()
        self.lat = lattice
        self.ω0 = ω0
        self.α = α
        self.Ω = Ω
        self.perturbation = perturbation

    def forward(self, k: torch.tensor):
        r11, r22, r33 = self.lat.r11, self.lat.r22, self.lat.r33
        γ1, γ2, γ3 = self.lat.γ1, self.lat.γ2, self.lat.γ3
        K1 = torch.exp(1.j * k.conj().T.mm(self.lat.a1))
        K2 = torch.exp(1.j * k.conj().T.mm(self.lat.a2))

        if self.perturbation:
            H = self.ω0**2 * torch.vstack([
                torch.hstack([(r11 + r22 + r33) * (2 - self.α) - 2 * self.Ω * σ.y / self.ω0**2,
                              -(γ1 + K1.conj() * γ2 + K2.conj() * γ3)]),
                torch.hstack([-(γ1 + K1 * γ2 + K2 * γ3),
                              (r11 + r22 + r33) * (2 - self.α) - 2 * self.Ω * σ.y / self.ω0**2]),
            ])
            return H
        L11 = torch.vstack([
            torch.hstack(
                [r11 + r22 + r33, -(r11 + K1.conj() * r22 + K2.conj() * r33)]),
            torch.hstack([-(r11 + K1 * r22 + K2 * r33), r11 + r22 + r33])
        ])
        L12 = torch.vstack([
            torch.hstack([-2 * self.Ω * σ.y, torch.zeros((2, 2))]),
            torch.hstack([torch.zeros(2, 2), -2 * self.Ω * σ.y])
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


class RibbonH(nn.Module):
    def __init__(self, lattice: MechanicalGrapheneLattice, ω0: float, α: float, Ω: float, perturbation=False) -> None:
        super(BulkH, self).__init__()
        self.lat = lattice
        self.ω0 = ω0
        self.α = α
        self.perturbation = perturbation

    def forward(self, k: torch.tensor, Ω: float):
        return


class MechanicalGraphene(nn.Module):
    def __init__(self, κ: float, α: float, m: float, l: float, mode: Mode, Ω=0., precision=1e-1, perturbation=False) -> None:
        """
        :param κ: Spring constant (N/m).
        :param α: R_0 / l_0.
        :param m: Mass (kg).
        :param l: Distance between masses (m).
        :param mode: Mode of Hamiltonian (Bulk | Ribbon)
        :param Ω: Roation frequency (default 0 Hz).
        :param precision: Precision of wavenumber (default 1e-1).
        :param perturbation: On/off perturbation approximation of Coriolis force (default False).
        """
        super(MechanicalGraphene, self).__init__()
        # Parameter of the model
        self.κ = κ
        self.α = α
        self.m = m
        self.l = l
        self.mode = mode
        self.Ω = Ω
        self.precision = precision
        self.perturbation = perturbation

        # Setup lattice
        self.lat = MechanicalGrapheneLattice(l, α)

        # Setup wavenumbers
        self.kxs = torch.arange(-self.lat.xw, self.lat.xw, precision)
        self.kys = torch.arange(-self.lat.yw, self.lat.yw, precision)

        # Wavenumber for 2d dispersion relation
        self.num_gk = int(np.linalg.norm(self.lat.GK) / precision)
        self.num_km = int(np.linalg.norm(self.lat.KM) / precision)
        self.num_mg = int(np.linalg.norm(self.lat.MG) / precision) + 1
        self.kxs_gkmg = np.hstack([
            [self.lat.gk[0, 0] * precision * i for i in range(self.num_gk)],
            [self.lat.km[0, 0] * precision * i + self.lat.GK[0, 0]
                for i in range(self.num_km)],
            [self.lat.mg[0, 0] * precision * i + self.lat.GK[0, 0] + self.lat.KM[0, 0]
                for i in range(self.num_mg)]
        ])
        self.kys_gkmg = np.hstack([
            [self.lat.gk[1, 0] * precision * i for i in range(self.num_gk)],
            [self.lat.km[1, 0] * precision * i + self.lat.GK[1, 0]
                for i in range(self.num_km)],
            [self.lat.mg[1, 0] * precision * i + self.lat.GK[1, 0] + self.lat.KM[1, 0]
                for i in range(self.num_mg)]
        ])

        ω0 = np.sqrt(κ / m)
        if mode == Mode.Bulk:
            self.h = BulkH(lattice=self.lat, ω0=ω0, α=α,
                           Ω=Ω, perturbation=perturbation)
        elif mode == Mode.Ribbon:
            self.h = RibbonH(lattice=self.lat, ω0=ω0, α=α,
                             Ω=Ω, perturbation=perturbation)
        else:
            raise AttributeError("Mode not supported.")
        self.perturbation = perturbation
        dim = 8
        if perturbation:
            dim = 4
        self.evals = torch.zeros(
            (len(self.kys), len(self.kxs), dim), dtype=torch.cdouble)
        self.evecs = torch.zeros(
            (len(self.kys), len(self.kxs), dim, dim), dtype=torch.cdouble)

        self.evals_gkmg = torch.zeros(
            (len(self.kys_gkmg), dim), dtype=torch.cdouble)
        self.evecs_gkmg = torch.zeros(
            (len(self.kys_gkmg), dim, dim), dtype=torch.cdouble)

    #     self.forward()
    #     self.forward(True)

    # def forward(self, gkmg=False):
    #     if gkmg:
    #         for i, (ky, kx) in enumerate(zip(self.kys_gkmg, self.kxs_gkmg)):
    #             k = torch.tensor([[kx], [ky]], dtype=torch.cdouble)
    #             evals, evecs = torch.linalg.eig(self.h(k))
    #             idcs = torch.argsort(evals.real)
    #             evals, evecs = evals[idcs], evecs[idcs]
    #             self.evals_gkmg[i] = evals
    #             self.evecs_gkmg[i] = evecs
    #         if self.perturbation:
    #             self.evals_gkmg = torch.sqrt(self.evals_gkmg)
    #         return self.evals_gkmg.real, self.evecs_gkmg

    #     for y, ky in enumerate(self.kys):
    #         for x, kx in enumerate(self.kxs):
    #             k = torch.tensor([[kx], [ky]], dtype=torch.cdouble)
    #             evals, evecs = torch.linalg.eig(self.h(k))
    #             idcs = torch.argsort(evals.real)
    #             evals, evecs = evals[idcs], evecs[idcs]
    #             self.evals[y, x] = evals
    #             self.evecs[y, x] = evecs
    #     if self.perturbation:
    #         self.evals = torch.sqrt(self.evals)
    #     return self.evals.real, self.evecs

    def forward(self, gkmg=False):
        if gkmg:
            for i, (ky, kx) in enumerate(zip(self.kys_gkmg, self.kxs_gkmg)):
                k = torch.tensor([[kx], [ky]], dtype=torch.cdouble)
                evals, evecs = torch.linalg.eig(self.h(k))
                idcs = torch.argsort(evals.real)
                evals, evecs = evals[idcs], evecs[idcs]
                self.evals_gkmg[i] = evals
                self.evecs_gkmg[i] = evecs
            if self.perturbation:
                self.evals_gkmg = torch.sqrt(self.evals_gkmg)
            return self.evals_gkmg.real, self.evecs_gkmg

        for y, ky in enumerate(self.kys):
            for x, kx in enumerate(self.kxs):
                k = torch.tensor([[kx], [ky]], dtype=torch.cdouble)
                evals, evecs = torch.linalg.eig(self.h(k))
                idcs = torch.argsort(evals.real)
                evals, evecs = evals[idcs], evecs[idcs]
                self.evals[y, x] = evals
                self.evecs[y, x] = evecs
        if self.perturbation:
            self.evals = torch.sqrt(self.evals)
        return self.evals.real, self.evecs

    def dispersion(self, gkmg=False):
        if gkmg:
            return self.evals_gkmg.real.numpy(), self.evecs_gkmg.numpy()
        return self.evals.real.numpy(), self.evecs.numpy()
