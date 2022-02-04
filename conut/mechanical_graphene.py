from enum import Enum
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from conut.util import pauli


HamiltonianType = Enum('HamiltonianType', ['Bulk', 'Ribbon'])
σ = pauli('cpu')


class MechanicalGrapheneLattice:
    def __init__(self, l: float, α: float, device: str) -> None:
        """
        :param l:
        """
        self.l = l
        self.α = α
        # Width of x, y in brillouin zone.
        self.xw = np.pi / np.sqrt(3) * 2 / l
        self.yw = np.pi / 3 * 4 / l

        x = torch.tensor([[1.], [0.]], dtype=torch.cdouble).to(device)  # x hat
        y = torch.tensor([[0.], [1.]], dtype=torch.cdouble).to(device)  # y hat

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
        z = torch.tensor([[0, 1], [-1, 0]], dtype=torch.cdouble).to(device)
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
        self.G = torch.tensor([[0.], [0.]]).to(device)
        self.K = (self.b1 / 2 + self.b2 / 4) / np.cos(np.pi / 6)**2
        self.M = self.b1 / 2

        self.GK = (self.K - self.G)
        self.KM = (self.M - self.K)
        self.MG = (self.G - self.M)
        self.gk = self.GK / torch.norm(self.GK)
        self.km = self.KM / torch.norm(self.KM)
        self.mg = self.MG / torch.norm(self.MG)


class BulkHamiltonian(nn.Module):
    def __init__(
            self,
            lattice: MechanicalGrapheneLattice,
            ω0: float,
            α: float,
            Ω: float,
            perturbation=False) -> None:
        super(BulkHamiltonian, self).__init__()
        self.lat = lattice
        self.ω0 = ω0
        self.α = α
        self.Ω = Ω
        self.perturbation = perturbation

        # self.z2 = Parameter(torch.zeros(2, 2))
        # self.z4 = Parameter(torch.zeros(4, 4))
        # self.i4 = Parameter(torch.eye(4))
        self.L12 = torch.vstack([
            torch.hstack([-2 * self.Ω * σ.y, torch.zeros(2, 2)]),
            torch.hstack([torch.zeros(2, 2), -2 * self.Ω * σ.y])
        ])
        self.M = torch.vstack([
            torch.hstack([torch.zeros(4, 4), torch.eye(4)]),
            torch.hstack([torch.eye(4), torch.zeros(4, 4)])
        ]).to(torch.cdouble)

    def forward(self, k: torch.tensor):
        batch_size = len(k)
        r11, r22, r33 = self.lat.r11, self.lat.r22, self.lat.r33
        γ1, γ2, γ3 = self.lat.γ1, self.lat.γ2, self.lat.γ3
        K1 = torch.exp(
            1.j * k.conj().permute(0, 2, 1).bmm(
                self.lat.a1.repeat(batch_size, 1, 1)))
        K2 = torch.exp(
            1.j * k.conj().permute(0, 2, 1).bmm(
                self.lat.a2.repeat(batch_size, 1, 1)))
        K1 = torch.squeeze(K1)
        K2 = torch.squeeze(K2)
        if len(K1.shape) == 0:
            K1 = torch.unsqueeze(K1, 0)
            K2 = torch.unsqueeze(K2, 0)

        if self.perturbation:
            Hs = torch.zeros((batch_size, 4, 4), dtype=torch.cdouble)
            for i, (K1_, K2_) in enumerate(zip(K1, K2)):
                σ = pauli('cpu')
                H = self.ω0**2 * torch.vstack([
                    torch.hstack([(r11 + r22 + r33) * (2 - self.α)
                                  - 2 * self.Ω * σ.y / self.ω0**2,
                                  -(γ1 + K1_.conj() * γ2 + K2_.conj() * γ3)]),
                    torch.hstack([-(γ1 + K1_ * γ2 + K2_ * γ3),
                                  (r11 + r22 + r33) * (2 - self.α)
                                  - 2 * self.Ω * σ.y / self.ω0**2]),
                ])
                Hs[i] = H
            return Hs

        Hs = torch.zeros((batch_size, 8, 8), dtype=torch.cdouble)

        for i, (K1_, K2_) in enumerate(zip(K1, K2)):
            L11 = torch.vstack([
                torch.hstack(
                    [r11 + r22 + r33, -(r11 + K1_.conj() * r22 + K2_.conj() * r33)]),
                torch.hstack(
                    [-(r11 + K1_ * r22 + K2_ * r33), r11 + r22 + r33])
            ])
            L = self.ω0**2 * torch.vstack([
                torch.hstack([L11, self.L12]),
                torch.hstack([self.z4, self.i4])
            ])
            H = torch.inverse(self.M).mm(L)
            Hs[i] = H
        return Hs


class RibbonHamiltonian(nn.Module):
    def __init__(
            self,
            lattice: MechanicalGrapheneLattice,
            ω0: float,
            α: float,
            Ω: float,
            perturbation=False) -> None:
        super(BulkHamiltonian, self).__init__()
        self.lat = lattice
        self.ω0 = ω0
        self.α = α
        self.perturbation = perturbation

    def forward(self, k: torch.tensor, Ω: float):
        return


class MechanicalGraphene(nn.Module):
    def __init__(
            self,
            κ: float,
            α: float,
            m: float,
            lattice: MechanicalGrapheneLattice,
            h_type: HamiltonianType,
            shape: int,
            Ω=0.,
            GKM=False,
            perturbation=False) -> None:
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
        self.lattice = lattice
        self.h_type = h_type
        self.Ω = Ω
        self.GKM = GKM
        self.perturbation = perturbation

        ω0 = np.sqrt(κ / m)
        if h_type == HamiltonianType.Bulk:
            self.h = BulkHamiltonian(lattice=lattice, ω0=ω0, α=α,
                                     Ω=Ω, perturbation=perturbation)
        elif h_type == HamiltonianType.Ribbon:
            self.h = RibbonHamiltonian(lattice=lattice, ω0=ω0, α=α,
                                       Ω=Ω, perturbation=perturbation)
        else:
            raise AttributeError("Mode not supported.")

        dim = 8
        if perturbation:
            dim = 4
        if GKM:
            self.evals = torch.zeros(
                (shape, dim), dtype=torch.cdouble)
            self.evecs = torch.zeros(
                (shape, dim, dim), dtype=torch.cdouble)
        else:
            self.evals = torch.zeros(
                (shape[0], shape[1], dim), dtype=torch.cdouble)
            self.evecs = torch.zeros(
                (shape[0], shape[1], dim, dim), dtype=torch.cdouble)

    def forward(self, idcs, ks):
        evals, evecs = torch.linalg.eig(self.h(ks))
        if self.GKM:
            for i, idx in enumerate(idcs):
                ia = torch.argsort(evals[i].real)
                self.evals[idx] = evals[i][ia]
                self.evecs[idx] = evecs[i][ia]
        else:
            for i, idx in enumerate(idcs):
                y, x = idx[0], idx[1]
                ia = torch.argsort(evals[i].real)
                self.evals[y, x] = evals[i][ia]
                self.evecs[y, x] = evecs[i][ia]
        if self.perturbation:
            self.evals = torch.sqrt(self.evals)
        return None

    def dispersion(self, gkmg=False):
        if gkmg:
            return self.evals_gkmg.real.numpy(), self.evecs_gkmg.numpy()
        return self.evals.real.numpy(), self.evecs.numpy()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.h = self.h.to(*args, **kwargs)
        return self
