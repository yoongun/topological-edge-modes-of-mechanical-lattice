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

        x = np.array([[1.], [0.]])  # x hat
        y = np.array([[0.], [1.]])  # y hat

        # Sublattice vector
        self.a1 = np.sqrt(3) * l * x
        self.a2 = (np.sqrt(3) * x + 3 * y) * l / 2.

        self.r1 = 1 / 3 * (self.a1 + self.a2)
        self.r2 = 1 / 3 * (-2 * self.a1 + self.a2)
        self.r3 = 1 / 3 * (self.a1 - 2 * self.a2)
        self.r1h = self.r1 / np.linalg.norm(self.r1)
        self.r2h = self.r2 / np.linalg.norm(self.r2)
        self.r3h = self.r3 / np.linalg.norm(self.r3)
        self.r11 = self.r1h * self.r1h.conj().T
        self.r22 = self.r2h * self.r2h.conj().T
        self.r33 = self.r3h * self.r3h.conj().T


class MGBulkH:
    def __init__(self, lattice: MechanicalGrapheneLattice, w0: float) -> None:
        self.lat = lattice
        self.w0 = w0

    def __call__(self, k: np.ndarray, O: float, perturbation=False):
        r11, r22, r33 = self.lat.r11, self.lat.r22, self.lat.r33
        K1 = np.exp(1.j * k.dot(self.lat.a1))
        K2 = np.exp(1.j * k.dot(self.lat.a2))

        if perturbation:
            H = self.w0**2 * np.vstack([
                np.hstack([r11 + r22 + r33 - 2 * O * σ.y / self.ω0**2,
                           -(r11 + K1.conj() * r22 + K2.conj() * r33)]),
                np.hstack([-(r11 + K1 * r22 + K2 * r33),
                           r11 + r22 + r33 - 2 * O * σ.y / self.ω0**2]),
            ])
            return H
        L11 = np.vstack([
            np.hstack(
                [r11 + r22 + r33, -(r11 + K1.conj() * r22 + K2.conj() * r33)]),
            np.hstack([-(r11 + K1 * r22 + K2 * r33), r11 + r22 + r33])
        ])
        y = np.array([[0., -1.j], [1.j, 0.]])
        L12 = np.vstack([
            np.hstack([-2 * O * y, np.zeros((2, 2))]),
            np.hstack([np.zeros((2, 2)), -2 * O * y])
        ])
        L = self.w0**2 * np.vstack([
            np.hstack([L11, L12]),
            np.hstack([np.zeros((4, 4)), np.eye(4)])
        ])
        M = np.vstack([
            np.hstack([np.zeros((4, 4)), np.eye(4)]),
            np.hstack([np.eye(4), np.zeros((4, 4))])
        ])

        H = np.linalg.inv(M).dot(L)
        return H


class MechanicalGrapheneBulk:
    def __init__(self, c: float, m: float, l: float, precision=1e-1) -> None:
        """
        :param κ:
        :param M:
        :param l:
        :param precision:
        """
        w0 = np.sqrt(c / m)
        lat = MechanicalGrapheneLattice(l)
        self.kxs = np.arange(-lat.xw, lat.xw, precision)
        self.kys = np.arange(-lat.yw, lat.yw, precision)
        self.h = MGBulkH(lat, w0)

        self.evals_all = np.zeros(
            (len(self.kys), len(self.kxs), 8), dtype=np.complex128)
        self.evecs_all = np.zeros(
            (len(self.kys), len(self.kxs), 8, 8), dtype=np.complex128)

    def __call__(self):
        for y, ky in enumerate(self.kys):
            for x, kx in enumerate(self.kxs):
                k = np.array([kx, ky])
                evals, evecs = np.linalg.eig(self.h(k, 0.))
                idcs = np.argsort(evals)
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
