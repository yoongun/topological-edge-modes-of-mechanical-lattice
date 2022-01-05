import numpy as np
from numpy import linalg as la
from typing import List, Tuple


class TriangularLattice:
    def __init__(self, k: List[float], m: List[float], precision: float = .01) -> None:
        """
        Represents dynamic system of 2 dimensional triangular lattice.

        :param k: Spring constant (2)
        :param m: Mass (3)
        :param precision: Precision for wavenumber q
        """
        self.k = k
        self.M = np.diag([m[0], m[0], m[1], m[1], m[2], m[2]])
        self.qxs = np.arange(-np.pi, np.pi, precision)
        self.qys = np.arange(-np.pi, np.pi, precision)

    def Q(self, qx, qy) -> np.ndarray:
        q = np.array([
            1 / 2 * qx + np.sqrt(3) / 2 * qy,
            1 / 2 * qx - np.sqrt(3) / 2 * qy,
            qy
        ])
        Q = np.exp(1.j * q)
        return Q

    @property
    def subspace_r(self) -> np.ndarray:
        r = np.array([
            [[1 / 2], [np.sqrt(3) / 2]],
            [[1 / 2], [-np.sqrt(3) / 2]],
            [[1.], [0.]]
        ])
        subspaces = np.array([
            np.outer(r[0], r[0]),
            np.outer(r[1], r[1]),
            np.outer(r[2], r[2])])
        return subspaces

    def H(self, qx, qy) -> np.ndarray:
        """
        Hamiltonian

        :return: Hamiltonian defined given k and qx, qy
        """
        k = self.k
        Q = self.Q(qx, qy)
        s = self.subspace_r

        alpha = k[1] * (s[0] + s[1] + s[2]) + k[0] * (s[0] + s[1] + s[2])
        h12 = -k[1] * (Q[2] * s[1] + Q[0] * s[2]) - k[0] * s[0]
        h13 = -k[1] * (Q[1].conj() * s[2] + Q[2] * s[0]) - k[0] * s[1]
        h23 = -k[1] * (Q[0].conj() * s[0] + Q[1].conj() * s[1]) - k[0] * s[2]
        H = np.vstack([
            np.hstack([alpha + s[2] * (k[1] - k[0]), h12, h13]),
            np.hstack([h12.conj(), alpha + s[1] * (k[1] - k[0]), h23]),
            np.hstack([h13.conj(), h23.conj(), alpha + s[0] * (k[1] - k[0])])
        ])

        return H

    def dispersion(self) -> List[Tuple[float, float]]:
        """
        Calculate the dispersion relation

        :return: List of angular frequency omega for each q (wavenumber)
        """
        M_inv = la.inv(self.M)
        ws = np.empty((len(self.qys), len(self.qxs), 6))
        evecs = np.empty((len(self.qys), len(self.qxs), 6, 6),
                         dtype=np.complex128)

        for y, qy in enumerate(self.qys):
            for x, qx in enumerate(self.qxs):
                eval_, evec = self._sort_eigen(
                    M_inv.dot(self.H(qx, qy)))
                ws[y, x] = np.sqrt(np.array(eval_.real))
                evecs[y, x] = evec
        return ws, evecs

    def _sort_eigen(self, mat: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Return sorted eigenvalue, eigenvector pair.

        :return: eigenvalue, eigenvector
        """
        eigenvals, eigenvecs = la.eig(mat)
        sorted_idx = np.argsort(eigenvals)
        return eigenvals[sorted_idx], eigenvecs[sorted_idx]
