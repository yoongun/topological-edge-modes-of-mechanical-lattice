import numpy as np
from numpy import linalg as la
from typing import List, Tuple


class MechanicalLattice2DSquare:
    def __init__(self, k: float, m: List[float], precision: float = .01) -> None:
        """
        Represents dynamic system of 1 dimensional mechanical lattice.

        :param k: Spring constant
        :param m: Mass
        :param precision: Precision for wavenumber q
        """
        self.k = k
        self.M = np.diag([m[0], m[0], m[1], m[1]])
        self.qxs = np.arange(-np.pi, np.pi, precision)
        self.qys = np.arange(-np.pi, np.pi, precision)

    def H(self, qx, qy):
        """
        Hamiltonian

        :return: Hamiltonian defined given k and qx, qy
        """
        k = self.k
        Q1 = np.exp(1.j * (qx - qy) / np.sqrt(2))  # NE direction
        Q2 = np.exp(1.j * (qx + qy) / np.sqrt(2))  # SE direction
        Q3 = np.exp(1.j * qx)  # E direction
        mat1 = 2 * np.eye(2)
        mat2 = np.array([
            [-1 - Q3.conj(), 0],
            [0, -Q1.conj() - Q2.conj()]
        ])
        mat3 = np.array([
            [-Q3 - 1, 0],
            [0, -Q1 - Q2]
        ])
        mat4 = 2 * np.eye(2)
        H = k * np.vstack([np.hstack([mat1, mat2]), np.hstack([mat3, mat4])])

        return H

    def dispersion(self) -> List[Tuple[float, float]]:
        """
        Calculate the dispersion relation

        :return: List of angular frequency omega for each q (wavenumber)
        """
        M_inv = la.inv(self.M)
        ws = np.empty((len(self.qys), len(self.qxs), 4))
        evecs = np.empty((len(self.qys), len(self.qxs), 4, 4),
                         dtype=np.complex128)

        for y, qy in enumerate(self.qys):
            for x, qx in enumerate(self.qxs):
                eval_, evec = self._sort_eigen(
                    M_inv.dot(self.H(qx, qy)))
                ws[y, x] = np.sqrt(np.array(eval_).real)
                evecs[y, x] = evec
        return ws, evecs

    def _sort_eigen(self, mat: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Return sorted eigenvalue, eigenvector pair.

        :return: eigenvalue, eigenvector
        """
        eigenvals, eigenvecs = la.eig(mat)
        min_idx = np.argsort(eigenvals)
        return eigenvals[min_idx], eigenvecs[min_idx]
