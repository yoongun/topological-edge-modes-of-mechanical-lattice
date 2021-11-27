import numpy as np
from numpy import linalg as la
from typing import List, Tuple


class KagomeLattice:
    """
    Reference:
    "A study of topological effects in 1D and 2D mechanical lattices" H. Chen (2018), et. al.
    from 'Journal of the Mechanics and Physics of Solids', Volum 117, Aug 2018, 22-36,
    https://www.sciencedirect.com/science/article/abs/pii/S0022509618301820
    """

    def __init__(self, k: List[float], m: List[float], precision: float = .01) -> None:
        """
        Represents dynamic system of Kagome lattice.

        :param k: Spring constant (2)
        :param m: Mass (333)
        :param precision: Precision for wavenumber q
        """
        self.k = k
        self.M = np.diag([m[0], m[0], m[1], m[1], m[2], m[2]])
        self.qxs = np.arange(-np.pi, np.pi, precision)
        self.qys = np.arange(-np.pi, np.pi, precision)

    def z(self, q) -> np.ndarray:
        """
        Return z for given wavevector q

        :param q: Wavevector
        """
        Q = np.exp(1.j * q)
        return self.k[0] + self.k[1] * Q

    def H(self, qx, qy) -> np.ndarray:
        """
        Hamiltonian

        :return: Hamiltonian defined given k and qx, qy
        """
        q = np.array([-qx / 2 + qy * np.sqrt(3) / 2,
                      -qx / 2 - qy * np.sqrt(3) / 2,
                      qx])

        z0 = self.k[0] + self.k[1]
        z = self.z(q)

        r = np.array([
            [[-1/2], [np.sqrt(3) / 2]],
            [[-1/2], [-np.sqrt(3) / 2]],
            [[1.], [0.]]
        ])

        r11 = np.outer(r[0], r[0])
        r22 = np.outer(r[1], r[1])
        r33 = np.outer(r[2], r[2])

        H = np.vstack([
            np.hstack([z0 * (r22 + r33), -z[2].conj() * r33, -z[1] * r22]),
            np.hstack([-z[2] * r33, z0 * (r33 + r11), -z[0].conj() * r11]),
            np.hstack([-z[1].conj() * r22, -z[0] * r11, z0 * (r11 + r22)])
        ])

        return H

    def dispersion(self) -> List[Tuple[float, float]]:
        """
        Calculate the dispersion relation

        :return: List of angular frequency omega for each q (wavenumber)
        """
        M_inv = la.inv(self.M)
        ws = np.empty((len(self.qys), len(self.qxs), 6))

        for y, qy in enumerate(self.qys):
            for x, qx in enumerate(self.qxs):
                eigen_val, _ = self._sort_eigen(
                    M_inv.dot(self.H(qx, qy)))
                ws[y, x] = np.sqrt(np.array(eigen_val.real))
        return ws

    def _sort_eigen(self, mat: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Return sorted eigenvalue, eigenvector pair.

        :return: eigenvalue, eigenvector
        """
        eigenvals, eigenvecs = la.eig(mat)
        sorted_idx = np.argsort(eigenvals)
        return eigenvals[sorted_idx], eigenvecs[sorted_idx]
