import numpy as np
from numpy import linalg as la
from typing import List, Tuple


class MechanicalLattice1D:
    def __init__(self, k: List[float], m: List[float], precision: float = .01) -> None:
        """
        Represents dynamic system of 1 dimensional mechanical lattice.

        :param k: Spring constants
        :param m: Mass
        :param precision: Precision for wavenumber q
        """
        if len(k) != len(m):
            raise ValueError(
                f"The length of k={len(k)} and m={len(m)} does not match.")
        self.k = k
        self.M = np.diag(m)
        self.qs = np.arange(-np.pi, np.pi, precision)

    def H(self, q):
        """
        Hamiltonian 

        :return: Hamiltonian defined given k and q
        """
        k = self.k
        Q = np.exp(1.j * q)
        return np.array([[k[0] + k[1], -k[0] - k[1] * Q.conj()],
                         [-k[0] - k[1] * Q, k[0] + k[1]]])

    def dispersion(self) -> List[Tuple[float, float]]:
        """
        Calculate the dispersion relation

        :return: List of angular frequency omega for each q (wavenumber) and its eigenvectors
        """
        M_inv = la.inv(self.M)
        eigenvals = []
        eigenvecs = []
        for q in self.qs:
            eigen_val, eigen_vec = self._min_eigen(M_inv.dot(self.H(q)))
            eigenvals.append(eigen_val)
            eigenvecs.append(eigen_vec)
        ws = np.sqrt(np.array(eigenvals).real)
        evs = np.array(eigenvecs)
        return ws, evs

    def _min_eigen(self, mat: np.ndarray) -> Tuple[float, float]:
        """
        Return eigenvalue, eigenvector pair of minimum eigenvalue.

        :return: eigenvalue, eigenvector
        """
        eigenvals, eigenvecs = la.eig(mat)
        min_idx = np.argsort(eigenvals)
        return eigenvals[min_idx], eigenvecs[min_idx]

    def beta(self) -> float:
        """
        Calculate varying contrast beta with given spring constants

        :return: Varying contrast beta
        """
        k = self.k
        return (k[0] - k[1]) / (k[0] + k[1])
