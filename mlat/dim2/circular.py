import numpy as np
from numpy import linalg as la
from typing import List, Tuple


class CircularLattice:
    def __init__(self, k: float, m: float, precision: float = .01) -> None:
        """
        Represents dynamic system of 2 dimensional circular lattice.

        :param k: A spring constant
        :param m: A mass
        :param precision: Precision for wavenumber q
        """
        self.N = 24  # number of points
        self.R = 10  # Radius (cm)
        self.k = k
        self.M = np.diag([m] * 2 * self.N)
        self.qs = np.arange(-np.pi, np.pi, precision)

        grid = []
        dtheta = 2 * np.pi / self.N
        for n in range(self.N):
            grid.append([
                self.R*np.cos(dtheta * n),
                self.R*np.sin(dtheta * n)])
        self.grid = np.array(grid)

    @property
    def r(self) -> np.ndarray:
        N = self.N
        r = np.empty((N, N, 2))
        for n1 in range(N):
            for n2 in range(N):
                r[n1, n2] = self.grid[n2] - self.grid[n1]
        return r

    @property
    def r_subspace(self) -> np.ndarray:
        N = self.N
        r = self.r
        r_subspace = np.empty((N, N, 2, 2))
        for n1 in range(N):
            for n2 in range(N):
                r_subspace[n1, n2] = np.outer(r[n1, n2], r[n1, n2])
        return r_subspace

    def H(self, q) -> np.ndarray:
        """
        Hamiltonian

        :return: Hamiltonian defined given k and Q
        """
        N = self.N
        r_subspace = self.r_subspace
        k = self.k
        H = np.empty((2 * N, 2 * N), dtype=np.complex128)
        Q = np.exp(1.j * q)

        for n, i in enumerate(range(0, 2 * N, 2)):
            H[i:i+2, i:i+2] = k * (r_subspace[(n-1) %
                                              N, n] + r_subspace[n, (n+1) % N])
            if i + 4 > N:
                continue
            H[i+2:i+4, i:i+2] = -k * Q.conj() * r_subspace[n, (n+1) % N]
            H[i:i+2, i+2:i+4] = -k * Q * r_subspace[n, n+1]
        H[0:2, (2*N-2):2*N] = -k * Q.conj() * r_subspace[0, N-1]
        H[(2*N-2):2*N, 0:2] = -k * Q * r_subspace[0, N-1]

        return H

    def dispersion(self) -> List[Tuple[float, float]]:
        """
        Calculate the dispersion relation

        :return: List of angular frequency omega for each q (wavenumber)
        """
        M_inv = la.inv(self.M)
        ws = np.empty((len(self.qs), 48))
        evecs = np.empty((len(self.qs), 48, 48), dtype=np.complex128)

        for i, q in enumerate(self.qs):
            eval_, evec = self._sort_eigen(M_inv.dot(self.H(q)))
            ws[i] = np.sqrt(np.array(eval_.real))
            evecs[i] = evec
        return ws, evecs

    def _sort_eigen(self, mat: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Return sorted eigenvalue, eigenvector pair.

        :return: eigenvalue, eigenvector
        """
        eigenvals, eigenvecs = la.eig(mat)
        sorted_idx = np.argsort(eigenvals)
        return eigenvals[sorted_idx], eigenvecs[sorted_idx]
