import numpy as np
from numpy import linalg as la
from typing import List, Tuple
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


class StringABLattice:
    """
    ABABAB...
    """

    def __init__(self, k: List[float], m: List[float], precision: float = .01) -> None:
        """
        Represents dynamic system of 1 dimensional mechanical lattice.

        :param k: Spring constants (2)
        :param m: Mass (2)
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

    def animate(self, q: float, N: int, mode: int, *, fps: int = 30, s: int = 3):
        """
        :param q: Wavenumber to animate [-pi, pi]
        :param N: Number of unit cells
        :param mode: Mode to animate (0 for acoustic, 1 for optical)
        :param fps: (Optional) Frame per second (/s) (default: 30 /s)
        :param s: (Optional) Animation duration (s) (default: 3 s)
        """
        ws, evs = self.dispersion()

        # Parameters
        idx = min(range(len(self.qs)), key=lambda i: abs(self.qs[i] - q))
        w = ws[idx, mode]  # /s

        # Construct frames
        frames = []
        for t in range(int(s * fps)):
            dt = t / fps
            dphase = dt * w * 2 * np.pi
            y = []
            for i in range(N):
                y.append(evs[idx, mode, 0] * np.exp(1.j * (q * i + dphase)))
                y.append(evs[idx, mode, 1] * np.exp(1.j * (q * i + dphase)))
            y = np.array(y)
            frames.append(
                go.Frame(data=[go.Scatter(y=y.real, line_shape='spline')]))

        # Figure components
        start_button = dict(
            label="Play",
            method="animate",
            args=[
                None,
                {
                    "frame": {"duration": 1000 / fps, "redraw": False},
                    "fromcurrent": True,
                    "transition": {"duration": 100}
                }])
        pause_button = dict(
            label="Pause",
            method="animate",
            args=[
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }])

        # Plot
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Dispersion relation animation",
                yaxis=dict(range=[-1., 1.], autorange=False),
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[start_button, pause_button
                                 ])
                ]
            ),
            frames=frames[1:])
        fig.show()

    def plot_dispersion_relation(self):
        ws, _ = self.dispersion()
        w0 = ws[:, 0]
        w1 = ws[:, 1]
        ws = np.append(w0, w1)

        x = np.append(self.qs, self.qs)
        y = ws
        index = np.append(np.repeat(0, len(self.qs)),
                          np.repeat(1, len(self.qs)))

        df = pd.DataFrame({
            "q": x,
            "w": y,
            "index": index,
        })

        fig = px.line(df, x="q", y="w", color='index')
        fig.show()
