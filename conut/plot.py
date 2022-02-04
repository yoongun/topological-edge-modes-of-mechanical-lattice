import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import RegularPolygon
from conut import HamiltonianType
import numpy as np
from typing import List
from conut import MechanicalGraphene


class Plot:
    def __init__(self, system, wave_number, lattice) -> None:
        self.sys = system
        self.evals = system.evals.detach().numpy().real
        self.evecs = system.evecs.detach().numpy()
        self.wn = wave_number
        self.lat = lattice

    def dispersion3d(self, bands: List[int], save=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(self.wn.kxs, self.wn.kys)

        vmax = np.amax(self.evals[:, :, max(bands)])
        p = None
        for b in bands:
            p = ax.plot_surface(
                X, Y, self.evals[:, :, b], vmin=0., vmax=vmax, cmap='gist_rainbow_r')

        ax.set_xlabel(r"$k_x$ [/m]", fontsize=14)
        ax.set_xlim(-self.lat.xw, self.lat.xw)
        ax.set_ylabel(r"$k_y$ [/m]", fontsize=14)
        ax.set_ylim(-self.lat.yw, self.lat.yw)
        ax.set_zlabel(r"$\omega$ [/s]", fontsize=14)
        ax.set_zlim(0)

        fig.colorbar(p, pad=0.15)
        if save:
            fig.savefig(
                f"κ{self.sys.κ}-α{self.sys.α}-m{self.sys.m}-l{self.sys.l}-Ω{self.sys.Ω}-p{self.sys.precision:.2f}-dispersion3d.png")
        plt.show()

    def dispersion(self, bands: List[int], save=False):
        if isinstance(self.sys, MechanicalGraphene):
            if self.sys.h_type == HamiltonianType.Bulk:
                self._dispersion_mgbulk(bands, save)
            elif self.sys.mode == HamiltonianType.Ribbon:
                self._dispersion_mgribbon(bands, save)
            else:
                raise NotImplementedError()

    def _dispersion_mgbulk(self, bands: List[int], save=False):
        fig = plt.figure()
        precision = self.wn.precision
        dr = np.arange(0, precision * len(self.wn.kxs), precision)
        for i in bands:
            plt.plot(dr, self.evals[:, i], color='blue')

        xticks = [
            0,
            self.wn.num_gk * precision,
            self.wn.num_gk * precision + self.wn.num_km * precision,
            self.wn.num_gk * precision + self.wn.num_km *
            precision + (self.wn.num_mg - 1) * precision
        ]
        xlabels = [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"]
        plt.xticks(xticks, xlabels, fontsize=14)
        plt.grid(axis='x', linestyle='dotted')
        plt.xlim(0, max(dr))
        plt.ylabel(r"$\omega$ [/s]", fontsize=14)
        plt.ylim(0)
        if save:
            fig.savefig(
                f"κ{self.sys.κ}-α{self.sys.α}-m{self.sys.m}-l{self.sys.l}-Ω{self.sys.Ω}-p{self.sys.precision:.2f}-dispersion-bulk.png")
        plt.show()

    def _dispersion_mgribbon(self, save=False):
        ...

    def band(self, n: int, save=False):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self.wn.kxs, self.wn.kys)

        vmax = np.amax(self.evals[:, :, n])
        cs = plt.contour(
            X, Y, self.evals[:, :, n], vmin=0., vmax=vmax, cmap='gist_rainbow_r')

        plt.xlabel(r"$k_x$ [/m]",  fontsize=14)
        plt.xlim(-self.lat.xw, self.lat.xw)
        plt.ylabel(r"$k_y$ [/m]", fontsize=14)
        plt.ylim(-self.lat.yw, self.lat.yw)

        norm = colors.Normalize(vmin=0., vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        fig.colorbar(sm)
        if save:
            fig.savefig(
                f"κ{self.sys.κ}-α{self.sys.α}-m{self.sys.m}-l{self.sys.l}-Ω{self.sys.Ω}-p{self.sys.precision:.2f}-band{n}.png")
        plt.show()

    def realspace(self, k: np.ndarray, band: int, save=False):
        if isinstance(self.sys, MechanicalGraphene):
            if self.sys.h_type == HamiltonianType.Bulk:
                self._realspace_mgbulk(k, band, save)
            elif self.sys.h_type == HamiltonianType.Ribbon:
                self._realspace_mgribbon(k, band, save)
            else:
                raise NotImplementedError()

    def _realspace_mgbulk(self, k: np.ndarray, band: int, save=False):
        idx = min(range(len(self.wn.kxs)),
                  key=lambda i: abs(self.wn.kxs[i]-k[0]))
        idy = min(range(len(self.wn.kys)),
                  key=lambda i: abs(self.wn.kys[i]-k[1]))
        vec = self.evecs[idy, idx, band]

        xi = vec[:2]
        eta = vec[2:4]

        a1 = np.squeeze(self.lat.a1.numpy().T)
        a2 = np.squeeze(self.lat.a2.numpy().T)
        o = np.array([0, -.5])
        offCoord = [
            o, o + a1, o + 2 * a1,
            o - a2, o + a1 - a2, o + 2 * a1 - a2, o + 3 * a1 - a2,
            o + a1 - 2 * a2, o + 2 * a1 - 2 * a2, o + 3 * a1 - 2 * a2
        ]

        K1 = np.exp(1.j * k.T.dot(a1))
        K2 = np.exp(1.j * k.T.dot(a2))

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        for c in offCoord:
            hexagon = RegularPolygon(
                (c[0] + np.sqrt(3) / 2, c[1]), numVertices=6, radius=self.lat.l, ec='silver', fill=False, lw=1.5)
            ax.add_patch(hexagon)
        plt.autoscale(enable=True)

        def arrow(pos, vec):
            # head_width=0.05
            # lw= not set
            ax.arrow(pos[0], pos[1], vec[0], vec[1], head_width=0.2,
                     head_length=0.1, fc='black', ec='red', lw=3.)

        # Xi
        for i in range(4):
            xi_ = (xi * K1**i).real
            pos = i * a1
            arrow(pos, xi_)
        for i in range(5):
            xi_ = (xi * K2.conj() * K1**i).real
            pos = i * a1 - a2
            arrow(pos, xi_)
        for i in range(4):
            xi_ = (xi * K2.conj()**2 * K1**(i + 1)).real
            pos = (i + 1) * a1 - 2 * a2
            arrow(pos, xi_)
        for i in range(3):
            xi_ = (xi * K2.conj()**3 * K1**(i + 2)).real
            pos = (i + 2) * a1 - 3 * a2
            arrow(pos, xi_)
        # Eta
        for i in range(3):
            eta_ = (eta * K1**i).real
            pos = np.squeeze(self.lat.r1) + i * a1
            arrow(pos, eta_)
        for i in range(4):
            eta_ = (eta * K2.conj() * K1**i).real
            pos = np.squeeze(self.lat.r1) + i * a1 - a2
            arrow(pos, eta_)
        for i in range(5):
            eta_ = (eta * K2.conj()**2 * K1**(i + 1)).real
            pos = np.squeeze(self.lat.r1) + i * a1 - 2 * a2
            arrow(pos, eta_)
        for i in range(4):
            eta_ = (eta * K2.conj()**3 * K1**(i + 2)).real
            pos = np.squeeze(self.lat.r1) + (i + 1) * a1 - 3 * a2
            arrow(pos, eta_)
        plt.axis('off')
        if save:
            fig.savefig(
                f"κ{self.sys.κ}-α{self.sys.α}-m{self.sys.m}-l{self.sys.l}-Ω{self.sys.Ω}-p{self.sys.precision:.2f}-realspace{band}.png")
        plt.show()

    def _realspace_mgribbon(self):
        ...
