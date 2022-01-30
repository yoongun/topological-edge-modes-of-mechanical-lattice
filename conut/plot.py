import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import RegularPolygon
import numpy as np
from typing import List
from conut import MechanicalGraphene, Mode


class Plot:
    def __init__(self, system) -> None:
        self.sys = system
        self.evals, self.evecs = system.dispersion()
        self.evals_gkmg, self.evecs_gkmg = system.dispersion(True)

    def dispersion3d(self, bands: List[int], save=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(self.sys.kxs, self.sys.kys)

        vmax = np.amax(self.evals[:, :, max(bands)])
        print(vmax)
        p = None
        for b in bands:
            p = ax.plot_surface(
                X, Y, self.evals[:, :, b], vmin=0., vmax=vmax, cmap='gist_rainbow_r')

        ax.set_xlabel(r"$k_x$ [/m]", fontsize=14)
        ax.set_xlim(-self.sys.lat.xw, self.sys.lat.xw)
        ax.set_ylabel(r"$k_y$ [/m]", fontsize=14)
        ax.set_ylim(-self.sys.lat.yw, self.sys.lat.yw)
        ax.set_zlabel(r"$\omega$ [/s]", fontsize=14)
        ax.set_zlim(0)

        fig.colorbar(p, pad=0.15)
        if save:
            fig.savefig(
                f"κ{self.sys.κ}-α{self.sys.α}-m{self.sys.m}-l{self.sys.l}-Ω{self.sys.Ω}-p{self.sys.precision:.2f}-dispersion3d.png")
        plt.show()

    def dispersion(self, bands: List[int], save=False):
        if isinstance(self.sys, MechanicalGraphene):
            if self.sys.mode == Mode.Bulk:
                self._dispersion_mgbulk(bands, save)
            elif self.sys.mode == Mode.Ribbon:
                self._dispersion_mgribbon(bands, save)
            else:
                raise NotImplementedError()

    def _dispersion_mgbulk(self, bands: List[int], save=False):
        fig = plt.figure()
        precision = self.sys.precision
        dr = np.arange(0, precision * len(self.sys.kxs_gkmg), precision)
        for i in bands:
            plt.plot(dr, self.evals_gkmg[:, i], color='blue')

        xticks = [
            0,
            self.sys.num_gk * precision,
            self.sys.num_gk * precision + self.sys.num_km * precision,
            self.sys.num_gk * precision + self.sys.num_km *
            precision + (self.sys.num_mg - 1) * precision
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
        X, Y = np.meshgrid(self.sys.kxs, self.sys.kys)

        vmax = np.amax(self.evals[:, :, n])
        cs = plt.contour(
            X, Y, self.evals[:, :, n], vmin=0., vmax=vmax, cmap='gist_rainbow_r')

        plt.xlabel(r"$k_x$ [/m]",  fontsize=14)
        plt.xlim(-self.sys.lat.xw, self.sys.lat.xw)
        plt.ylabel(r"$k_y$ [/m]", fontsize=14)
        plt.ylim(-self.sys.lat.yw, self.sys.lat.yw)

        norm = colors.Normalize(vmin=0., vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        fig.colorbar(sm)
        if save:
            fig.savefig(
                f"κ{self.sys.κ}-α{self.sys.α}-m{self.sys.m}-l{self.sys.l}-Ω{self.sys.Ω}-p{self.sys.precision:.2f}-band{n}.png")
        plt.show()

    def realspace(self, save=False):
        if isinstance(self.sys, MechanicalGraphene):
            if self.sys.mode == Mode.Bulk:
                self._realspace_mgbulk(save)
            elif self.sys.mode == Mode.Ribbon:
                self._realspace_mgribbon(save)
            else:
                raise NotImplementedError()

    def _realspace_mgbulk(self, k: np.ndarray, band: int, save=False):
        idx = min(range(len(self.sys.kxs)),
                  key=lambda i: abs(self.sys.kxs[i]-k[0]))
        # idy = min(range(len(self.sys.kys)),
        #           key=lambda i: abs(self.sys.kys[i]-k[1]))
        # vec = self.evecs_all[idy, idx, band]
        # xi = vec[:2]
        # eta = vec[2:4]

        # a1 = self.sys.lat.a1
        # a2 = self.sys.lat.a2
        # dx = np.sqrt(3)
        # offCoord = [
        #     [0, -.5], [dx, -.5], [2*dx, -.5],
        #     [0, -.5] - a2, [dx, -.5] - a2, [2*dx, -.5] - a2, [3*dx, -.5] - a2,
        #     [0, -.5] - 2*a2 + a1, [dx, -.5] - 2 *
        #     a2 + a1, [2*dx, -.5] - 2*a2 + a1
        # ]

        # K1 = np.exp(1.j * k.dot(a1))
        # K2 = np.exp(1.j * k.dot(a2))

        # fig, ax = plt.subplots(1)
        # ax.set_aspect('equal')

        # for c in offCoord:
        #     hexagon = RegularPolygon(
        #         (c[0] + np.sqrt(3) / 2, c[1]), numVertices=6, radius=self.sys.l, ec='silver', fill=False, lw=1.5)
        #     ax.add_patch(hexagon)
        # plt.autoscale(enable=True)

        # def arrow(pos, vec):
        #     ax.arrow(pos[0], pos[1], vec[0], vec[1], head_width=0.05,
        #              head_length=0.1, fc='black', ec='red')

        # # Xi
        # for i in range(4):
        #     xi_ = (xi * K1**i).real
        #     pos = i * a1
        #     arrow(pos, xi_)
        # for i in range(5):
        #     xi_ = (xi * K2.conj() * K1**i).real
        #     pos = i * a1 - a2
        #     arrow(pos, xi_)
        # for i in range(4):
        #     xi_ = (xi * K2.conj()**2 * K1**(i + 1)).real
        #     pos = (i + 1) * a1 - 2 * a2
        #     arrow(pos, xi_)
        # for i in range(3):
        #     xi_ = (xi * K2.conj()**3 * K1**(i + 2)).real
        #     pos = (i + 2) * a1 - 3 * a2
        #     arrow(pos, xi_)
        # # Eta
        # for i in range(3):
        #     eta_ = (eta * p**i).real
        #     pos = np.squeeze(R1) + i * a1
        #     arrow(pos, eta_)
        # for i in range(4):
        #     eta_ = (xi * s.conj() * p**i).real
        #     pos = np.squeeze(R1) + i * a1 - a2
        #     arrow(pos, eta_)
        # for i in range(5):
        #     eta_ = (xi * s.conj()**2 * p**(i + 1)).real
        #     pos = np.squeeze(R1) + i * a1 - 2 * a2
        #     arrow(pos, eta_)
        # for i in range(4):
        #     eta_ = (xi * s.conj()**3 * p**(i + 2)).real
        #     pos = np.squeeze(R1) + (i + 1) * a1 - 3 * a2
        #     arrow(pos, eta_)
        # plt.axis('off')
        # plt.show()

    def _realspace_mgribbon(self):
        ...
