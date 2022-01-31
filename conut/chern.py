import numpy as np
from enum import Enum
from shapely import Polygon

BZType = Enum('BZType', ['Graphene', 'SSH'])


def calc_cherns(evecs, bz: BZType):
    """Calculate 1st Chern number given eigen-vectors. Using Japanese algorithm
    (2005).
    :param evecs: eigen-vectors of all band (assuming gapless).
    :return: list of Chern number of all bands. Should summed to 0.
    """
    if BZ == BZ.Graphene:
        x = np.array([[1.], [0.]])  # x hat
        y = np.array([[0.], [1.]])  # y hat
        a1_ = 2 * np.pi / np.sqrt(3) / a * (x - y / np.sqrt(3))
        a2_ = 4 * np.pi / 3 / a * y
        K = (a1_ / 2 + a2_ / 4) / np.cos(np.pi / 6)**2

        b1 = 2 * np.pi / np.sqrt(3) / a * (x - y / np.sqrt(3))
        b2 = 4 * np.pi / 3 / a * y

        bz = Polygon([K, -K + b1 + b2, K - b1, -K, K - b1 - b2, -K + b1])

        evecs_all_bz = np.zeros(
            (len(kys), len(kxs), 4, 4), dtype=np.complex128)

        for y, ky in enumerate(kys):
            for x, kx in enumerate(kxs):
                p = Point(kx, ky)
                if not bz.contains(p):
                    continue
                evecs_all_bz[y, x] = evecs_all[y, x]

    evecs = np.array(evecs)
    dimy, dimx, nbands, _ = evecs.shape
    cnlist = np.zeros(nbands)

    for y in range(dimy-1):
        for x in range(dimx-1):
            u12 = evecs[y, x + 1].conjugate().T @ evecs[y, x]
            u23 = evecs[y + 1, x + 1].conjugate().T @ evecs[y, x + 1]
            u34 = evecs[y + 1, x].conjugate().T @ evecs[y + 1, x + 1]
            u41 = evecs[y, x].conjugate().T @ evecs[y + 1, x]
            t12 = np.diag(u12.diagonal())
            t23 = np.diag(u23.diagonal())
            t34 = np.diag(u34.diagonal())
            t41 = np.diag(u41.diagonal())
            tplaquet = t41 @ t34 @ t23 @ t12
            cnlist += np.angle(tplaquet.diagonal())
    cnlist /= 2 * np.pi
    return cnlist
