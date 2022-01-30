import numpy as np

def calc_chern(hVec):
    """Calculate 1st Chern number given eigen-vectors. Using Japanese algorithm
    (2005).
    :param hVec: eigen-vectors of all band (assuming gapless).
    :return: list of Chern number of all bands. Should summed to 0.
    """
    hVec = np.array(hVec)
    dimy, dimx, nlevels, _ = hVec.shape
    cnlist = np.zeros(nlevels)

    for iy in range(dimy-1):
        for ix in range(dimx-1):
            u12 = hVec[iy, ix + 1].conjugate().T @ hVec[iy, ix]
            u23 = hVec[iy + 1, ix + 1].conjugate().T @ hVec[iy, ix + 1]
            u34 = hVec[iy + 1, ix].conjugate().T @ hVec[iy + 1, ix + 1]
            u41 = hVec[iy, ix].conjugate().T @ hVec[iy + 1, ix]
            t12 = np.diag(u12.diagonal())
            t23 = np.diag(u23.diagonal())
            t34 = np.diag(u34.diagonal())
            t41 = np.diag(u41.diagonal())
            tplaquet = t41 @ t34 @ t23 @ t12
            cnlist += np.angle(tplaquet.diagonal())

    cnlist /= 2 * np.pi
    # cnlist = chop(cnlist)
    return cnlist


def chop(array, tol=1e-7):
    """Realize Mathematica Chop[].
    :param array: 1D array.
    :param tol: tolerance to be chopped. default to 1e-7
    :return: chopped array. (original array alse modified.)
    """
    for i in range(len(array)):
        a = array[i]
        if np.abs(a-round(a)) < tol:
            array[i] = round(a)
    return array


x = np.array([[1.], [0.]])  # x hat
y = np.array([[0.], [1.]])  # y hat
a1_ = 2 * np.pi / np.sqrt(3) / a * (x - y / np.sqrt(3))
a2_ = 4 * np.pi / 3 / a * y
K = (a1_ / 2 + a2_ / 4) / np.cos(np.pi / 6)**2
b1 = 2 * np.pi / np.sqrt(3) / a * (x - y / np.sqrt(3))
b2 = 4 * np.pi / 3 / a * y

h = precision
bz = Polygon([K, -K + b1 + b2, K - b1, -K, K - b1 - b2, -K + b1])

evecs_all_bz = np.zeros((len(kys), len(kxs), 4, 4), dtype=np.complex128)

for y, ky in enumerate(kys):
    for x, kx in enumerate(kxs):
        p = Point(kx, ky)
        if not bz.contains(p):
            continue
        evecs_all_bz[y, x] = evecs_all[y, x]


print(calc_chern(evecs_all_bz))
