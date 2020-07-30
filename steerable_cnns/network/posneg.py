import numpy as np
import chainer.functions as F

from steerable_cnns import group


def posneg(x):
    n, c, h, w = x.shape
    y = F.concat((x, -x), axis=2)
    return y.reshape((n, c * 2, h, w))


def posneg_representation(capsule):
    matrices = np.empty((capsule.grp.order, capsule.dim * 2, capsule.dim * 2),
                        dtype=capsule.matrices.dtype)
    for mtx, g in zip(matrices, group.elems(capsule.grp)):
        caps_g = capsule(g)
        mtx[0::2, 0::2] = caps_g
        mtx[0::2, 1::2] = -caps_g
        mtx[1::2, 0::2] = -caps_g
        mtx[1::2, 1::2] = caps_g
        np.maximum(mtx, 0, out=mtx)

    return group.MatrixRepresentation(capsule.grp, matrices)
