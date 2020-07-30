import numpy as np  # type: ignore

from steerable_cnns import group


class d4(group.FiniteGroup):
    __slots__ = ('m', 'r')

    order = 8

    def __init__(self, value: int = 0):
        if not 0 <= value < self.order:
            raise ValueError
        self.m: int = value // 4
        self.r: int = value % 4

    def __int__(self: 'd4') -> int:
        return self.m * 4 + self.r

    def __add__(self: 'd4', other: 'd4') -> 'd4':
        if other.m == 0:
            m = self.m
            r = (self.r + other.r) % 4
        else:
            m = 1 - self.m
            r = (other.r - self.r) % 4
        return self.__class__(m * 4 + r)

    def __neg__(self: 'd4') -> 'd4':
        if self.m == 0:
            return d4((4 - self.r) % 4)
        else:
            return self


a1: group.MatrixRepresentation[d4] = \
    group.MatrixRepresentation(d4, np.array(
        [[[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]]],
        dtype=np.int32))


a2: group.MatrixRepresentation[d4] = \
    group.MatrixRepresentation(d4, np.array(
        [[[1]], [[1]], [[1]], [[1]], [[-1]], [[-1]], [[-1]], [[-1]]],
        dtype=np.int32))


b1: group.MatrixRepresentation[d4] = \
    group.MatrixRepresentation(d4, np.array(
        [[[1]], [[-1]], [[1]], [[-1]], [[1]], [[-1]], [[1]], [[-1]]],
        dtype=np.int32))


b2: group.MatrixRepresentation[d4] = \
    group.MatrixRepresentation(d4, np.array(
        [[[1]], [[-1]], [[1]], [[-1]], [[-1]], [[1]], [[-1]], [[1]]],
        dtype=np.int32))


e: group.MatrixRepresentation[d4] = \
    group.MatrixRepresentation(d4, np.array([
        [[1, 0], [0, 1]],
        [[0, -1], [1, 0]],
        [[-1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
        [[-1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
        [[0, -1], [-1, 0]],
    ], dtype=np.int32))


regular: group.MatrixRepresentation[d4] = \
    group.cosets_representation(d4, [d4(0)])


qm: group.MatrixRepresentation[d4] = \
    group.cosets_representation(d4, [d4(0), d4(4)])


qmr2: group.MatrixRepresentation[d4] = \
    group.cosets_representation(d4, [d4(0), d4(6)])


qmr3: group.MatrixRepresentation[d4] = \
    group.cosets_representation(d4, [d4(0), d4(7)])


@group.act_spatially.register(d4)
def _(g: d4, maps: np.ndarray) -> np.ndarray:
    code = int(g)
    if code == 0:
        return maps
    elif code == 1:
        return np.swapaxes(maps, -1, -2)[..., ::-1]
    elif code == 2:
        return maps[..., ::-1, ::-1]
    elif code == 3:
        return np.swapaxes(maps[..., ::-1], -1, -2)
    elif code == 4:
        return maps[..., ::-1]
    elif code == 5:
        return np.swapaxes(maps, -1, -2)
    elif code == 6:
        return maps[..., ::-1, :]
    elif code == 7:
        return np.swapaxes(maps[..., ::-1], -1, -2)[..., ::-1]
    else:
        raise TypeError
