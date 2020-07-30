import functools
from abc import ABCMeta, abstractmethod
from typing import ClassVar, Generic, Iterator, List, Sequence, Type, TypeVar

import numpy as np  # type: ignore
import scipy as sp  # type: ignore


FiniteGroupT = TypeVar('FiniteGroupT', bound='FiniteGroup')


class FiniteGroup(metaclass=ABCMeta):
    """Abstract class of finite group.

    Args:
        value (int): The number of an element.

    Class attribute:
        order (int): The order of a group.
    """
    __slots__ = ()

    order: ClassVar[int]

    @abstractmethod
    def __init__(self, value: int = 0):
        raise NotImplementedError

    @abstractmethod
    def __int__(self: FiniteGroupT) -> int:
        raise NotImplementedError

    @abstractmethod
    def __add__(self: FiniteGroupT, other: FiniteGroupT) -> FiniteGroupT:
        raise NotImplementedError

    @abstractmethod
    def __neg__(self: FiniteGroupT) -> FiniteGroupT:
        raise NotImplementedError

    def __sub__(self: FiniteGroupT, other: FiniteGroupT) -> FiniteGroupT:
        return self + -other

    def __eq__(self, other):
        return int(self) == int(other)

    def __hash__(self):
        return hash(int(self))


def elems(grp: Type[FiniteGroupT]) -> Iterator[FiniteGroupT]:
    """Enumerate elements of a finite group."""
    for i in range(grp.order):
        yield grp(i)


class MatrixRepresentation(Generic[FiniteGroupT]):
    """Group representation as linear operators.

    A representation `rho` is a set of isomorphic linear operators indexed by
    finite group elements. It must preserve group structure.
    That is, for each group element `g` and `h`, `rho` must satisfy
    `rho(g) . rho(h) = rho(g . h)` and `rho(g)^-1 = rho(g^-1)`.

    Args:
        grp (subclass of FiniteGroup): Finite group.
        matrices (numpy.ndarray): Matrices that define linear operators.
            Its shape must be `(grp.order, dim, dim)`, where `dim` is
            the dimension of the representation.
    """
    def __init__(self, grp: Type[FiniteGroupT], matrices: np.ndarray):
        self.grp: Type[FiniteGroupT] = grp
        if matrices.ndim != 3:
            raise ValueError
        if matrices.shape[0] != grp.order:
            raise ValueError
        if matrices.shape[1] != matrices.shape[2]:
            raise ValueError
        self.matrices: np.ndarray = matrices

    @property
    def dim(self) -> int:
        return int(self.matrices.shape[1])

    def __eq__(self, other):
        return self.matrices.tobytes() == other.matrices.tobytes()

    def __hash__(self):
        return hash(self.matrices.tobytes())

    def __call__(self, h: FiniteGroupT) -> np.ndarray:
        return self.matrices[int(h)]


def directsum_representation(rhos: Sequence[MatrixRepresentation[FiniteGroupT]]
                             ) -> MatrixRepresentation[FiniteGroupT]:
    """Return the direct sum of given reprensentations.

    For each group element `g`, `directsum(rho1, ..., rhok)(g)` is the block
    diagonal matrix constructed from `rho1(g), ..., rhok(g)`.
    """
    if len(rhos) == 0:
        raise ValueError('rhos must be non-empty')
    dim = sum(rho.dim for rho in rhos)
    matrices = np.zeros((rhos[0].grp.order, dim, dim), rhos[0].matrices.dtype)

    k = 0
    for rho in rhos:
        new_k = k + rho.dim
        for g, mtx in zip(elems(rho.grp), matrices):
            mtx[k:new_k, k:new_k] = rho(g)
        k = new_k

    return MatrixRepresentation[FiniteGroupT](rho.grp, matrices)


class Cosets(Generic[FiniteGroupT]):
    """Set of cosets in a finite group."""
    grp: ClassVar[Type[FiniteGroupT]]
    order: ClassVar[int]
    _mapping: ClassVar[List[int]]

    __slot__ = ('g',)

    def __init__(self, g: FiniteGroupT):
        self.g: FiniteGroupT = g

    @classmethod
    def from_int(cls, value: int = 0) -> 'Cosets[FiniteGroupT]':
        return cls(cls.grp(cls._mapping.index(value)))

    def __int__(self) -> int:
        return self._mapping[int(self.g)]

    def acted(self, g: FiniteGroupT) -> 'Cosets[FiniteGroupT]':
        return self.__class__(g + self.g)


def cosets(grp: Type[FiniteGroupT],
           subgroup: Sequence[FiniteGroupT]) -> Type[Cosets[FiniteGroupT]]:
    """Return the set of cosets of the given subgroup."""
    mapping = [-1] * grp.order
    counter = 0

    for i in range(grp.order):
        if mapping[i] != -1:
            continue
        g = grp(i)
        for h in subgroup:
            mapping[int(g + h)] = counter
        counter += 1

    class _Cosets(Cosets):
        pass

    _Cosets.grp = grp  # type: ignore
    _Cosets.order = counter
    _Cosets._mapping = mapping

    return _Cosets


def cosets_representation(grp: Type[FiniteGroupT],
                          subgroup: Sequence[FiniteGroupT]
                          ) -> MatrixRepresentation[FiniteGroupT]:
    """Define representation by permutation of cosets by each group element."""
    order = grp.order
    cst = cosets(grp, subgroup)
    qorder = cst.order

    matrices = []
    for i in range(order):
        mtx = np.zeros((qorder, qorder), dtype=np.int32)
        g = grp(i)
        for j in range(qorder):
            mtx[int(cst.from_int(j).acted(g)), j] = 1
        matrices.append(mtx)

    matrices = np.stack(matrices)
    return MatrixRepresentation(grp, matrices)


@functools.singledispatch
def act_spatially(g: FiniteGroupT, maps: np.ndarray) -> np.ndarray:
    """Apply spatial action that represent a group element to each map.

    Args:
        g (FiniteGroup): Group element.
        maps (numpy.ndarray): Tensor that has at least two dimensions.
            If it has only two dimensions, it is a map of spatially distributed
            values. Otherwise, it is regarded as a stack of maps. The action is
            individually applied to each map.
    """
    raise NotImplementedError


def induced_representation(rho: MatrixRepresentation[FiniteGroupT],
                           ksize: int) -> MatrixRepresentation[FiniteGroupT]:
    """Induce a steerable representation that acts to each image column.

    The image columns are obtained by im2col transformation, which copies
    every ksize x ksize patch on an image. The resultant representation has
    the dimension `dim(rho) * ksize * ksize`.
    """
    matrices = []
    for g in elems(rho.grp):
        permute_t = np.identity(ksize * ksize, dtype=int)
        permute_t = (act_spatially(g, permute_t.reshape(-1, ksize, ksize)).
                     reshape(permute_t.shape))
        permute = permute_t.T

        dim = rho.dim * ksize * ksize
        mtx = rho(g)[:, :, None, None] * permute
        mtx = mtx.transpose((0, 2, 1, 3)).reshape((dim, dim))
        matrices.append(mtx)

    matrices = np.stack(matrices)
    return MatrixRepresentation(rho.grp, matrices)


@functools.lru_cache()
def intertwiner_basis(pi: MatrixRepresentation[FiniteGroupT],
                      rho: MatrixRepresentation[FiniteGroupT]) -> np.ndarray:
    """Return a basis of intertwiner between two given representations.

    The intertwiner is a matrix `M` such that `M @ pi(g) = rho(g) @ M` for
    any group element `g`, where `@` denotes matrix multiplication.

    The shape of the returned matrix is `(dim(rho) * dim(pi), dim(operator))`.

    .. note::
        This function compute a kernel space using SciPy's rank-revealing QR
        decomposition routine. It is faster than SVD. We once tried SymPy but
        it was even much slower than SVD.
    """
    eqmatrix = np.zeros((pi.grp.order, rho.dim, pi.dim, rho.dim, pi.dim),
                        dtype=pi.matrices.dtype)
    for g in elems(pi.grp):
        n = int(g)
        pi_g = pi(g)
        rho_g = rho(g)
        for i in range(rho.dim):
            for j in range(pi.dim):
                # Encode the constraint
                # `(M @ pi(g))[i, j] - (rho(g) @ M)[i, j] = 0`
                # into `eqmatrix[n, i, j]`.
                eqmatrix[n, i, j, i, :] += pi_g[:, j]
                eqmatrix[n, i, j, :, j] -= rho_g[i, :]
    eqmatrix = eqmatrix.reshape((-1, rho.dim * pi.dim))

    q, r, p = sp.linalg.qr(eqmatrix.T, overwrite_a=True, pivoting=True)
    rank = np.sum(abs(np.diag(r)) > 1e-5)
    kernel = q[:, rank:]
    return kernel.copy()
