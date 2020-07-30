from typing import Type

import numpy as np  # type: ignore

from steerable_cnns import group


def validate_group(grp: Type[group.FiniteGroupT]) -> None:
    order = grp.order
    e = grp(0)

    for i in range(order):
        gi = grp(i)
        for j in range(order):
            gj = grp(j)
            for k in range(order):
                gk = grp(k)
                if (gi + gj) + gk != gi + (gj + gk):
                    raise ValueError('Associativity failed.')

    for i in range(order):
        g = grp(i)
        if e + g != g:
            raise ValueError('Left identity failed.')
        if g + e != g:
            raise ValueError('Right identity failed.')

    for i in range(order):
        g = grp(i)
        if -g + g != e:
            raise ValueError('Left inverse failed')
        if g + -g != e:
            raise ValueError('Right inverse failed')


def validate_matrix_repr(rho: group.MatrixRepresentation[group.FiniteGroupT]
                         ) -> None:
    for gi in group.elems(rho.grp):
        for gj in group.elems(rho.grp):
            if np.any(rho(gi) @ rho(gj) != rho(gi + gj)):
                raise ValueError('Representation invalid.')


def validate_spatial_action(grp: Type[group.FiniteGroupT],
                            h: int, w: int) -> None:
    feature = np.arange(h * w).reshape((1, h, w))

    for gi in group.elems(grp):
        for gj in group.elems(grp):
            if np.any(group.act_spatially(gi, group.act_spatially(gj, feature)) !=  # NOQA
                      group.act_spatially(gi + gj, feature)):
                raise ValueError('Spatial action invalid.')


def validate_induced_repr(rho: group.MatrixRepresentation[group.FiniteGroupT],
                          ksize: int):
    pi = group.induced_representation(rho, ksize)
    dim = rho.dim
    feature = np.arange(dim * ksize * ksize).reshape((dim, ksize, ksize))

    for g in group.elems(rho.grp):
        rho_feature = ((rho(g) @ feature.reshape((dim, ksize * ksize))).
                       reshape((dim, ksize, ksize)))
        if np.any(pi(g) @ feature.ravel() !=
                  group.act_spatially(g, rho_feature).ravel()):
            raise ValueError('Induced pi invalid.')
