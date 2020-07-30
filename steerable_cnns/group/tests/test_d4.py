import pytest

from steerable_cnns.group import d4, validate


def test_d4():
    e = d4.d4()
    r = d4.d4(1)
    m = d4.d4(4)
    mr = d4.d4(5)
    mr3 = d4.d4(7)

    assert m + e == m
    assert m + r == mr
    assert r + m == mr3
    assert mr - r == m

    with pytest.raises(ValueError):
        d4.d4(-1)

    with pytest.raises(ValueError):
        d4.d4(8)


def test_d4_group():
    validate.validate_group(d4.d4)


def test_a1():
    validate.validate_matrix_repr(d4.a1)


def test_a2():
    validate.validate_matrix_repr(d4.a2)


def test_b1():
    validate.validate_matrix_repr(d4.b1)


def test_b2():
    validate.validate_matrix_repr(d4.b2)


def test_e():
    validate.validate_matrix_repr(d4.e)


def test_regular():
    validate.validate_matrix_repr(d4.regular)
    assert d4.regular(0).shape == (8, 8)


def test_qm():
    validate.validate_matrix_repr(d4.qm)
    assert d4.qm(0).shape == (4, 4)


def test_qmr2():
    validate.validate_matrix_repr(d4.qmr2)
    assert d4.qmr2(0).shape == (4, 4)


def test_qmr3():
    validate.validate_matrix_repr(d4.qmr3)
    assert d4.qmr3(0).shape == (4, 4)


def test_d4_spatial_action():
    validate.validate_spatial_action(d4.d4, 7, 11)


def test_induced_repr_a1():
    validate.validate_induced_repr(d4.a1, 7)


def test_induced_repr_a2():
    validate.validate_induced_repr(d4.a2, 7)


def test_induced_repr_b1():
    validate.validate_induced_repr(d4.b1, 7)


def test_induced_repr_b2():
    validate.validate_induced_repr(d4.b2, 7)


def test_induced_repr_e():
    validate.validate_induced_repr(d4.e, 7)
