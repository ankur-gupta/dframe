from __future__ import print_function
from __future__ import absolute_import
from builtins import range

import pytest
from dframe import Array
from dframe import is_na, is_missing, is_none
from dframe import which, where, find
from dframe import unique


class TestArrayMissing:
    x1 = Array([1, 2, 3, 4, 5, 6, 7])
    x2 = Array([1, 2, 3, None, 5, None, 7])

    y1 = Array(['a', 'b', 'c', 'None', 'd'])
    y2 = Array(['a', 'b', 'c', None, 'd'])

    def test_no_missing_values(self):
        assert isinstance(is_na(self.x1), Array)
        assert isinstance(is_none(self.x1), Array)
        assert isinstance(is_missing(self.x1), Array)

        assert len(is_na(self.x1)) == len(self.x1)
        assert len(is_none(self.x1)) == len(self.x1)
        assert len(is_missing(self.x1)) == len(self.x1)

        assert is_na(self.x1).dtype is bool
        assert is_none(self.x1).dtype is bool
        assert is_missing(self.x1).dtype is bool

        assert not any(is_na(self.x1))
        assert not any(is_none(self.x1))
        assert not any(is_missing(self.x1))

        assert not all(is_na(self.x1))
        assert not all(is_none(self.x1))
        assert not all(is_missing(self.x1))

        assert all(map(lambda x: x is False, is_na(self.x1)))
        assert all(map(lambda x: x is False, is_none(self.x1)))
        assert all(map(lambda x: x is False, is_missing(self.x1)))

        assert any(map(lambda x: x is None, is_na(self.x1))) is False
        assert any(map(lambda x: x is None, is_none(self.x1))) is False
        assert any(map(lambda x: x is None, is_missing(self.x1))) is False

        assert isinstance(is_na(self.y1), Array)
        assert isinstance(is_none(self.y1), Array)
        assert isinstance(is_missing(self.y1), Array)

        assert len(is_na(self.y1)) == len(self.y1)
        assert len(is_none(self.y1)) == len(self.y1)
        assert len(is_missing(self.y1)) == len(self.y1)

        assert is_na(self.y1).dtype is bool
        assert is_none(self.y1).dtype is bool
        assert is_missing(self.y1).dtype is bool

        assert not any(is_na(self.y1))
        assert not any(is_none(self.y1))
        assert not any(is_missing(self.y1))

        assert not all(is_na(self.y1))
        assert not all(is_none(self.y1))
        assert not all(is_missing(self.y1))

        assert all(map(lambda x: x is False, is_na(self.y1)))
        assert all(map(lambda x: x is False, is_none(self.y1)))
        assert all(map(lambda x: x is False, is_missing(self.y1)))

        assert any(map(lambda x: x is None, is_na(self.y1))) is False
        assert any(map(lambda x: x is None, is_none(self.y1))) is False
        assert any(map(lambda x: x is None, is_missing(self.y1))) is False

    def test_missing_values(self):
        assert isinstance(is_na(self.x2), Array)
        assert isinstance(is_none(self.x2), Array)
        assert isinstance(is_missing(self.x2), Array)

        assert len(is_na(self.x2)) == len(self.x2)
        assert len(is_none(self.x2)) == len(self.x2)
        assert len(is_missing(self.x2)) == len(self.x2)

        assert is_na(self.x2).dtype is bool
        assert is_none(self.x2).dtype is bool
        assert is_missing(self.x2).dtype is bool

        assert any(is_na(self.x2))
        assert any(is_none(self.x2))
        assert any(is_missing(self.x2))

        assert not all(is_na(self.x2))
        assert not all(is_none(self.x2))
        assert not all(is_missing(self.x2))

        assert is_na(self.x2)[3] is True
        assert is_na(self.x2)[5] is True
        assert is_none(self.x2)[3] is True
        assert is_none(self.x2)[5] is True
        assert is_missing(self.x2)[3] is True
        assert is_missing(self.x2)[5] is True

        assert any(map(lambda x: x is None, is_na(self.x2))) is False
        assert any(map(lambda x: x is None, is_none(self.x2))) is False
        assert any(map(lambda x: x is None, is_missing(self.x2))) is False

        assert isinstance(is_na(self.y2), Array)
        assert isinstance(is_none(self.y2), Array)
        assert isinstance(is_missing(self.y2), Array)

        assert len(is_na(self.y2)) == len(self.y2)
        assert len(is_none(self.y2)) == len(self.y2)
        assert len(is_missing(self.y2)) == len(self.y2)

        assert is_na(self.y2).dtype is bool
        assert is_none(self.y2).dtype is bool
        assert is_missing(self.y2).dtype is bool

        assert any(is_na(self.y2))
        assert any(is_none(self.y2))
        assert any(is_missing(self.y2))

        assert not all(is_na(self.y2))
        assert not all(is_none(self.y2))
        assert not all(is_missing(self.y2))

        assert is_na(self.y2)[3] is True
        assert is_none(self.y2)[3] is True
        assert is_missing(self.y2)[3] is True

        assert any(map(lambda x: x is None, is_na(self.y2))) is False
        assert any(map(lambda x: x is None, is_none(self.y2))) is False
        assert any(map(lambda x: x is None, is_missing(self.y2))) is False


class TestUnique:
    x1 = Array([1, 2, 3, 4, 5, 6, 7, 8])
    x2 = Array([1, 1, 1, 1, 1, 1, 1, 1])
    x3 = Array([1, 1, 1, 1, 2, 2, 2, 2])

    y1 = Array(['', ' ', 'a'])
    y2 = Array(['', ' ', 'a', 'a', 'a', '', '', ' '])

    z1 = Array([1, 2, None, 4, None])

    def test_already_unique_x1(self):
        u = unique(self.x1)
        assert isinstance(u, Array)
        assert u.dtype == self.x1.dtype
        assert len(u) == len(self.x1)
        assert self.x1.equals(u)
        assert all(self.x1 == u)

    def test_not_unique_x2(self):
        u = unique(self.x2)
        assert isinstance(u, Array)
        assert u.dtype == self.x2.dtype
        assert len(u) == 1
        assert u[0] == 1

    def test_not_unique_x3(self):
        u = unique(self.x3)
        assert isinstance(u, Array)
        assert u.dtype == self.x3.dtype
        assert len(u) == 2
        assert u[0] == 1
        assert u[1] == 2

    def test_already_unique_y1(self):
        u = unique(self.y1)
        assert isinstance(u, Array)
        assert u.dtype == self.y1.dtype
        assert len(u) == len(self.y1)
        assert self.y1.equals(u)
        assert all(self.y1 == u)

    def test_not_unique_y2(self):
        u = unique(self.y2)
        assert isinstance(u, Array)
        assert u.dtype == self.y2.dtype
        assert len(u) == 3
        assert u[0] == ''
        assert u[1] == ' '
        assert u[2] == 'a'

    def test_unique_with_none(self):
        u = unique(self.z1)
        assert isinstance(u, Array)
        assert u.dtype == self.z1.dtype
        assert len(u) == 4
        assert u[0] == 1
        assert u[1] == 2
        assert u[2] == 4
        assert u[3] is None

