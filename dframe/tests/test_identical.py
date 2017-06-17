from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import identical, Array, DataFrame


class TestBase:
    def test_none(self):
        assert identical(None, None) is True

        assert identical(None, 1) is False
        assert identical(None, 1.0) is False
        assert identical(None, -1) is False
        assert identical(None, -1.0) is False
        assert identical(None, 'a') is False
        assert identical(None, '') is False
        assert identical(None, 'None') is False
        assert identical(None, [None]) is False
        assert identical(None, set([None])) is False
        assert identical(None, list([None])) is False
        assert identical(None, {None}) is False
        assert identical(None, {'a': None}) is False
        assert identical(None, list()) is False
        assert identical(None, set()) is False
        assert identical(None, dict()) is False
        assert identical(None, {}) is False
        assert identical(None, []) is False
        assert identical(None, {}) is False

    def test_int(self):
        assert identical(1, 1) is True
        assert identical(1, 2 - 1) is True
        assert identical(1, 0 + 1) is True
        assert identical(1 * 2, 1 * 2) is True
        assert identical(2 / 1, 2 / 1) is True
        assert identical(10 / 5, 10 / 5) is True

        assert identical(0, 0) is True
        assert identical(0, 0 + 0) is True
        assert identical(0, 0 + 0 + 0) is True

        assert identical(-1, -1) is True
        assert identical(-1, 0 - 1) is True
        assert identical(-1, 100 - 101) is True

        assert identical(0, 0.0) is False
        assert identical(1, 1.0) is False
        assert identical(1, 1.0 - 0) is False
        assert identical(1, 1.0 - 0.0) is False
        assert identical(0, 1.0 - 1.0) is False
        assert identical(-1, 0.0 - 1) is False

        assert identical(1, '1') is False
        assert identical(1, '') is False
        assert identical(1, None) is False
        assert identical(1, 2) is False
        assert identical(1, []) is False
        assert identical(1, [1]) is False
        assert identical(1, [1.0]) is False
        assert identical(1, {1}) is False
        assert identical(1, {'a': 1}) is False
        assert identical(1, True) is False
        assert identical(1, False) is False
        assert identical(0, True) is False
        assert identical(0, False) is False

    def test_float(self):
        assert identical(1.0, 1.0) is True
        assert identical(1.0, 1.0 - 0.0) is True

        assert identical(1.141, 1.141) is True
        assert identical(1.141 * 2, 1.141 * 2) is True
        assert identical(-1.141, -1.141) is True

        assert identical(-1.141, '-1.141') is False
        assert identical(1.0, '1.0') is False
        assert identical(0.0, []) is False
        assert identical(0.0, {}) is False
        assert identical(0.0, None) is False
        assert identical(0.0, 'a') is False
        assert identical(0.0, '') is False
        assert identical(0.0, []) is False
        assert identical(0.0, True) is False
        assert identical(0.0, False) is False

    def test_bool(self):
        assert identical(True, True) is True
        assert identical(False, False) is True
        assert identical(True, False) is False
        assert identical(False, True) is False

        assert identical(True, [True]) is False
        assert identical(False, [False]) is False
        assert identical(True, 1) is False
        assert identical(False, 0) is False

        assert identical(True, None) is False
        assert identical(False, None) is False

        for var in [True, False]:
            assert identical(var, []) is False
            assert identical(var, {}) is False
            assert identical(var, set()) is False
            assert identical(var, set([])) is False
            assert identical(var, '') is False
            assert identical(var, ' ') is False
            assert identical(var, 'a') is False

    def test_str(self):
        for var in ['a', 'b', '', ' ', '.', 'abc']:
            assert identical(var, var) is True

        assert identical('', ' ') is False
        assert identical('', 'a') is False
        assert identical('', None) is False
        assert identical('', 1) is False
        assert identical('', 1.0) is False
        assert identical('', []) is False
        assert identical('', {}) is False
        assert identical('', set()) is False
        assert identical('', ['']) is False
        assert identical('', {'': 1}) is False

        assert identical('abc', ' abc') is False
        assert identical('abc', 'abc ') is False
        assert identical('abc', 'abcd') is False
        assert identical('abc', 'ab') is False
        assert identical('abc', ['abc']) is False

    def test_list(self):
        for var in [[], [[]], [None], [1], ['a'], range(3), [1, 'a']]:
            assert identical(var, var) is True

        assert identical([1, 2, 3], range(1, 4)) is True

        assert identical([], ['']) is False
        assert identical([], ['a']) is False
        assert identical([], '') is False
        assert identical([], '[]') is False
        assert identical([], 'a') is False
        assert identical([], 1) is False
        assert identical([], 1.0) is False
        assert identical([], {}) is False
        assert identical([], set()) is False
        assert identical([], [1, 2, 3]) is False
        assert identical([1, 2, 3], slice(1, 4)) is False

    def test_dict(self):
        x = {'a': 1, 'b': 'v', 'c': -0.1}
        y = {'a': 1, 'b': 'v', 'c': -0.1}
        assert identical(x, x) is True
        assert identical(x, dict(x)) is True
        assert identical(x, y) is True
        assert identical(dict(x), dict(y)) is True

        assert identical({}, {}) is True
        assert identical({}, dict()) is True

        y['a'] = []
        assert identical(x, y) is False

    def test_set(self):
        assert identical(set(), set()) is True
        assert identical(set(range(100)), set(range(100))) is True
        assert identical({1}, {1}) is True
        assert identical({1, 'a', 1.234}, {1, 'a', 1.234}) is True

        # These are quirks of sets and python
        assert identical({1}, {1.0}) is True
        assert identical({1}, {1.0, 1}) is True

        assert identical({1}, {1.0, 2}) is False
        assert identical({1, 'a', 1.234}, {}) is False


class TestArray:
    x = Array([1, 2, 3])
    y = Array([1.0, 2.0, 3.0])
    z = Array([1, 2, 3, 4])
    w = Array([1, 2])

    def test_int_dtype(self):
        assert identical(self.x, self.x) is True
        assert identical(self.x, self.y) is False
        assert identical(self.x, self.z) is False
        assert identical(self.x, self.w) is False

    def test_float_dtype(self):
        assert identical(self.y, self.y) is True

