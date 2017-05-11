from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import Array, DataFrame


class TestScalarSliceDualIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_int_slice(self):
        assert isinstance(self.x[0, :], DataFrame)
        assert not isinstance(self.x[0, :], list)
        assert not isinstance(self.x[0, :], Array)

        assert isinstance(self.x[1, :], DataFrame)
        assert not isinstance(self.x[1, :], list)
        assert not isinstance(self.x[1, :], Array)

        assert isinstance(self.x[2, :], DataFrame)
        assert not isinstance(self.x[2, :], list)
        assert not isinstance(self.x[2, :], Array)

        assert isinstance(self.x[-1, :], DataFrame)
        assert not isinstance(self.x[-1, :], list)
        assert not isinstance(self.x[-1, :], Array)

        assert isinstance(self.x[0, 0:], DataFrame)
        assert not isinstance(self.x[0, 0:], list)
        assert not isinstance(self.x[0, 0:], Array)

        assert isinstance(self.x[1, 0:], DataFrame)
        assert not isinstance(self.x[1, 0:], list)
        assert not isinstance(self.x[1, 0:], Array)

        assert isinstance(self.x[0, 0:], DataFrame)
        assert not isinstance(self.x[0, 0:], list)
        assert not isinstance(self.x[0, 0:], Array)

        assert isinstance(self.x[1, 0:], DataFrame)
        assert not isinstance(self.x[1, 0:], list)
        assert not isinstance(self.x[1, 0:], Array)

        assert isinstance(self.x[-1, 0:], DataFrame)
        assert not isinstance(self.x[-1, 0:], list)
        assert not isinstance(self.x[-1, 0:], Array)

        assert isinstance(self.x[1, 0:1], DataFrame)
        assert not isinstance(self.x[1, 0:1], list)
        assert not isinstance(self.x[1, 0:1], Array)

        assert isinstance(self.x[1, 0:2], DataFrame)
        assert not isinstance(self.x[1, 0:2], list)
        assert not isinstance(self.x[1, 0:2], Array)

        assert isinstance(self.x[1, 0:3:2], DataFrame)
        assert not isinstance(self.x[1, 0:3:2], list)
        assert not isinstance(self.x[1, 0:3:2], Array)
