from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import Array, DataFrame


class TestScalarListDualIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_int_list(self):
        assert isinstance(self.x[0, [0]], DataFrame)
        assert isinstance(self.x[0, [0, 1]], DataFrame)
