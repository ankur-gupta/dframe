from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import DataFrame


class TestEmptyDataFrame:
    x = DataFrame()
    y = DataFrame({})

    def test_valid_object_creation(self):
        assert isinstance(self.x, DataFrame)
        assert isinstance(self.y, DataFrame)
        assert self.x.shape == (0, 0)
        assert self.y.shape == (0, 0)
        assert self.x.nrow == 0
        assert self.y.nrow == 0
        assert self.x.ncol == 0
        assert self.y.ncol == 0
        assert len(self.x.dtypes) == 0
        assert len(self.y.dtypes) == 0
        assert len(self.x.names) == 0
        assert len(self.y.names) == 0
