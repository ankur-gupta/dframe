from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import Array, DataFrame


class TestSliceSingleIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_slice_indexing(self):
        assert isinstance(self.x[:], DataFrame)
        assert not isinstance(self.x[:], list)
        assert not isinstance(self.x[:], Array)

        assert isinstance(self.x[0:1], DataFrame)
        assert not isinstance(self.x[0:1], list)
        assert not isinstance(self.x[0:1], Array)

        assert isinstance(self.x[0:2], DataFrame)
        assert not isinstance(self.x[0:2], list)
        assert not isinstance(self.x[0:2], Array)

        assert isinstance(self.x[0:3:2], DataFrame)
        assert not isinstance(self.x[0:3:2], list)
        assert not isinstance(self.x[0:3:2], Array)

        assert isinstance(self.x[0:], DataFrame)
        assert not isinstance(self.x[0:], list)
        assert not isinstance(self.x[0:], Array)

        assert isinstance(self.x[1:], DataFrame)
        assert not isinstance(self.x[1:], list)
        assert not isinstance(self.x[1:], Array)

        assert isinstance(self.x[2:], DataFrame)
        assert not isinstance(self.x[2:], list)
        assert not isinstance(self.x[2:], Array)

        # Empty dataframe
        assert isinstance(self.x[3:], DataFrame)
        assert not isinstance(self.x[3:], list)
        assert not isinstance(self.x[3:], Array)

        assert isinstance(self.x[4:], DataFrame)
        assert not isinstance(self.x[4:], list)
        assert not isinstance(self.x[4:], Array)

        assert isinstance(self.x[5:], DataFrame)
        assert not isinstance(self.x[5:], list)
        assert not isinstance(self.x[5:], Array)

        # Reverse
        assert isinstance(self.x[::-1], DataFrame)
        assert not isinstance(self.x[::-1], list)
        assert not isinstance(self.x[::-1], Array)

    def test_invalid_slice_indexing(self):
        with pytest.raises(TypeError):
            self.x[0.0:]
        with pytest.raises(TypeError):
            self.x[0.0:1.0]
        with pytest.raises(TypeError):
            self.x[0.0:2.0]
        with pytest.raises(TypeError):
            self.x[0.0:3.0:1]
        with pytest.raises(TypeError):
            self.x[0.0:3:1]
        with pytest.raises(TypeError):
            self.x[0:3:1.0]

