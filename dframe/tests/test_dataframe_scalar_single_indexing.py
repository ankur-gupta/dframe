from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import Array, DataFrame


class TestScalarTypeSingleIndexing:
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})

    def test_valid_int_indexing(self):
        ''' Ensure that int indexing works correctly '''
        assert isinstance(self.x[0], Array)
        assert not isinstance(self.x[0], DataFrame)
        assert self.x[0].equals(Array([1, 2, 3]))

        assert isinstance(self.x[1], Array)
        assert not isinstance(self.x[1], DataFrame)
        assert self.x[1].equals(Array(['a', 'b', 'c']))

        assert isinstance(self.x[-1], Array)
        assert not isinstance(self.x[-1], DataFrame)
        assert self.x[-1].equals(Array(['a', 'b', 'c']))

        assert isinstance(self.x[-2], Array)
        assert not isinstance(self.x[-2], DataFrame)
        assert self.x[-2].equals(Array([1, 2, 3]))

    def test_invalid_int_indexing(self):
        ''' Ensure that int indexing throws an error when needed '''
        with pytest.raises(IndexError):
            self.x[2]
        with pytest.raises(IndexError):
            self.x[3]
        with pytest.raises(IndexError):
            self.x[100]

        with pytest.raises(IndexError):
            self.x[-3]
        with pytest.raises(IndexError):
            self.x[-4]

    def test_valid_float_indexing(self):
        ''' No float index can be valid '''
        pass

    def test_invalid_float_indexing(self):
        ''' Float indexing should always return an error '''
        with pytest.raises(KeyError):
            self.x[0.0]
        with pytest.raises(KeyError):
            self.x[1.0]
        with pytest.raises(KeyError):
            self.x[-1.0]
        with pytest.raises(KeyError):
            self.x[87932.983993]

    def test_valid_str_indexing(self):
        assert isinstance(self.x['a'], Array)
        assert not isinstance(self.x['a'], DataFrame)
        assert self.x['a'].equals(Array([1, 2, 3]))

        assert isinstance(self.x['b'], Array)
        assert not isinstance(self.x['b'], DataFrame)
        assert self.x['b'].equals(Array(['a', 'b', 'c']))

    def test_invalid_str_indexing(self):
        with pytest.raises(KeyError):
            self.x['c']
        with pytest.raises(KeyError):
            self.x['']
        with pytest.raises(KeyError):
            self.x[' ']
        with pytest.raises(KeyError):
            self.x['invalid index']
