from __future__ import print_function
from __future__ import absolute_import
from builtins import range

import pytest
from dframe import Array


class TestEmptyArray:
    x = Array([])

    def test_valid_object_creation(self):
        ''' Ensure object is created correctly '''
        assert isinstance(self.x, Array)
        assert len(self.x) == 0
        assert self.x.dtype is type(None)

    def test_invalid_indexing_for_empty_column(self):
        ''' Ensure we see errors in indexing an empty Array object '''
        with pytest.raises(IndexError):
            self.x[0]
        with pytest.raises(IndexError):
            self.x[1]
        with pytest.raises(IndexError):
            self.x[100]
        with pytest.raises(IndexError):
            self.x[-1]
        with pytest.raises(IndexError):
            self.x[[0]]
        with pytest.raises(IndexError):
            self.x[[0, 1, 2]]

    def test_invalid_indexing_by_type(self):
        with pytest.raises(TypeError):
            # Array cannot be indexed using str
            self.x['a']
        with pytest.raises(TypeError):
            # Array cannot be indexed using float
            self.x[34.45]

    def test_valid_indexing_slice(self):
        assert isinstance(self.x[:], Array)
        assert isinstance(self.x[1:], Array)
        assert isinstance(self.x[:2], Array)
        assert isinstance(self.x[1:2], Array)


class TestBasicArrayIndexing:
    n = 5
    y = Array(range(n))

    def test_valid_object_creation(self):
        ''' Ensure object is created correctly '''
        assert len(self.y) == self.n
        assert self.y.dtype is int
        for i in range(self.n):
            assert self.y[i] == i

    def test_valid_int_indexing(self):
        for i in range(self.n):
            assert isinstance(self.y[i], int)
        for i in [-1, -2, -3, -4, -5]:
            assert isinstance(self.y[i], int)

    def test_invalid_int_indexing(self):
        with pytest.raises(IndexError):
            self.y[5]
        with pytest.raises(IndexError):
            self.y[500]
        with pytest.raises(IndexError):
            self.y[-6]
        with pytest.raises(IndexError):
            self.y[-7]

    def test_invalid_indexing_by_type(self):
        with pytest.raises(TypeError):
            # Array cannot be indexed using str
            self.y['a']
        with pytest.raises(TypeError):
            # Array cannot be indexed using str
            self.y['']
        with pytest.raises(TypeError):
            # Array cannot be indexed using str
            self.y[' ']
        with pytest.raises(TypeError):
            # Array cannot be indexed using str
            self.y['invalid']
        with pytest.raises(TypeError):
            # Array cannot be indexed using float
            self.y[34.45]


class TestArrayIsIn:
    x = Array([])
    y = Array([1, 2, 3])

    def test_empty1(self):
        output = self.x.isin([1])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

        output = self.x.isin([1, 2, 3, None])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

        output = self.x.isin([None])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

        output = self.x.isin([''])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

        output = self.x.isin([' '])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

        output = self.x.isin([' ', 'a', None])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

        output = self.x.isin([1.0, -1.0])
        assert isinstance(output, Array)
        assert len(output) == len(self.x)
        assert (output.dtype is bool) or (output.dtype is type(None))

    def test_errors(self):
        with pytest.raises(ValueError):
            self.x.isin(1)
        # Look here (June 14, 2017): string is iterable.
        with pytest.raises(ValueError):
            self.x.isin('a')
        with pytest.raises(ValueError):
            self.x.isin('')
        with pytest.raises(ValueError):
            self.x.isin('wefjkewfk')
        with pytest.raises(ValueError):
            self.x.isin(' ')
        with pytest.raises(ValueError):
            self.x.isin(None)
        with pytest.raises(ValueError):
            self.x.isin(1.0)
        with pytest.raises(ValueError):
            self.x.isin(-1.0)

        with pytest.raises(ValueError):
            self.y.isin(1)
        with pytest.raises(ValueError):
            self.y.isin('a')
        with pytest.raises(ValueError):
            self.y.isin('')
        with pytest.raises(ValueError):
            self.y.isin('wefjkewfk')
        with pytest.raises(ValueError):
            self.y.isin(' ')
        with pytest.raises(ValueError):
            self.y.isin(None)
        with pytest.raises(ValueError):
            self.y.isin(1.0)
        with pytest.raises(ValueError):
            self.y.isin(-1.0)

    def test_integer_array(self):
        output = self.y.isin([1])
        assert isinstance(output, Array)
        assert len(output) == len(self.y)
        assert (output.dtype is bool) or (output.dtype is type(None))
        assert output[0] is True
        assert output[1] is False
        assert output[2] is False

        output = self.y.isin([0])
        assert isinstance(output, Array)
        assert len(output) == len(self.y)
        assert (output.dtype is bool) or (output.dtype is type(None))
        assert output[0] is False
        assert output[1] is False
        assert output[2] is False

        output = self.y.isin(['a'])
        assert isinstance(output, Array)
        assert len(output) == len(self.y)
        assert (output.dtype is bool) or (output.dtype is type(None))
        assert output[0] is False
        assert output[1] is False
        assert output[2] is False

        output = self.y.isin([1.0])
        assert isinstance(output, Array)
        assert len(output) == len(self.y)
        assert (output.dtype is bool) or (output.dtype is type(None))
        assert output[0] is False
        assert output[1] is False
        assert output[2] is False

        output = self.y.isin([[1]])
        assert isinstance(output, Array)
        assert len(output) == len(self.y)
        assert (output.dtype is bool) or (output.dtype is type(None))
        assert output[0] is False
        assert output[1] is False
        assert output[2] is False


