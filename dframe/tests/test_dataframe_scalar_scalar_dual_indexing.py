from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import DataFrame


class TestScalarScalarDualIndexing:
    # Define a dataframe that will be used for testing
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})

    def test_valid_int_int(self):
        assert self.x[0, 0] == 1
        assert self.x[0, 1] == 'a'
        assert self.x[1, 0] == 2
        assert self.x[1, 1] == 'b'
        assert self.x[2, 0] == 3
        assert self.x[2, 1] == 'c'
        assert self.x[0, -1] == 'a'
        assert self.x[1, -1] == 'b'
        assert self.x[2, -1] == 'c'
        assert self.x[0, -2] == 1
        assert self.x[1, -2] == 2
        assert self.x[2, -2] == 3

    def test_invalid_int_int(self):
        with pytest.raises(IndexError):
            self.x[3, 0]
        with pytest.raises(IndexError):
            self.x[3, 1]
        with pytest.raises(IndexError):
            self.x[3, 2]
        with pytest.raises(IndexError):
            self.x[3, 3]
        with pytest.raises(IndexError):
            self.x[3, -1]
        with pytest.raises(IndexError):
            self.x[3, -2]
        with pytest.raises(IndexError):
            self.x[3, -3]

        with pytest.raises(IndexError):
            self.x[0, 2]
        with pytest.raises(IndexError):
            self.x[0, 3]
        with pytest.raises(IndexError):
            self.x[0, -3]
        with pytest.raises(IndexError):
            self.x[1, 2]
        with pytest.raises(IndexError):
            self.x[1, 3]
        with pytest.raises(IndexError):
            self.x[1, -3]
        with pytest.raises(IndexError):
            self.x[2, 2]
        with pytest.raises(IndexError):
            self.x[2, 3]
        with pytest.raises(IndexError):
            self.x[2, -3]

        with pytest.raises(IndexError):
            self.x[-4, 0]
        with pytest.raises(IndexError):
            self.x[-4, 1]
        with pytest.raises(IndexError):
            self.x[-4, 2]
        with pytest.raises(IndexError):
            self.x[-4, -3]

        with pytest.raises(IndexError):
            self.x[10, 1]
        with pytest.raises(IndexError):
            self.x[1, 10]
        with pytest.raises(IndexError):
            self.x[10, 10]

    def test_valid_int_str(self):
        assert self.x[0, 'a'] == 1
        assert self.x[1, 'a'] == 2
        assert self.x[2, 'a'] == 3
        assert self.x[-1, 'a'] == 3
        assert self.x[-2, 'a'] == 2
        assert self.x[-3, 'a'] == 1
        assert self.x[0, 'b'] == 'a'
        assert self.x[1, 'b'] == 'b'
        assert self.x[2, 'b'] == 'c'
        assert self.x[-1, 'b'] == 'c'
        assert self.x[-2, 'b'] == 'b'
        assert self.x[-3, 'b'] == 'a'

    def test_invalid_int_str(self):
        with pytest.raises(KeyError):
            self.x[0, 'c']
        with pytest.raises(KeyError):
            self.x[0, 'invalid column']
        with pytest.raises(KeyError):
            self.x[0, ' ']
        with pytest.raises(KeyError):
            self.x[0, '']
        with pytest.raises(KeyError):
            self.x[1, 'c']
        with pytest.raises(KeyError):
            self.x[1, 'invalid column']
        with pytest.raises(KeyError):
            self.x[1, ' ']
        with pytest.raises(KeyError):
            self.x[1, '']
        with pytest.raises(KeyError):
            self.x[2, 'c']
        with pytest.raises(KeyError):
            self.x[2, 'invalid column']
        with pytest.raises(KeyError):
            self.x[2, ' ']
        with pytest.raises(KeyError):
            self.x[2, '']

        with pytest.raises(KeyError):
            self.x[3, 'c']
        with pytest.raises(KeyError):
            self.x[3, 'invalid column']
        with pytest.raises(KeyError):
            self.x[3, ' ']
        with pytest.raises(KeyError):
            self.x[3, '']
        with pytest.raises(KeyError):
            self.x[-1, 'c']
        with pytest.raises(KeyError):
            self.x[-1, 'invalid column']
        with pytest.raises(KeyError):
            self.x[-1, ' ']
        with pytest.raises(KeyError):
            self.x[-1, '']
        with pytest.raises(KeyError):
            self.x[-2, 'c']
        with pytest.raises(KeyError):
            self.x[-2, 'invalid column']
        with pytest.raises(KeyError):
            self.x[-2, ' ']
        with pytest.raises(KeyError):
            self.x[-2, '']

        with pytest.raises(IndexError):
            self.x[3, 'a']
        with pytest.raises(IndexError):
            self.x[4, 'a']
        with pytest.raises(IndexError):
            self.x[-4, 'a']
        with pytest.raises(IndexError):
            self.x[-5, 'a']
        with pytest.raises(IndexError):
            self.x[3, 'b']
        with pytest.raises(IndexError):
            self.x[4, 'b']
        with pytest.raises(IndexError):
            self.x[-4, 'b']
        with pytest.raises(IndexError):
            self.x[-5, 'b']
        with pytest.raises(KeyError):
            self.x[3, 'invalid column']
        with pytest.raises(KeyError):
            self.x[4, 'invalid column']
        with pytest.raises(KeyError):
            self.x[-4, 'invalid column']
        with pytest.raises(KeyError):
            self.x[-5, 'invalid column']
        with pytest.raises(KeyError):
            self.x[3, ' ']
        with pytest.raises(KeyError):
            self.x[4, ' ']
        with pytest.raises(KeyError):
            self.x[-4, ' ']
        with pytest.raises(KeyError):
            self.x[-5, ' ']
        with pytest.raises(KeyError):
            self.x[3, '']
        with pytest.raises(KeyError):
            self.x[4, '']
        with pytest.raises(KeyError):
            self.x[-4, '']
        with pytest.raises(KeyError):
            self.x[-5, '']

    def test_valid_str_int(self):
        ''' Rows cannot be addressed by str '''
        pass

    def test_invalid_str_int(self):
        ''' Rows cannot be addressed by str '''
        with pytest.raises(TypeError):
            self.x['a', 0]
        with pytest.raises(TypeError):
            self.x['a', 1]
        with pytest.raises(IndexError):
            self.x['a', 2]
        with pytest.raises(IndexError):
            self.x['a', 3]
        with pytest.raises(TypeError):
            self.x['a', -1]
        with pytest.raises(TypeError):
            self.x['a', -2]
        with pytest.raises(IndexError):
            self.x['a', -3]

        with pytest.raises(TypeError):
            self.x['b', 0]
        with pytest.raises(TypeError):
            self.x['b', 1]
        with pytest.raises(IndexError):
            self.x['b', 2]
        with pytest.raises(IndexError):
            self.x['b', 3]
        with pytest.raises(TypeError):
            self.x['b', -1]
        with pytest.raises(TypeError):
            self.x['b', -2]
        with pytest.raises(IndexError):
            self.x['b', -3]

        with pytest.raises(TypeError):
            self.x['', 0]
        with pytest.raises(TypeError):
            self.x['', 1]
        with pytest.raises(IndexError):
            self.x['', 2]
        with pytest.raises(IndexError):
            self.x['', 3]
        with pytest.raises(TypeError):
            self.x['', -1]
        with pytest.raises(TypeError):
            self.x['', -2]
        with pytest.raises(IndexError):
            self.x['', -3]

        with pytest.raises(TypeError):
            self.x[' ', 0]
        with pytest.raises(TypeError):
            self.x[' ', 1]
        with pytest.raises(IndexError):
            self.x[' ', 2]
        with pytest.raises(IndexError):
            self.x[' ', 3]
        with pytest.raises(TypeError):
            self.x[' ', -1]
        with pytest.raises(TypeError):
            self.x[' ', -2]
        with pytest.raises(IndexError):
            self.x[' ', -3]

        with pytest.raises(TypeError):
            self.x['invalid column', 0]
        with pytest.raises(TypeError):
            self.x['invalid column', 1]
        with pytest.raises(IndexError):
            self.x['invalid column', 2]
        with pytest.raises(IndexError):
            self.x['invalid column', 3]
        with pytest.raises(TypeError):
            self.x['invalid column', -1]
        with pytest.raises(TypeError):
            self.x['invalid column', -2]
        with pytest.raises(IndexError):
            self.x['invalid column', -3]

    def test_valid_str_str(self):
        ''' Rows cannot be addressed by str '''
        pass

    def test_invalid_str_str(self):
        ''' Rows cannot be addressed by str '''
        with pytest.raises(TypeError):
            self.x['a', 'a']
        with pytest.raises(TypeError):
            self.x['a', 'b']
        with pytest.raises(KeyError):
            self.x['a', 'c']
        with pytest.raises(KeyError):
            self.x['a', '']
        with pytest.raises(KeyError):
            self.x['a', ' ']
        with pytest.raises(KeyError):
            self.x['a', 'invalid column']

        with pytest.raises(TypeError):
            self.x['b', 'a']
        with pytest.raises(TypeError):
            self.x['b', 'b']
        with pytest.raises(KeyError):
            self.x['b', 'c']
        with pytest.raises(KeyError):
            self.x['b', '']
        with pytest.raises(KeyError):
            self.x['b', ' ']
        with pytest.raises(KeyError):
            self.x['b', 'invalid column']

        with pytest.raises(TypeError):
            self.x['', 'a']
        with pytest.raises(TypeError):
            self.x['', 'b']
        with pytest.raises(KeyError):
            self.x['', 'c']
        with pytest.raises(KeyError):
            self.x['', '']
        with pytest.raises(KeyError):
            self.x['', ' ']
        with pytest.raises(KeyError):
            self.x['', 'invalid column']

        with pytest.raises(TypeError):
            self.x[' ', 'a']
        with pytest.raises(TypeError):
            self.x[' ', 'b']
        with pytest.raises(KeyError):
            self.x[' ', 'c']
        with pytest.raises(KeyError):
            self.x[' ', '']
        with pytest.raises(KeyError):
            self.x[' ', ' ']
        with pytest.raises(KeyError):
            self.x[' ', 'invalid column']

        with pytest.raises(TypeError):
            self.x['invalid column', 'a']
        with pytest.raises(TypeError):
            self.x['invalid column', 'b']
        with pytest.raises(KeyError):
            self.x['invalid column', 'c']
        with pytest.raises(KeyError):
            self.x['invalid column', '']
        with pytest.raises(KeyError):
            self.x['invalid column', ' ']
        with pytest.raises(KeyError):
            self.x['invalid column', 'invalid column']



