import pytest
from ..dataframe import DataFrame
# from dframe import DataFrame


class TestAtomicTypeSingleIndexing:
    # Define a dataframe that will be used for testing
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})

    def test_valid_int_indexing(self):
        ''' Ensure that int indexing works correctly '''
        assert isinstance(self.x[0], list)
        assert not isinstance(self.x[0], DataFrame)
        assert self.x[0] == [1, 2, 3]

        assert isinstance(self.x[1], list)
        assert not isinstance(self.x[1], DataFrame)
        assert self.x[1] == ['a', 'b', 'c']

        assert isinstance(self.x[-1], list)
        assert not isinstance(self.x[-1], DataFrame)
        assert self.x[-1] == ['a', 'b', 'c']

        assert isinstance(self.x[-2], list)
        assert not isinstance(self.x[-2], DataFrame)
        assert self.x[-2] == [1, 2, 3]

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
        assert isinstance(self.x['a'], list)
        assert not isinstance(self.x['a'], DataFrame)
        assert self.x['a'] == [1, 2, 3]

        assert isinstance(self.x['b'], list)
        assert not isinstance(self.x['b'], DataFrame)
        assert self.x['b'] == ['a', 'b', 'c']

    def test_invalid_str_indexing(self):
        with pytest.raises(KeyError):
            self.x['c']
        with pytest.raises(KeyError):
            self.x['']
        with pytest.raises(KeyError):
            self.x[' ']
        with pytest.raises(KeyError):
            self.x['invalid index']

    def test_valid_bool_indexing(self):
        ''' FILL ME '''
        pass

    def test_invalid_bool_indexing(self):
        ''' FILL ME '''
        pass

    def test_valid_none_indexing(self):
        ''' FILL ME '''
        pass

    def test_invalid_none_indexing(self):
        ''' FILL ME '''
        pass


class TestListSingleIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_list_of_int_indexing(self):
        assert isinstance(self.x[[0]], DataFrame)
        assert not isinstance(self.x[[0]], list)
        assert isinstance(self.x[[0]], DataFrame)
        assert not isinstance(self.x[[-1]], list)
        assert isinstance(self.x[[-2]], DataFrame)
        assert not isinstance(self.x[[-2]], list)
        assert isinstance(self.x[[0, 1, 2]], DataFrame)
        assert not isinstance(self.x[[0, 1, 2]], list)

    def test_valid_list_of_str_indexing(self):
        assert isinstance(self.x[['a']], DataFrame)
        assert not isinstance(self.x[['a']], list)
        assert isinstance(self.x[['a', 'b']], DataFrame)
        assert not isinstance(self.x[['a', 'b']], list)
        assert isinstance(self.x[['a', 'b', 'c']], DataFrame)
        assert not isinstance(self.x[['a', 'b', 'c']], list)

    def test_invalid_list_of_int_indexing(self):
        # Repeated columns cannot be returned because a dataframe
        # always has unique column names
        with pytest.raises(Exception):
            self.x[[0, 0]]
        with pytest.raises(Exception):
            self.x[[0, 0, 1]]
        with pytest.raises(Exception):
            self.x[[0, 0, 0]]
        with pytest.raises(Exception):
            self.x[[0, 0, 0]]
        with pytest.raises(Exception):
            self.x[[0, 0, 1, 1]]

        # Test out of range error
        with pytest.raises(IndexError):
            self.x[[3]]
        with pytest.raises(IndexError):
            self.x[[3, 4]]
        with pytest.raises(IndexError):
            self.x[[0, 1, 3]]
        with pytest.raises(IndexError):
            self.x[[0, 3]]
        with pytest.raises(IndexError):
            self.x[[-4]]
        with pytest.raises(IndexError):
            self.x[[-5]]

    def test_invalid_list_of_str_indexing(self):
        # Repeated columns cannot be returned because a dataframe
        # always has unique column names
        with pytest.raises(Exception):
            self.x[['a', 'a']]
        with pytest.raises(Exception):
            self.x[['a', 'a', 'b']]
        with pytest.raises(Exception):
            self.x[['a', 'a', 'a']]
        with pytest.raises(Exception):
            self.x[['a', 'a', 'b', 'b']]
        with pytest.raises(Exception):
            self.x[['a', 'invalid index']]

        # Test some common incorrect cases
        with pytest.raises(KeyError):
            self.x[['invalid column name']]
        with pytest.raises(KeyError):
            self.x[['']]
        with pytest.raises(KeyError):
            self.x[[' ']]

    def test_invalid_mixed_list_indexing(self):
        ''' Cannot mix int and str indexing '''
        with pytest.raises(KeyError):
            self.x[[0, 'a']]
        with pytest.raises(KeyError):
            self.x[[0, 'b']]
        with pytest.raises(KeyError):
            self.x[[0, 1, 'a']]
        with pytest.raises(KeyError):
            self.x[[0, 1, 'c']]
        with pytest.raises(KeyError):
            self.x[[0, '0']]
        with pytest.raises(KeyError):
            self.x[[0, '-1']]
        with pytest.raises(KeyError):
            self.x[[0, 'invalid index']]
        with pytest.raises(KeyError):
            self.x[[-1, 'invalid index']]
        with pytest.raises(KeyError):
            self.x[[0, -1, 'invalid index']]


class TestSetSingleIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_set_indexing(self):
        ''' No set indexing is valid '''
        pass

    def test_invalid_set_indexing(self):
        ''' No set indexing is valid '''
        with pytest.raises(KeyError):
            self.x[set([0, 1, 2])]
        with pytest.raises(KeyError):
            self.x[{0, 1, 2}]
        with pytest.raises(KeyError):
            self.x[{0}]
        with pytest.raises(KeyError):
            self.x[set([0])]
        with pytest.raises(KeyError):
            self.x[{-1}]
        with pytest.raises(KeyError):
            self.x[set([-1])]
        with pytest.raises(KeyError):
            self.x[set(['a'])]
        with pytest.raises(KeyError):
            self.x[{'a'}]
        with pytest.raises(KeyError):
            self.x[set(['invalid column name'])]
        with pytest.raises(KeyError):
            self.x[{'invalid column name'}]
        with pytest.raises(KeyError):
            self.x[set(['d'])]
        with pytest.raises(KeyError):
            self.x[{'d'}]
        with pytest.raises(KeyError):
            self.x[set(['a', 'b'])]
        with pytest.raises(KeyError):
            self.x[{'a', 'b'}]
        with pytest.raises(KeyError):
            self.x[set(['a', 'b'])]


class TestDictSingleIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_dict_indexing(self):
        ''' No dict indexing is valid '''
        pass

    def test_invalid_dict_indexing(self):
        ''' No dict indexing is valid '''
        with pytest.raises(KeyError):
            self.x[{'a': None, 'b': None}]
        with pytest.raises(KeyError):
            self.x[{'a': None}]
        with pytest.raises(KeyError):
            self.x[{'a': 1}]
        with pytest.raises(KeyError):
            self.x[{'a': [1, 2, 3]}]
        with pytest.raises(KeyError):
            self.x[{'some key': [1, 2, 3], 'b': True}]


class TestSliceSingleIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_slice_indexing(self):
        assert isinstance(self.x[:], DataFrame)
        assert not isinstance(self.x[:], list)
        assert isinstance(self.x[0:1], DataFrame)
        assert not isinstance(self.x[0:1], list)
        assert isinstance(self.x[0:2], DataFrame)
        assert not isinstance(self.x[0:2], list)
        assert isinstance(self.x[0:3:2], DataFrame)
        assert not isinstance(self.x[0:3:2], list)
        assert isinstance(self.x[0:], DataFrame)
        assert not isinstance(self.x[0:], list)
        assert isinstance(self.x[1:], DataFrame)
        assert not isinstance(self.x[1:], list)
        assert isinstance(self.x[2:], DataFrame)
        assert not isinstance(self.x[2:], list)

        # Empty dataframe
        assert isinstance(self.x[3:], DataFrame)
        assert not isinstance(self.x[3:], list)
        assert isinstance(self.x[4:], DataFrame)
        assert not isinstance(self.x[4:], list)
        assert isinstance(self.x[5:], DataFrame)
        assert not isinstance(self.x[5:], list)

        # Reverse
        assert isinstance(self.x[::-1], DataFrame)
        assert not isinstance(self.x[::-1], list)

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
