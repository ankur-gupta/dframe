from __future__ import print_function
from __future__ import absolute_import

import pytest
from dframe import Array, DataFrame


class TestListSingleIndexing:
    ''' These should return dataframes and not underlying list objects '''
    x = DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c'],
                   'c': [True, False, True]})

    def test_valid_list_of_int_indexing(self):
        assert isinstance(self.x[[0]], DataFrame)
        assert not isinstance(self.x[[0]], list)
        assert not isinstance(self.x[[0]], Array)

        assert isinstance(self.x[[0]], DataFrame)
        assert not isinstance(self.x[[-1]], list)
        assert not isinstance(self.x[[-1]], Array)

        assert isinstance(self.x[[-2]], DataFrame)
        assert not isinstance(self.x[[-2]], list)
        assert not isinstance(self.x[[-2]], Array)

        assert isinstance(self.x[[0, 1, 2]], DataFrame)
        assert not isinstance(self.x[[0, 1, 2]], list)
        assert not isinstance(self.x[[0, 1, 2]], Array)

    def test_valid_list_of_str_indexing(self):
        assert isinstance(self.x[['a']], DataFrame)
        assert not isinstance(self.x[['a']], list)
        assert not isinstance(self.x[['a']], Array)

        assert isinstance(self.x[['a', 'b']], DataFrame)
        assert not isinstance(self.x[['a', 'b']], list)
        assert not isinstance(self.x[['a', 'b']], Array)

        assert isinstance(self.x[['a', 'b', 'c']], DataFrame)
        assert not isinstance(self.x[['a', 'b', 'c']], list)
        assert not isinstance(self.x[['a', 'b', 'c']], Array)

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
        with pytest.raises(Exception):
            self.x[[0, 'a']]
        with pytest.raises(Exception):
            self.x[[0, 'b']]
        with pytest.raises(Exception):
            self.x[[0, 1, 'a']]
        with pytest.raises(Exception):
            self.x[[0, 1, 'c']]
        with pytest.raises(Exception):
            self.x[[0, '0']]
        with pytest.raises(Exception):
            self.x[[0, '-1']]
        with pytest.raises(Exception):
            self.x[[0, 'invalid index']]
        with pytest.raises(Exception):
            self.x[[-1, 'invalid index']]
        with pytest.raises(Exception):
            self.x[[0, -1, 'invalid index']]
