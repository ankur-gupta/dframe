import pytest
from .._column import Column


class TestEmptyColumnObject:
    x = Column([])

    def test_valid_object_creation(self):
        ''' Ensure object is created correctly '''
        assert len(self.x) == 0
        assert self.x.data == []
        assert self.x.dtype is type(None)

    def test_invalid_indexing_for_empty_column(self):
        ''' Ensure we see errors in indexing an empty Column object '''
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
            # Column cannot be indexed using str
            self.x['a']
        with pytest.raises(TypeError):
            # Column cannot be indexed using float
            self.x[34.45]

    def test_valid_indexing_slice(self):
        assert isinstance(self.x[:], Column)
        assert isinstance(self.x[1:], Column)
        assert isinstance(self.x[:2], Column)
        # This behavior matches list and pandas.
        assert isinstance(self.x[1:2], Column)


class TestBasicColumnIndexing:
    y = Column([1, 2, 3, 4, 5])

    def test_valid_object_creation(self):
        ''' Ensure object is created correctly '''
        assert len(self.y) == 5
        assert self.y.data == [1, 2, 3, 4, 5]
        assert self.y.dtype is int

    def test_valid_int_indexing(self):
        assert isinstance(self.y[0], Column)
        assert isinstance(self.y[1], Column)
        assert isinstance(self.y[2], Column)
        assert isinstance(self.y[3], Column)
        assert isinstance(self.y[4], Column)
        assert isinstance(self.y[-1], Column)
        assert isinstance(self.y[-2], Column)
        assert isinstance(self.y[-5], Column)

        assert self.y[0].data == [1]
        assert self.y[1].data == [2]
        assert self.y[2].data == [3]
        assert self.y[3].data == [4]
        assert self.y[4].data == [5]
        assert self.y[-1].data == [5]
        assert self.y[-2].data == [4]
        assert self.y[-5].data == [1]

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
            # Column cannot be indexed using str
            self.y['a']
        with pytest.raises(TypeError):
            # Column cannot be indexed using float
            self.y[34.45]
