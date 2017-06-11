from __future__ import print_function
from __future__ import absolute_import
from builtins import range

import pytest
from dframe import Array, DataFrame
import numpy as np
import pandas as pd


class TestDataFrameFromRows:
    x = [[1, 'a', 3.4], [2, 'b', 5.3]]
    y = [[1, 'a', 3.4, 8], [2, 'b', 5.3]]
    names = ['a', 'b', 'c']

    def test_from_rows_empty(self):
        df = DataFrame.from_rows([])
        assert isinstance(df, DataFrame)
        assert df.shape == (0, 0)
        assert len(df.names) == 0
        assert df.names.equals(Array([]))
        assert len(df.dtypes) == 0
        assert df.dtypes.equals(Array([]))

    def test_from_rows_basic(self):
        xdf = DataFrame.from_rows(self.x)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == (2, 3)
        assert xdf.names.equals(Array(['C0', 'C1', 'C2']))
        assert xdf.dtypes.equals(Array([int, str, float]))
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                assert xdf[i, j] == self.x[i][j]

        xdf = DataFrame.from_rows(self.x, self.names)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == (2, 3)
        assert xdf.names.equals(Array(self.names))
        assert xdf.dtypes.equals(Array([int, str, float]))
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                assert xdf[i, j] == self.x[i][j]

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            DataFrame.from_rows([], ['a'])
        with pytest.raises(ValueError):
            DataFrame.from_rows([], [''])
        with pytest.raises(ValueError):
            DataFrame.from_rows(self.y)
        with pytest.raises(ValueError):
            DataFrame.from_rows(self.y, self.names)
        with pytest.raises(ValueError):
            DataFrame.from_rows(self.x, self.names[0:2])
        with pytest.raises(ValueError):
            DataFrame.from_rows(self.y, self.names[0:2])
        with pytest.raises(Exception):
            DataFrame.from_rows(1, self.names[0:2])


class TestDataFrameFromColumns:
    x = [[1, 2], ['a', 'b'], [3.4, 5.3]]
    y = [[1, 2], ['a', 'b'], [3.4, 5.3, 9.0]]
    names = ['a', 'b', 'c']

    def test_from_columns_basic(self):
        xdf = DataFrame.from_columns(self.x)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == (2, 3)
        assert xdf.names.equals(Array(['C0', 'C1', 'C2']))
        assert xdf.dtypes.equals(Array([int, str, float]))
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                assert xdf[i, j] == self.x[j][i]

        xdf = DataFrame.from_columns(self.x, self.names)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == (2, 3)
        assert xdf.names.equals(Array(self.names))
        assert xdf.dtypes.equals(Array([int, str, float]))
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                assert xdf[i, j] == self.x[j][i]

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            DataFrame.from_columns([], ['a'])
        with pytest.raises(ValueError):
            DataFrame.from_columns([], [''])
        with pytest.raises(ValueError):
            DataFrame.from_columns(self.y)
        with pytest.raises(ValueError):
            DataFrame.from_columns(self.y, self.names)
        with pytest.raises(ValueError):
            DataFrame.from_columns(self.x, self.names[0:2])
        with pytest.raises(ValueError):
            DataFrame.from_columns(self.y, self.names[0:2])
        with pytest.raises(Exception):
            DataFrame.from_columns(1, self.names[0:2])


class TestDataFrameFromNumpy:
    x = np.reshape(range(3 * 4), (3, 4))
    names = ['a', 'b', '', 'c']
    y = np.array([np.nan, 2.0, 3.4, 2.1, 9.0, 7.8]).reshape((3, 2))

    def test_empty_numpy(self):
        df = DataFrame.from_numpy(np.array([[]]))
        assert isinstance(df, DataFrame)
        assert df.shape == (0, 0)
        assert len(df.names) == 0
        assert df.names.equals(Array([]))
        assert len(df.dtypes) == 0
        assert df.dtypes.equals(Array([]))

    def test_from_numpy_basic(self):
        xdf = DataFrame.from_numpy(self.x)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == self.x.shape
        assert xdf.names.equals(Array(['C0', 'C1', 'C2', 'C3']))
        assert all(xdf.dtypes == int)
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                assert xdf[i, j] == self.x[i, j]

        xdf = DataFrame.from_numpy(self.x, self.names)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == self.x.shape
        assert xdf.names.equals(Array(self.names))
        assert all(xdf.dtypes == int)
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                assert xdf[i, j] == self.x[i, j]

    def test_from_numpy_with_nan(self):
        ydf = DataFrame.from_numpy(self.y)
        assert isinstance(ydf, DataFrame)
        assert ydf.shape == self.y.shape
        assert ydf.names.equals(Array(['C0', 'C1']))
        assert all(ydf.dtypes == float)
        for i in range(ydf.nrow):
            for j in range(ydf.ncol):
                if np.isnan(self.y[i, j]):
                    assert ydf[i, j] is None
                else:
                    assert ydf[i, j] == self.y[i, j]

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            DataFrame.from_numpy(np.array([]))
        with pytest.raises(ValueError):
            DataFrame.from_numpy(np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            DataFrame.from_numpy(np.zeros((2, 3, 4)))
        with pytest.raises(ValueError):
            DataFrame.from_numpy(np.zeros(()))


class TestDataFrameFromPandas:
    x = {'a': [1, 2, 3], 'b': ['a', 'b', 'c'], 'c': [True, False, True]}

    def test_empty_pandas(self):
        df = DataFrame.from_pandas(pd.DataFrame())
        assert isinstance(df, DataFrame)
        assert df.shape == (0, 0)
        assert len(df.names) == 0
        assert df.names.equals(Array([]))
        assert len(df.dtypes) == 0
        assert df.dtypes.equals(Array([]))

        df = DataFrame.from_pandas(pd.DataFrame([]))
        assert isinstance(df, DataFrame)
        assert df.shape == (0, 0)
        assert len(df.names) == 0
        assert df.names.equals(Array([]))
        assert len(df.dtypes) == 0
        assert df.dtypes.equals(Array([]))

        df = DataFrame.from_pandas(pd.DataFrame({}))
        assert isinstance(df, DataFrame)
        assert df.shape == (0, 0)
        assert len(df.names) == 0
        assert df.names.equals(Array([]))
        assert len(df.dtypes) == 0
        assert df.dtypes.equals(Array([]))

    def test_from_pandas_basic(self):
        xpd = pd.DataFrame(self.x)
        x_from_dict = DataFrame(self.x)
        x_from_pd = DataFrame.from_pandas(xpd)
        assert isinstance(x_from_dict, DataFrame)
        assert isinstance(x_from_pd, DataFrame)
        x_from_pd = x_from_pd[self.x.keys()]
        x_from_dict = x_from_dict[self.x.keys()]
        assert x_from_dict.shape == x_from_pd.shape
        assert x_from_dict.names.equals(x_from_pd.names)
        for i in range(x_from_dict.nrow):
            for j in range(x_from_dict.ncol):
                assert x_from_dict[i, j] == x_from_pd[i, j]

    def test_from_pandas_with_nan(self):
        xpd = pd.DataFrame(self.x)
        xpd.iloc[0, 0] = np.nan
        xdf = DataFrame.from_pandas(xpd)
        assert isinstance(xdf, DataFrame)
        assert xdf.shape == xpd.shape
        assert xdf[0, 0] is None
        for i in range(xdf.nrow):
            for j in range(xdf.ncol):
                if i == 0 and j == 0:
                    pass
                else:
                    assert xdf[i, j] == xpd.iloc[i, j]

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            DataFrame.from_pandas(pd.Series())
        with pytest.raises(ValueError):
            DataFrame.from_pandas(pd.Series([1, 2, 3]))


class TestDataFrameFromShape:
    def test_empty_from_shape(self):
        df = DataFrame.from_shape((0, 0))
        assert isinstance(df, DataFrame)
        assert df.shape == (0, 0)
        assert len(df.names) == 0
        assert df.names.equals(Array([]))
        assert len(df.dtypes) == 0
        assert df.dtypes.equals(Array([]))

    def test_from_shape_basic(self):
        shape = (2, 3)
        df = DataFrame.from_shape(shape)
        assert isinstance(df, DataFrame)
        assert df.shape == shape
        assert df.names.equals(Array(['C0', 'C1', 'C2']))
        assert all(df.dtypes == type(None))
        for i in range(df.nrow):
            for j in range(df.ncol):
                assert df[i, j] is None

    def test_invalid_construction(self):
        with pytest.raises(TypeError):
            DataFrame.from_shape()
        with pytest.raises(ValueError):
            DataFrame.from_shape((0, ))
        with pytest.raises(ValueError):
            DataFrame.from_shape(())
        with pytest.raises(ValueError):
            DataFrame.from_shape((-1, 0))
        with pytest.raises(ValueError):
            DataFrame.from_shape((0, -1))
        with pytest.raises(ValueError):
            DataFrame.from_shape((-1, -3))
        with pytest.raises(ValueError):
            DataFrame.from_shape((1.0, -3))
        with pytest.raises(ValueError):
            DataFrame.from_shape((1.0, 1))
        with pytest.raises(ValueError):
            DataFrame.from_shape((10, 1.0))
