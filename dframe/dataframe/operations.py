from __future__ import absolute_import
from __future__ import print_function

from dframe.scalar import is_list_same, is_list_unique
from dframe.compat import Iterable
from dframe.dataframe import DataFrame


def is_iterable_dataframe(x):
    ''' Returns True when all elements are DataFrame type.
        Empty list returns True. None(s) are not treated as DataFrame type.

        Args
        -----
        x (iterable)

        Returns
        --------
        bool
    '''
    assert isinstance(x, Iterable)
    for elem in x:
        if not isinstance(elem, DataFrame):
            return False
    return True


def hstack(dfs):
    ''' Horizontally stack a sequence of DataFrames. This is same as cbind().

        Args
        -----
            dfs (list-like): an iterable of DataFrame objects. All DataFrame
                objects must have the same number of rows. DataFrames must not
                have common column names.

        Returns
        --------
        DataFrame
    '''
    if is_iterable_dataframe(dfs):
        if is_list_same([df.nrow for df in dfs]):
            names = [name for df in dfs for name in df.names]
            if is_list_unique(names):
                items = [item for df in dfs for item in df.items()]
                return DataFrame.from_items(items)
            else:
                msg = 'repeated column names found'
                raise ValueError(msg)
        else:
            msg = 'DataFrames do not have the same number of rows'
            raise ValueError(msg)
    else:
        msg = 'all elements of input list must be DataFrame type'
        raise ValueError(msg)


def cbind(dfs):
    ''' Horizontally stack a sequence of DataFrames. This is same as hstack().

        Args
        -----
            dfs (list-like): an iterable of DataFrame objects. All DataFrame
                objects must have the same number of rows. DataFrames must not
                have common column names.

        Returns
        --------
        DataFrame
    '''
    return hstack(dfs)


def vstack(dfs):
    ''' Vertically stack a sequence of DataFrames.

        Args
        -----
            dfs (list-like): an iterable of DataFrame objects. All DataFrame
                objects must have the same number of columns. Columns must have
                the same order of dtypes across DataFrames. Column names need
                not be the same across DataFrames.

        Returns
        --------
        DataFrame: Column names are the same as the column names of the first
            DataFrame of the sequence.
    '''
    if is_iterable_dataframe(dfs):
        if is_list_same([df.ncol for df in dfs]):
            if len(dfs) == 0:
                names = []
            else:
                names = dfs[0].names
            dtypes = [tuple(df.dtypes) for df in dfs]
            if is_list_same(dtypes):
                items = [(name, [elem for df in dfs for elem in df[j]])
                         for j, name in enumerate(names)]
                return DataFrame.from_items(items)
            else:
                msg = 'columns must have the same dtypes in the same order'
                raise ValueError(msg)
        else:
            msg = 'DataFrames do not have the same number of columns'
            raise ValueError(msg)
    else:
        msg = 'all elements of input list must be DataFrame type'
        raise ValueError(msg)


def rbind(dfs):
    ''' Vertically stack a sequence of DataFrames.

        Args
        -----
            dfs (list-like): an iterable of DataFrame objects. All DataFrame
                objects must have the same number of columns. Columns must have
                the same order of dtypes across DataFrames and same set of
                column names. Column names need not be in the same order across
                DataFrames.

        Returns
        --------
        DataFrame: The order of column names is the same as the first
            DataFrame in the sequence.
    '''
    if is_iterable_dataframe(dfs):
        if is_list_same([df.ncol for df in dfs]):
            names_per_df = [frozenset(df.names) for df in dfs]
            if is_list_same(names_per_df):
                if len(names_per_df) == 0:
                    names = []
                else:
                    names = dfs[0].names
                dfs = [df[names] for df in dfs]
                dtypes = [tuple(df.dtypes) for df in dfs]
                if is_list_same(dtypes):
                    items = [(name, [elem for df in dfs for elem in df[name]])
                             for name in names]
                    return DataFrame.from_items(items)
                else:
                    msg = 'columns must have the same dtypes'
                    raise ValueError(msg)
            else:
                msg = 'DataFrames do not have the same column names'
                raise ValueError(msg)
        else:
            msg = 'DataFrames do not have the same number of columns'
            raise ValueError(msg)
    else:
        msg = 'all elements of input list must be DataFrame type'
        raise ValueError(msg)

