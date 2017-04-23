from dataframe import DataFrame
from utils import is_list_unique, is_list_same


def is_iterable_dataframe_type(x):
    # Empty list returns True
    for item in x:
        if not isinstance(item, DataFrame):
            return False
    return True


def hstack(dfs):
    if is_iterable_dataframe_type(dfs):
        if is_list_same([df.nrow for df in dfs]):
            names_per_df = [name for df in dfs for name in df.names]
            if is_list_unique(names_per_df):
                items = [column for df in dfs for column in df.items()]
                return DataFrame.from_items(items)
            else:
                msg = 'column names must not be repeated across DataFrames'
                raise ValueError(msg)
        else:
            msg = 'all DataFrames must have the same number of rows'
            raise ValueError(msg)
    else:
        msg = 'all elements of input list must be DataFrame types'
        raise ValueError(msg)


def vstack(dfs):
    if is_iterable_dataframe_type(dfs):
        if is_list_same([df.ncol for df in dfs]):
            names_per_df = [frozenset(df.names) for df in dfs]
            if is_list_same(names_per_df):
                if len(names_per_df) == 0:
                    names = []
                else:
                    names = dfs[0].names
                dfs = [df[names] for df in dfs]
                dtypes_per_df = [tuple(df.dtypes) for df in dfs]
                if is_list_same(dtypes_per_df):
                    rows = [row for df in dfs for row in df.rows()]
                    return DataFrame.from_rows(rows)
                else:
                    msg = ('columns of the same name must have the same '
                           'dtype')
                    raise ValueError(msg)
            else:
                msg = 'all DataFrames must have the same column names'
                raise ValueError(msg)
        else:
            msg = 'all DataFrames must have the same number of columns'
            raise ValueError(msg)
    else:
        msg = 'all elements of input list must be DataFrame types'
        raise ValueError(msg)
