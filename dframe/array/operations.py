from __future__ import absolute_import
from __future__ import print_function

from dframe.array import Array


def is_na(array):
    assert isinstance(array, Array)
    return Array([e is None for e in array])


def is_missing(array):
    return is_na(array)


def is_none(array):
    return is_na(array)


def which(array, ignore_missing=False):
    assert isinstance(array, Array)
    if array.dtype is bool:
        if not ignore_missing:
            if any(is_na(array)):
                msg = 'logical array contains missing values (None)'
                raise IndexError(msg)
        return Array([i for i, e in enumerate(array) if e])
    else:
        msg = 'array must be logical (dtype = bool)'
        raise TypeError(msg)


def find(array, ignore_missing=False):
    return which(array, ignore_missing)


def where(array, ignore_missing=False):
    return which(array, ignore_missing)


def unique(array):
    return Array(list(set(array)))
