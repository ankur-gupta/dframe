from __future__ import absolute_import

from ..compat import Iterable
from ..dtypes import is_string, is_integer, is_bool
from .scalar import is_scalar
from .lists import is_list_same, is_list_unique


def is_iterable_unique(x):
    assert not is_scalar(x)
    return is_list_unique([elem for elem in x])


def is_iterable_same(x):
    assert not is_scalar(x)
    return is_list_same([elem for elem in x])


def is_iterable_string(x):
    ''' Returns True when all elements are string type.
        Empty list returns True. None(s) are not treated as string type.

        Args
        -----
        x (iterable)

        Returns
        --------
        bool
    '''
    assert isinstance(x, Iterable)
    for elem in x:
        if not is_string(elem):
            return False
    return True


def is_iterable_integer(x):
    ''' Returns True when all elements are integer type.
        Empty list returns True. None(s) are not treated as integer type.

        Args
        -----
        x (iterable)

        Returns
        --------
        bool
    '''
    assert isinstance(x, Iterable)
    for elem in x:
        if not is_integer(elem):
            return False
    return True


def is_iterable_bool(x):
    ''' Returns True when all elements are bool type.
        Empty list returns True. None(s) are not treated as bool type.

        Args
        -----
        x (iterable)

        Returns
        --------
        bool
    '''
    assert isinstance(x, Iterable)
    for elem in x:
        if not is_bool(elem):
            return False
    return True
