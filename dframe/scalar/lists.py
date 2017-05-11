from __future__ import absolute_import


def is_list_unique(x):
    assert isinstance(x, list)
    return len(x) == len(set(x))


def is_list_same(x):
    assert isinstance(x, list)
    n_same_elements = len(set(x))
    return n_same_elements == 0 or n_same_elements == 1
