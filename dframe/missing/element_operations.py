def __not__(x):
    if x is None:
        return None
    else:
        return not x


def __neg__(x):
    if x is None:
        return None
    else:
        return x.__neg__()


def __pos__(x):
    if x is None:
        return None
    else:
        return x.__pos__()


def __abs__(x):
    if x is None:
        return None
    else:
        return x.__abs__()


def __eq__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x == y


def __ne__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x != y


def __ge__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x >= y


def __gt__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x > y


def __le__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x <= y


def __lt__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x < y


def __or__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__or__(y)


def __and__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__and__(y)


def __xor__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__xor__(y)


def __add__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__add__(y)


def __sub__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__sub__(y)


def __mul__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__mul__(y)


def __pow__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__pow__(y)


def __mod__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__mod__(y)


def __div__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__div__(y)


def __truediv__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__truediv__(y)


def __floordiv__(x, y):
    if (x is None) or (y is None):
        return None
    else:
        return x.__floordiv__(y)
