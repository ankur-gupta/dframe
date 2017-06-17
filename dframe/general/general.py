from __future__ import absolute_import
from __future__ import print_function


def identical(x, y):
    if x is y:
        output = True
    elif type(x) is not type(y):
        output = False
    elif isinstance(x, (int, float, bool, str, list, dict, set)):
        output = x == y
    else:
        try:
            output = x.equals(y)
            if (output is not True) and (output is not False):
                raise AttributeError('x.equals() is not useful')
        except AttributeError:
            try:
                output = y.equals(x)
                if (output is not True) and (output is not False):
                    raise AttributeError('y.equals() is not useful')
            except AttributeError:
                if (x == y) is True:
                    output = True
                else:
                    output = False
    return output
