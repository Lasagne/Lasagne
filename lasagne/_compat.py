"""
    Some py2/py3 compatibility support based on a stripped down
    version of six so we don't have to depend on a specific version
    of it.

    Based on _compat.py in Jinja2
    https://github.com/mitsuhiko/jinja2/blob/master/jinja2/_compat.py


"""
import sys

PY2 = sys.version_info[0] == 2

if not PY2:
    range_type = range
    text_type = str
    string_types = (str,)

    import pickle

else:
    text_type = unicode
    range_type = xrange
    string_types = (str, unicode)

    import cPickle as pickle
