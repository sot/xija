# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .model import *
from .component import *
from .files import files

__version__ = '3.11.1'

def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
