# Licensed under a 3-clause BSD style license - see LICENSE.rst
import ska_helpers

from .model import *
from .component import *
from .files import files

__version__ = ska_helpers.get_version(__package__)

def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
