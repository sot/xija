from .model import *
from .component import *
from .files import files

__version__ = '3.7.1'


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
