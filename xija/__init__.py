from .model import *
from .component import *
from .files import files
from .version import version as __version__


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
