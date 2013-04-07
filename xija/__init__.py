from .model import *
from .component import *
from .files import files
from .version import version as __version__

def test(*args, **kwargs):
    """Run self tests"""
    from . import tests
    tests.test(*args, **kwargs)
