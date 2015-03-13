from .model import *
from .component import *
from .files import files
from .version import version as __version__


def test(*args, **kwargs):
    """Run self tests"""
    import os
    import pytest
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    pytest.main(args=['xija'] + list(args))
