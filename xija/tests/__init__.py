"""
Run self-tests
"""

def test(*args, **kwargs):
    import os
    import pytest
    os.chdir(os.path.dirname(__file__))
    pytest.main(*args, **kwargs)
