from .version import version as __version__

# If a __builtin__ variable has been pre-defined that means
# this file is being called as part of running setup.py.
# In that case don't do any other imports, only the version
# is needed.
try:
    _RUNNING_SETUP_PY_
except NameError:
    from .model import *
    from .component import *
    from .files import files
