# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Get Chandra model specifications
"""
import contextlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List

from Ska.File import get_globfiles
from ska_helpers import chandra_models
from ska_helpers.paths import chandra_models_repo_path, xija_models_path

__all__ = [
    "get_xija_model_spec",
    "get_xija_model_names",
    "get_repo_version",
    "get_github_version",
]

CHANDRA_MODELS_LATEST_URL = (
    "https://api.github.com/repos/sot/chandra_models/releases/latest"
)


# Define local names for API back-compatibility
get_repo_version = chandra_models.get_repo_version
get_github_version = chandra_models.get_github_version


@contextlib.contextmanager
def temp_directory():
    """Get name of a temporary directory that is deleted at the end.

    Like tempfile.TemporaryDirectory but without the bug that it fails to
    remove read-only files within the temp dir. Git repos can have read-only
    files.  https://bugs.python.org/issue26660.
    """
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def get_xija_model_spec(
    model_name, version=None, repo_path=None, check_version=False, timeout=5
) -> tuple:
    """
    Get Xija model specification for the specified ``model_name``.

    Supported model names include (but are not limited to): ``'aca'``,
    ``'acisfp'``, ``'dea'``, ``'dpa'``, ``'psmc'``, ``'minusyz'``, and
    ``'pftank2t'``.

    Use ``get_xija_model_names()`` for the full list.

    Examples
    --------
    Get the latest version of the ``acisfp`` model spec from the local Ska data
    directory ``$SKA/data/chandra_models``, checking that the version matches
    the latest release tag on GitHub.

    >>> import xija
    >>> from xija.get_model_spec import get_xija_model_spec
    >>> model_spec, version = get_xija_model_spec('acisfp', check_version=True)
    >>> model = xija.XijaModel('acisfp', model_spec=model_spec,
    ...                        start='2020:001', stop='2020:010')
    >>> model.make()
    >>> model.calc()

    Get the ``aca`` model spec from release version 3.30 of chandra_models from
    GitHub.

    >>> repo_path = 'https://github.com/sot/chandra_models.git'
    >>> model_spec, version = get_xija_model_spec('aca', version='3.30',
    ...                                           repo_path=repo_path)

    Parameters
    ----------
    model_name : str
        Name of model
    version : str, None
        Tag, branch or commit of chandra_models to use (default=latest tag from
        repo)
    repo_path : str, Path
        Path to directory or URL containing chandra_models repository (default
        is $SKA/data/chandra_models)
    check_version : bool
        Check that ``version`` matches the latest release on GitHub
    timeout : int, float
        Timeout (sec) for querying GitHub for the expected chandra_models version.
        Default = 5 sec.

    Returns
    -------
    tuple of dict, str
        Xija model specification dict, chandra_models version
    """
    if repo_path is None:
        repo_path = chandra_models_repo_path()

    if version is None:
        version = os.environ.get("CHANDRA_MODELS_DEFAULT_VERSION")

    with chandra_models.get_local_repo(repo_path, version) as (repo, repo_path_local):
        model_spec, version = _get_xija_model_spec(
            model_name, version, repo_path_local, check_version, timeout, repo=repo
        )
    return model_spec, version


def _get_xija_model_spec(
    model_name, version=None, repo_path=None, check_version=False, timeout=5, repo=None
) -> tuple:
    models_path = xija_models_path(repo_path)

    if not models_path.exists():
        raise FileNotFoundError(f"xija models directory {models_path} does not exist")

    file_glob = str(models_path / "*" / f"{model_name.lower()}_spec.json")
    try:
        # get_globfiles() default requires exactly one file match and returns a list
        file_name = get_globfiles(file_glob)[0]
    except ValueError:
        names = get_xija_model_names(repo_path)
        raise ValueError(
            f'no models matched {model_name}. Available models are: {", ".join(names)}'
        )

    model_spec = json.load(open(file_name, "r"))

    # Get version and ensure that repo is clean and tip is at latest tag
    if version is None:
        version = chandra_models.get_repo_version(repo=repo)

    if check_version:
        chandra_models.assert_latest_version(version, timeout)

    return model_spec, version


def get_xija_model_names(repo_path=None) -> List[str]:
    """Return list of available xija model names.

    Examples
    --------
    >>> from xija.get_model_spec import get_xija_model_names
    >>> names = get_xija_model_names()
    ['aca',
     'acisfp',
     'dea',
     'dpa',
     '4rt700t',
     'minusyz',
     'pm1thv2t',
     'pm2thv1t',
     'pm2thv2t',
     'pftank2t',
     'pline03t_model',
     'pline04t_model',
     'psmc',
     'tcylaft6']

    Parameters
    ----------
    repo_path : str, Path
        Path to directory containing chandra_models repository (default is
        $SKA/data/chandra_models)

    Returns
    -------
    list
        List of available xija model names
    """
    models_path = xija_models_path(repo_path)

    fns = get_globfiles(
        str(models_path / "*" / "*_spec.json"), minfiles=0, maxfiles=None
    )
    names = [re.sub(r"_spec\.json", "", Path(fn).name) for fn in sorted(fns)]

    return names
