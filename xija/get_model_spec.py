# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Get Chandra model specifications
"""
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

from git import Repo
import requests
from Ska.File import get_globfiles

__all__ = ['get_xija_model_spec', 'get_xija_model_names', 'get_repo_version',
           'get_github_version']

REPO_PATH = Path(os.environ['SKA'], 'data', 'chandra_models')
MODELS_PATH = REPO_PATH / 'chandra_models' / 'xija'
CHANDRA_MODELS_URL = 'https://api.github.com/repos/sot/chandra_models/releases'


def _models_path(repo_path=REPO_PATH) -> Path:
    return Path(repo_path) / 'chandra_models' / 'xija'


def get_xija_model_spec(model_name, repo_path=REPO_PATH, check_version=False,
                        timeout=5) -> dict:
    """
    Get Xija model specification for the specified ``model_name``.

    Supported model names include (but are not limited to): ``'aca'``,
    ``'acisfp'``, ``'dea'``, ``'dpa'``, ``'psmc'``, ``'minusyz'``, and
    ``'pftank2t'``.

    Use ``get_xija_model_names()`` for the full list.

    Examples
    --------
    >>> import xija
    >>> from xija.get_model_spec import get_xija_model_spec
    >>> model_spec = get_xija_model_spec('acisfp')
    >>> model = xija.XijaModel('acisfp', model_spec=model_spec, start='2012:001', stop='2012:010')
    >>> model.make()
    >>> model.calc()

    Parameters
    ----------
    model_name : str
        Name of model
    repo_path : str, Path
        Path to directory containing chandra_models repository (default is
        $SKA/data/chandra_models)
    check_version : bool
        Check that the chandra_models repo at the same version as GitHub
    timeout : int, float
        Timeout (sec) for querying GitHub for the expected chandra_models version.
        Default = 5 sec.

    Returns
    -------
    dict
        Xija model specification dict
    """
    models_path = _models_path(repo_path)

    if not models_path.exists():
        raise FileNotFoundError(f'xija models directory {models_path} does not exist')

    # Get version and also check that local repo is clean and tag is at tip of HEAD
    repo_version = get_repo_version(repo_path)

    if check_version:
        gh_version = get_github_version(timeout=timeout)
        if gh_version is None:
            warnings.warn('Could not verify GitHub chandra_models release tag '
                          f'due to timeout ({timeout} sec)')
        elif repo_version != gh_version:
            raise ValueError(f'version mismatch: local repo {repo_version} vs '
                             f'github {gh_version}')

    file_glob = str(models_path / '*' / f'{model_name.lower()}_spec.json')
    try:
        # get_globfiles() default requires exactly one file match and returns a list
        file_name = get_globfiles(file_glob)[0]
    except ValueError:
        names = get_xija_model_names()
        raise ValueError(f'no models matched {model_name}. Available models are: '
                         f'{", ".join(names)}')

    model_spec = json.load(open(file_name, 'r'))
    return model_spec


def get_xija_model_names(repo_path=REPO_PATH) -> List[str]:
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
    models_path = _models_path(repo_path)

    fns = get_globfiles(str(models_path / '*' / '*_spec.json'), minfiles=0, maxfiles=None)
    names = [re.sub(r'_spec\.json', '', Path(fn).name) for fn in sorted(fns)]

    return names


def get_repo_version(repo_path: Path = REPO_PATH) -> str:
    """Return version (most recent tag) of models repository.

    Returns
    -------
    str
        Version (most recent tag) of models repository
    """
    repo = Repo(repo_path)

    if repo.is_dirty():
        raise ValueError('repo is dirty')

    tags = sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)
    tag_repo = tags[-1]
    if tag_repo.commit != repo.head.commit:
        raise ValueError(f'repo tip is not at tag {tag_repo}')

    return tag_repo.name


def get_github_version(url: str = CHANDRA_MODELS_URL,
                       timeout: Union[int, float] = 5) -> Optional[bool]:
    """Get latest chandra_models GitHub repo release tag (version).

    This queries GitHub for the latest release of chandra_models.

    Parameters
    ----------
    url : str
        URL for chandra_models releases on GitHub API
    timeout : int, float
        Request timeout (sec, default=5)

    Returns
    -------
    str, None
        Tag name (str) of latest chandra_models release on GitHub.
        None if the request timed out, indicating indeterminate answer.
    """
    try:
        req = requests.get(url, timeout=timeout)
    except requests.ConnectTimeout:
        return None

    if req.status_code != requests.codes.ok:
        req.raise_for_status()

    tags_gh = sorted(req.json(), key=lambda tag: tag['published_at'])
    tag_gh_name = tags_gh[-1]['tag_name']

    return tag_gh_name
