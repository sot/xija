# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Get Chandra model specifications
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Union

from git import Repo
import requests
from Ska.File import get_globfiles

__all__ = ['get_xija_model_file', 'get_xija_model_names', 'get_repo_version',
           'check_github_version']

REPO_PATH = Path(os.environ['SKA'], 'data', 'chandra_models')
MODELS_PATH = REPO_PATH / 'chandra_models' / 'xija'
CHANDRA_MODELS_URL = 'https://api.github.com/repos/sot/chandra_models/releases'


def _models_path(repo_path=REPO_PATH) -> Path:
    return Path(repo_path) / 'chandra_models' / 'xija'


def get_xija_model_file(model_name, repo_path=REPO_PATH) -> str:
    """
    Get file name of Xija model specification for the specified ``model_name``.

    Supported model names include (but are not limited to): ``'aca'``,
    ``'acisfp'``, ``'dea'``, ``'dpa'``, ``'psmc'``, ``'minusyz'``, and
    ``'pftank2t'``.

    Use ``get_xija_model_names()`` for the full list.

    Examples
    --------
    >>> import xija
    >>> from xija.get_model_spec import get_xija_model_file
    >>> model_spec = get_xija_model_file('acisfp')
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

    Returns
    -------
    str
        File name of the corresponding Xija model specification
    """
    models_path = _models_path(repo_path)

    if not models_path.exists():
        raise FileNotFoundError(f'xija models directory {models_path} does not exist')

    file_glob = str(models_path / '*' / f'{model_name.lower()}_spec.json')
    try:
        # get_globfiles() default requires exactly one file match and returns a list
        file_name = get_globfiles(file_glob)[0]
    except ValueError:
        names = get_xija_model_names()
        raise ValueError(f'no models matched {model_name}. Available models are: '
                         f'{", ".join(names)}')

    return file_name


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


def check_github_version(tag_name: str, url: str = CHANDRA_MODELS_URL,
                         timeout: Union[int, float] = 5) -> Optional[bool]:
    """Check that latest chandra_models GitHub repo release matches ``tag_name``.

    This queries GitHub for the latest release of chandra_models.

    Parameters
    ----------
    tag_name : str
        Tag name e.g. '3.32'
    url : str
        URL for chandra_models releases on GitHub API
    timeout : int, float
        Request timeout (sec, default=5)

    Returns
    -------
    bool, None
        True if chandra_models release on GitHub matches tag_name.
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

    return tag_gh_name == tag_name
