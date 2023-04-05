# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

import pytest
import requests
import git

from ..get_model_spec import (
    get_xija_model_spec,
    get_xija_model_names,
    get_repo_version,
    get_github_version,
)

try:
    # Fast request to see if GitHub is available
    req = requests.get(
        'https://raw.githubusercontent.com/sot/chandra_models/master/README.md',
        timeout=5,
    )
    HAS_GITHUB = req.status_code == 200
except Exception:
    HAS_GITHUB = False


def test_get_model_spec_aca_3_30():
    # Version 3.30
    spec, version = get_xija_model_spec('aca', version='3.30')
    assert spec['name'] == 'aacccdpt'
    assert 'comps' in spec
    assert spec["datestop"] == "2018:305:11:52:30.816"


def test_get_model_spec_aca_latest():
    # Latest version
    spec, version = get_xija_model_spec('aca', check_version=HAS_GITHUB)
    assert spec['name'] == 'aacccdpt'
    # version 3.30 value, changed in 3.31.1/3.32
    assert spec["datestop"] != "2018:305:11:52:30.816"
    assert 'comps' in spec


@pytest.mark.skipif('not HAS_GITHUB')
def test_get_model_spec_aca_from_github():
    # Latest version
    repo_path = 'https://github.com/sot/chandra_models.git'
    spec, version = get_xija_model_spec('aca', repo_path=repo_path, version='3.30')
    assert spec['name'] == 'aacccdpt'
    assert spec["datestop"] == "2018:305:11:52:30.816"
    assert 'comps' in spec


def test_get_model_file_fail():
    with pytest.raises(ValueError, match='no models matched xxxyyyzzz'):
        get_xija_model_spec('xxxyyyzzz')

    with pytest.raises(git.GitCommandError, match='does not exist'):
        get_xija_model_spec('aca', repo_path='__NOT_A_DIRECTORY__')


def test_get_xija_model_names():
    names = get_xija_model_names()
    assert all(name in names for name in ('aca', 'acisfp', 'dea', 'dpa', 'pftank2t'))


def test_get_repo_version():
    version = get_repo_version()
    assert isinstance(version, str)
    assert re.match(r'^[0-9.]+$', version)


@pytest.mark.skipif('not HAS_GITHUB')
def test_check_github_version():
    version = get_repo_version()
    status = get_github_version() == version
    assert status is True

    status = get_github_version() == 'asdf'
    assert status is False

    # Force timeout
    status = get_github_version(timeout=0.00001)
    assert status is None

    with pytest.raises(requests.ConnectionError):
        get_github_version(url='https://______bad_url______')
