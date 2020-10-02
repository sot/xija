# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import json
from pathlib import Path
import re
import pytest
import requests

from ..get_model_spec import (get_xija_model_file, get_xija_model_names,
                              get_repo_version, check_github_version)


def test_get_model_file_aca():
    fn = get_xija_model_file('aca')
    assert fn.startswith(os.environ['SKA'])
    assert Path(fn).name == 'aca_spec.json'
    spec = json.load(open(fn))
    assert spec['name'] == 'aacccdpt'


def test_get_model_file_fail():
    with pytest.raises(ValueError, match='no models matched xxxyyyzzz'):
        get_xija_model_file('xxxyyyzzz')

    with pytest.raises(FileNotFoundError, match='xija models directory'):
        get_xija_model_file('aca', repo_path='__NOT_A_DIRECTORY__')


def test_get_xija_model_names():
    names = get_xija_model_names()
    assert all(name in names for name in ('aca', 'acisfp', 'dea', 'dpa', 'pftank2t'))


def test_get_repo_version():
    version = get_repo_version()
    assert isinstance(version, str)
    assert re.match(r'^[0-9.]+$', version)


def test_check_github_version():
    version = get_repo_version()
    status = check_github_version(version)
    assert status is True

    status = check_github_version('asdf')
    assert status is False

    # Force timeout
    status = check_github_version(version, timeout=0.00001)
    assert status is None

    with pytest.raises(requests.ConnectionError):
        check_github_version(version, 'https://______bad_url______')
