"""
Version numbering for this module. The `major`, `minor`, and `bugfix` variables
hold the respective parts of the version number (bugfix is '0' if absent). The
`release` variable is True if this is a release, and False if this is a
development version.

NOTE: this code copied from astropy.version and simplified.  Any license
restrictions therein are applicable.
"""

version = '0.3'

_versplit = version.replace('dev', '').split('.')
major = int(_versplit[0])
minor = int(_versplit[1])
if len(_versplit) < 3:
    bugfix = 0
else:
    bugfix = int(_versplit[2])
del _versplit

release = not version.endswith('dev')


def _get_git_devstr():
    """Determines the number of revisions in this repository and returns "" if
    this is not possible.

    Returns
    -------
    devstr : str
        A string that begins with 'dev' to be appended to the version number
        string.
    """
    from os import path
    from subprocess import Popen, PIPE

    currdir = path.abspath(path.split(__file__)[0])

    try:
        p = Popen(['git', 'rev-list', 'HEAD'], cwd=currdir,
                  stdout=PIPE, stderr=PIPE, stdin=PIPE)
        stdout, stderr = p.communicate()
    except:
        # git command failed for some reason (probably no git)
        return ''

    if p.returncode != 0:
        return ''
    else:
        revs = stdout.decode('ascii').split('\n')
        return  '-r%s-%s' % (len(revs), revs[0][:7])


if not release:
    version = version + _get_git_devstr()
