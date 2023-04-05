# Licensed under a 3-clause BSD style license - see LICENSE.rst

from fnmatch import fnmatch


def get_limit_spec(limit):
    """
    Based on an input *limit* string of the form:

    <system>.<type>.<direction>.<qualifier>

    (where the "qualifier" element is optional) return the various pieces
    into a dictionary for convenience.

    Parameters
    ----------
    limit : str
        The input limit string.

    Returns
    -------
    dict
        A dictionary specifying the names of the various pieces of the
        limit string.
    """
    words = limit.split(".")
    if len(words) < 3:
        raise RuntimeError(f"{limit} is not a valid limit string")
    limit_spec = {"system": words[0], "type": words[1], "direction": words[2]}
    if len(words) == 4:
        limit_spec["qualifier"] = words[3]
    else:
        limit_spec["qualifier"] = None
    return limit_spec


LIMIT_COLORS = {
    "*.acisi": "blue",
    "*.aciss": "purple",
    "*.aciss_hot": "red",
    "*.cold_ecs": "dodgerblue",
    "odb.caution.*": "gold",
    "safety.caution.*": "gold",
    "odb.warning.*": "red",
    "safety.warning.*": "red",
    "planning.caution.*": "dodgerblue",
    "planning.warning.*": "green",
    "planning.penalty.*": "gray",
    "*": "black",
}


def get_limit_color(limit):
    """
    Based on an input *limit* string of the form:

    <system>.<type>.<direction>.<qualifier>

    (where the "qualifier" element is optional) return the color to
    be used for plotting.

    Parameters
    ----------
    limit : str
        The input limit string.

    Returns
    -------
    str
        A string giving the color of the line.
    """
    for k, v in LIMIT_COLORS.items():
        if fnmatch(limit, k):
            color = v
            break
    return color
