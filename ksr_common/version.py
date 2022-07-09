# -*- coding: utf-8 -*-
"""
.. codeauthor:: Alexander Vorndran <alexander.vorndran@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke.fischedick@tu-ilmenau.de>
"""

VERSION_MAJOR = 0
VERSION_MINOR = 2
VERSION_PATCH = 0
VERSION_SUFFIX = ""


def get_version(with_suffix=False):
    """Return the package version

    Parameters
    ----------
    with_suffix : bool, default: False
        return the version package version with a suffix (e.g. dev, rc1)

    Returns
    -------
    str
        the package version formatted as <major>.<minor>.<patch> with an
        optional suffix
    """
    version = "{}.{}.{}".format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)
    if with_suffix and VERSION_SUFFIX:
        version += "-{}".format(VERSION_SUFFIX)
    return version
