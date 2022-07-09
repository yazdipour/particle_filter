# -*- coding: utf-8 -*-
"""
.. codeauthor:: Alexander Vorndran <alexander.vorndran@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke.fischedick@tu-ilmenau.de>
"""
from . import hash_function
from . import string_suite
from .version import get_version as _get_version

__all__ = ["hash_function", "string_suite"]

__version__ = _get_version()
