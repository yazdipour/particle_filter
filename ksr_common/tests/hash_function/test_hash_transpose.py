#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Alexander Vorndran <alexander.vorndran@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke.fischedick@tu-ilmenau.de>
"""
import six
import pytest
import numpy as np

from ...hash_function import numpy_hash_normalized, js_hash


@pytest.mark.parametrize("shape", six.moves.range(2, 21))
def test_numpy_hash_normalized_square(shape):
    """
    Test numpy_hash_normalized with a fixed number of square inputs
    """
    arr = np.arange(shape*shape).reshape((shape, shape))
    np_digest = numpy_hash_normalized(arr, as_jshash=False)
    np_digest_trans = numpy_hash_normalized(arr.T, as_jshash=False)
    assert np_digest != np_digest_trans


@pytest.mark.parametrize("shape", six.moves.range(2, 21))
def test_js_hash_square(shape):
    """
    Test numpy_hash_normalized with a fixed number of square inputs
    """
    arr = np.arange(shape*shape).reshape((shape, shape))
    np_digest = js_hash(arr)
    np_digest_trans = js_hash(arr.T)
    assert np_digest != np_digest_trans


@pytest.mark.parametrize("shape", [
    (2, 1),
    (3, 1),
    (4, 1),
    (10, 1),
    (100, 1),
    (2, 25),
    (3, 25),
    (4, 25),
])
def test_numpy_hash_normalized_nonsquare(shape, n_runs=50):
    """
    Test random transposed inputs
    """
    rng = np.random.RandomState([2018, 5, 8])
    for _ in six.moves.range(n_runs):
        arr = np.around(rng.random_sample(size=shape), decimals=3)
        np_hash = numpy_hash_normalized(arr, as_jshash=False)
        np_hash_transposed = numpy_hash_normalized(arr.T, as_jshash=False)
        assert np_hash != np_hash_transposed


@pytest.mark.parametrize("shape", [
    (2, 1),
    (3, 1),
    (4, 1),
    (10, 1),
    (100, 1),
    (2, 25),
    (3, 25),
    (4, 25),
])
def test_js_hash_nonsquare(shape, n_runs=50):
    """
    Test random transposed inputs
    """
    rng = np.random.RandomState([2018, 5, 8])
    for _ in six.moves.range(n_runs):
        arr = np.around(rng.random_sample(size=shape), decimals=3)
        assert js_hash(arr) != js_hash(arr.T)
