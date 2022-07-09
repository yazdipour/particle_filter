# -*- coding: utf-8 -*-
"""
.. codeauthor:: Alexander Vorndran <alexander.vorndran@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke.fischedick@tu-ilmenau.de>
"""
import math

import pytest
import numpy as np

from ...hash_function import numpy_hash_normalized, js_hash


@pytest.mark.parametrize("precision", [2, 3])
@pytest.mark.parametrize("as_jshash", [False, True])
@pytest.mark.parametrize("shape", [
    (1, ),
    (1, 1),
    (1, 20),
    (2, 2),
    (3, 3),
    (10, 10),
    (20, 20),
    (20, 1),
    (42, 37),
    (100, ),
    (100, 1),
    (640, 480, 3)
])
def test_random_numpy_hash_normalized(shape, precision, as_jshash, n_runs=50):
    """Test numpy_hash_normalized with random data"""
    rng = np.random.RandomState([2018, 5, 5])   # pylint: disable=no-member
    arr = np.around(rng.random_sample(size=shape), decimals=precision)

    assert arr.shape == shape
    arr_hash = numpy_hash_normalized(arr, as_jshash=as_jshash, precision=precision)

    n_modifications = arr.size if n_runs > arr.size else n_runs
    mod_indices = rng.choice(np.arange(arr.size), size=(n_modifications, ))

    # create a modifier that is smaller than the default precision change
    mod = math.pow(10.0, -precision-1)
    for i in mod_indices:
        mod_arr = np.zeros_like(arr)
        mod_arr.flat[i] = mod  # pylint: disable=unsupported-assignment-operation

        arr_add_same_hash = numpy_hash_normalized(
            arr+4.9*mod_arr, as_jshash=as_jshash, precision=precision)
        assert arr_add_same_hash == arr_hash

        arr_sub_same_hash = numpy_hash_normalized(
            arr-4.9*mod_arr, as_jshash=as_jshash, precision=precision)
        assert arr_sub_same_hash == arr_hash

        arr_add_diff_hash = numpy_hash_normalized(
            arr+5.1*mod_arr, as_jshash=as_jshash, precision=precision)
        assert arr_add_diff_hash != arr_hash

        arr_sub_diff_hash = numpy_hash_normalized(
            arr-5.1*mod_arr, as_jshash=as_jshash, precision=precision)
        assert arr_sub_diff_hash != arr_hash


@pytest.mark.slow
@pytest.mark.parametrize("precision", [2, 3])
@pytest.mark.parametrize("shape", [
    (1, ),
    (1, 1),
    (1, 20),
    (2, 2),
    (3, 3),
    (10, 10),
    (20, 20),
    (20, 1),
    (42, 37),
    (100, ),
    (100, 1),
    pytest.param(
        (640, 480, 3),
        marks=pytest.mark.xfail(reason="js_hash can only handle 1D/2D inputs")
    )
])
def test_random_js_hash_normalized(shape, precision, n_runs=50):
    """Test js_hash (with rounding) with random data"""
    rng = np.random.RandomState([2018, 5, 5]) # pylint: disable=no-member
    arr = np.around(rng.random_sample(size=shape), decimals=precision)

    def js_hash_wrapper(sth):
        return js_hash(np.around(sth, decimals=precision))

    assert arr.shape == shape
    arr_hash = js_hash_wrapper(arr)

    n_modifications = arr.size if n_runs > arr.size else n_runs
    mod_indices = rng.choice(np.arange(arr.size), size=(n_modifications, ))

    for i in mod_indices:
        # add a modifier that is smaller than the default precision change
        mod = math.pow(10.0, -precision-1)
        mod_arr = np.zeros_like(arr)
        mod_arr.flat[i] = mod  # pylint: disable=unsupported-assignment-operation

        arr_add_same_hash = js_hash_wrapper(arr+4.9*mod_arr)
        assert arr_add_same_hash == arr_hash

        arr_sub_same_hash = js_hash_wrapper(arr-4.9*mod_arr)
        assert arr_sub_same_hash == arr_hash

        arr_add_diff_hash = js_hash_wrapper(arr+5.1*mod_arr)
        assert arr_add_diff_hash != arr_hash

        arr_sub_diff_hash = js_hash_wrapper(arr-5.1*mod_arr)
        assert arr_sub_diff_hash != arr_hash
