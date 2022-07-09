# -*- coding: utf-8 -*-
"""
Content:
General purpose and application specific hash-functions.

@author: Markus Eisenbach, Alexander Vorndran
@date:   2015/03/18, 2018/04/11
"""
# pylint: disable=no-member
from __future__ import print_function

import math
import hashlib

import numpy

__all__ = ["js_hash", "numpy_hash", "numpy_hash_normalized"]


def numpy_hash(np_arr, as_jshash=True):
    """
    Faster way to hash (large) numpy arrays

    This function makes use of np.save, which is cross platform and hashlib as
    general purpose hash implementation.

    Parameters
    ----------
    np_arr : numpy.ndarray
        Array to be hashed. The array is assumed to be contiguous as the
        function
    as_jshash : bool, optional
        If `True`, the intermediate hash value is passed to js_hash to produce a
        compatible output. This is default behavior. On `False`, the full
        general purpose hash SHA256-digest in hex format is returned.

    Returns
    -------
    digest : {long, string}
        The data digest value. The return type depends on the setting of the
        `as_jshash` parameter (see description above).
    """
    if not np_arr.flags.c_contiguous:
        raise ValueError("np_arr must be c_contiguous")
    # Windows places a L after the shape values, so remove it if necessary
    digest = hashlib.sha256(repr(np_arr.shape).replace("L", "").encode())
    digest.update(np_arr.data)
    digest = digest.hexdigest()
    if as_jshash:
        return js_hash(digest)
    return digest


def numpy_hash_normalized(arr, as_jshash=True, precision=3):
    """Apply numpy_hash to "normalized" array

    The normalization process includes casting the whole array to float64,
    rounding it to a given number of digits, and makes a contiguous copy of
    the array data if necessary.
    """
    arr_ = numpy.round(
        numpy.asfarray(arr, numpy.float64), precision
    )
    if not arr_.flags.c_contiguous:
        arr_ = arr_.copy()
    return numpy_hash(arr_, as_jshash)


def _js_hash_update(hash_, value_8bit):
    """
    Update of JS hash function

    The hashing scheme is based on an algorithm attributed to Justin Sobel
    (see: http://www.partow.net/programming/hashfunctions/#JSHashFunction)
    """
    hash2 = numpy.uint32(hash_ << 5) # left shift
    hash3 = hash_ >> 2 # right shift
    hash5 = numpy.uint32(hash2 + value_8bit + hash3) # sum with overflow
    hash_updated = hash_ ^ hash5 # bitwise xor
    return hash_updated


def js_hash(something):
    """
    General purpose JS hash function

    The hashing scheme is based on an algorithm attributed to Justin Sobel
    (see: http://www.partow.net/programming/hashfunctions/#JSHashFunction)
    """
    # pylint: disable=too-many-branches, too-many-locals, too-many-statements
    # constants
    # pylint: disable=invalid-name
    UINT16_SCALE_FACTOR = 0xFFFF
    INITIAL_HASH = 0x4E67C6A7
    # pylint: enable=invalid-name

    matrix = something

    # check type
    is_string_type = False
    if isinstance(matrix, (list, tuple)):
        matrix = numpy.array(matrix)
    if isinstance(matrix, str):
        is_string_type = True
        # convert to ascii values
        matrix = numpy.array([[ord(c) for c in matrix]])

    # get shape of matrix
    if len(matrix.shape) < 2:
        matrix = numpy.atleast_2d(matrix)
    sx, sy = matrix.shape
    n = sx * sy

    # make it a vector (quantizize using 16 bits)
    array_uint16 = None
    if is_string_type:
        n_half = n // 2
        if (n % 2) == 1:  # odd
            matrix = numpy.concatenate((matrix, [[0]]), axis=1)
            n_half = (n + 1) // 2
        reshaped_matrix = numpy.reshape(matrix, (2, n_half))
        ls_bytes = reshaped_matrix[0, :]
        ms_bytes = numpy.multiply(reshaped_matrix[1, :], 0x100)
        array_uint16 = numpy.add(ls_bytes, ms_bytes)
        array_uint16 = numpy.concatenate((array_uint16, [n]))
    else:
        reshaped_matrix = numpy.reshape(matrix, n)
        min_value = numpy.min(reshaped_matrix)
        max_value = numpy.max(reshaped_matrix)
        dif = max_value - min_value
        descriptor_dif = 0
        magnitute_dif = 0
        if dif > 0:
            magnitute_dif = math.ceil(math.log10(dif))
            base_dif = 10 ** magnitute_dif
            descriptor_dif = round((dif / base_dif) * 0x7FFF) + 0x7FFF
            magnitute_dif += 0x7FFF
        else:
            dif = 1
        descriptor_max = 0
        magnitute_max = 0
        if max_value > 0:
            magnitute_max = math.ceil(math.log10(max_value))
            base_max = 10 ** magnitute_max
            descriptor_max = round((max_value / base_max) * 0x7FFF)
            magnitute_max += 0x7FFF
        elif max_value < 0:
            max_value_inv = -max_value
            magnitute_max = math.ceil(math.log10(max_value_inv))
            base_max = 10 ** magnitute_max
            descriptor_max = round(
                (max_value_inv / base_max) * 0x7FFF) + 0x7FFF
            magnitute_max += 0x7FFF
        min_sub = numpy.subtract(reshaped_matrix, min_value)
        normalized_mat = numpy.divide(min_sub, float(dif))
        uint16_scaled = numpy.multiply(normalized_mat, UINT16_SCALE_FACTOR)
        matrix_uint16 = numpy.round(uint16_scaled)
        array_uint16 = numpy.squeeze(numpy.asarray(matrix_uint16))
        if n == 1:
            array_uint16 = [array_uint16]
        descr = [n, sx, sy, descriptor_dif, magnitute_dif,
                 descriptor_max, magnitute_max]
        array_uint16 = numpy.concatenate((array_uint16, descr))

    # initialize hash value
    init = numpy.uint32(INITIAL_HASH)
    hash_ = init

    for element in array_uint16:
        # convert to integer type
        value = numpy.uint32(element)
        # update hash value with most significant 8 bits of 16bit element
        value_most_significant_byte = value >> 8  # right shift
        hash_ = _js_hash_update(hash_, value_most_significant_byte)
        # update hash value with least significant 8 bits of 16bit element
        value_least_significant_byte = value & 0xFF  # bitwise and
        hash_ = _js_hash_update(hash_, value_least_significant_byte)

    return hash_


def hash_matrikelnummer(matrikelnummer):
    """Generate a hash based on a Matrikelnummer"""
    # constants
    # pylint: disable=invalid-name
    BYTE_MASK = 0xFF
    UINT32_MASK = 0xFFFFFFFF
    UINT32_CORRECTION = 0x100000000
    INITIAL_HASH = 0x4E67C6A7
    # pylint: enable=invalid-name

    # preprocessing to get many differences for small changes
    # optimized for range from 10000 to 150000 (= range of matrikel)
    # additional benefit: makes inversion a bit harder
    v1 = float(matrikelnummer) * 180 / math.pi
    v2 = (float(matrikelnummer) - 10) * 180 / math.pi
    v3 = (float(matrikelnummer) / 3 - 100) * 180 / math.pi
    v4 = (float(matrikelnummer) / 2 - 33) * 180 / math.pi
    val = round((math.sin(v1) + math.sin(v2) + math.cos(v3)
                 - math.cos(v4) + 2.5) * (179 ** 4))
    # bring val into uint32 range
    while val < 0:
        val += UINT32_CORRECTION
    while val > UINT32_MASK:
        val -= UINT32_CORRECTION
    # convert to uint32
    nr = numpy.uint32(val)

    # divide nr into 4 bytes
    b1 = nr & BYTE_MASK
    b2 = (nr >> 8) & BYTE_MASK
    b3 = (nr >> 16) & BYTE_MASK
    b4 = (nr >> 24) & BYTE_MASK

    # initialize hash value
    init = numpy.uint32(INITIAL_HASH)
    hash_ = init ^ nr # bitwise xor

    # update hash value for all 4 bytes
    hash_ = _js_hash_update(hash_, b1)
    hash_ = _js_hash_update(hash_, b2)
    hash_ = _js_hash_update(hash_, b3)
    hash_ = _js_hash_update(hash_, b4)

    # optimize hash for displaying only last 3 digits
    # try to get equal distribution for all 3-digit-suffixes (101 - 998)
    hash_matrikel = ((hash_ // 898) % 4000000) * 1000 + (hash_ % 898) + 101

    return hash_matrikel
