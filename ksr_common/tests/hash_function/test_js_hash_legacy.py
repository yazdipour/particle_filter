# -*- coding: utf-8 -*-
"""
This module bundles the informal tests as defined by Markus Eisenbach
in the original version of the hash_function module
"""
from __future__ import print_function

import pytest
import numpy as np

from ...hash_function import js_hash, hash_matrikelnummer


def test_js_hash_matrix():
    """
    Test cases for js_hash matrix input
    """
    print("Test cases for hash functions")

    # matrix
    mat = np.matrix('1 2; 3 4')
    mat2 = np.matrix('2 3; 4 5')
    # one value of mat3 has noticeable difference to mat
    # thus it should get a different hash
    mat3 = np.matrix('1 2.001; 3 4')
    # mat4 is similar to mat except small differences (e.g. rounding errors)
    # thus it should get the same hash
    mat4 = np.matrix('0.999999 2.0000001; 2.999999 4.000001')
    print("====== np.matrix ======")
    print("Matrix = ", mat)
    print("HASH(Matrix) = ", js_hash(mat))
    print("HASH(Matrix.T) = ", js_hash(mat.T))
    print("Matrix2 = ", mat2)
    print("HASH(Matrix2) = ", js_hash(mat2))
    print("HASH(Matrix2.T) = ", js_hash(mat2.T))
    print("Matrix3 = ", mat3)
    print("HASH(Matrix3) = ", js_hash(mat3))
    print("HASH(Matrix3.T) = ", js_hash(mat3.T))
    print("cos(Matrix) = ", np.cos(mat))
    print("HASH(cos(Matrix)) = ", js_hash(np.cos(mat)))
    print("Matrix4 = ", mat4)
    print("HASH(Matrix4) = ", js_hash(mat4))
    print("HASH(Matrix4.T) = ", js_hash(mat4.T))
    # --- test cases ---
    assert js_hash(mat) != js_hash(mat.T), \
        "TEST A01 [HASH(Matrix) != HASH(Matrix.T)] successful"
    assert js_hash(mat) != js_hash(mat2), \
        "TEST A02 [HASH(Matrix) != HASH(Matrix2)] successful"
    assert js_hash(mat) != js_hash(mat2.T), \
        "TEST A03 [HASH(Matrix) != HASH(Matrix2.T)] successful"
    assert js_hash(mat.T) != js_hash(mat2), \
        "TEST A04 [HASH(Matrix.T) != HASH(Matrix2)] successful"
    assert js_hash(mat.T) != js_hash(mat2.T), \
        "TEST A05 [HASH(Matrix.T) != HASH(Matrix2.T)] successful"
    assert js_hash(mat) != js_hash(mat3), \
        "TEST A06 [HASH(Matrix) != HASH(Matrix3)] successful"
    assert js_hash(mat) != js_hash(mat3.T), \
        "TEST A07 [HASH(Matrix) != HASH(Matrix3.T)] successful"
    assert js_hash(mat.T) != js_hash(mat3), \
        "TEST A08 [HASH(Matrix.T) != HASH(Matrix3)] successful"
    assert js_hash(mat.T) != js_hash(mat3.T), \
        "TEST A09 [HASH(Matrix.T) != HASH(Matrix3.T)] successful"
    assert js_hash(mat) != js_hash(np.cos(mat)), \
        "TEST A10 [HASH(Matrix) != HASH(cos(Matrix))] successful"
    assert js_hash(mat) == js_hash(mat4), \
        "TEST A11 [HASH(Matrix) == HASH(Matrix4)] successful"
    assert js_hash(mat.T) == js_hash(mat4.T), \
        "TEST A12 [HASH(Matrix.T) == HASH(Matrix4.T)] successful"


def test_js_hash_array():
    """
    Test array input to js_hash function
    """
    # array
    arr = np.array([[1, 2], [3, 4]])
    mat = np.matrix('1 2; 3 4')
    print("====== np.array ======")
    print("Array = ", arr)
    print("HASH(Array) = ", js_hash(arr))
    print("HASH(Array.T) = ", js_hash(arr.T))
    # --- test cases ---
    print("TEST B01 [HASH(Array) != HASH(Array.T)] successful = ",
          js_hash(arr) != js_hash(arr.T))
    print("TEST B02 [HASH(Matrix) == HASH(Array)] successful = ",
          js_hash(mat) == js_hash(arr))
    print("TEST B03 [HASH(Matrix.T) == HASH(Array.T)] successful = ",
          js_hash(mat.T) == js_hash(arr.T))


def test_js_hash_list():
    """
    Test list input to js_hash function
    """
    # list
    lis = [[1, 2], [3, 4]]
    lis_transpose = [[1, 3], [2, 4]]
    mat = np.matrix('1 2; 3 4')
    print("====== list ======")
    print("List = ", lis)
    print("HASH(List) = ", js_hash(lis))
    print("HASH(List.T) = ", js_hash(lis_transpose))
    # --- test cases ---
    print("TEST C01 [HASH(List) != HASH(List.T)] successful = ",
          js_hash(lis) != js_hash(lis_transpose))
    print("TEST C02 [HASH(Matrix) == HASH(List)] successful = ",
          js_hash(mat) == js_hash(lis))
    print("TEST C03 [HASH(Matrix.T) == HASH(List.T)] successful = ",
          js_hash(mat.T) == js_hash(lis_transpose))


def test_js_hash_tuple():
    """
    Test tuple input to js_hash function
    """
    # tuple
    tup = ((1, 2), (3, 4))
    tup_transpose = ((1, 3), (2, 4))
    mat = np.matrix('1 2; 3 4')
    print("====== tuple ======")
    print("Tuple = ", tup)
    print("HASH(Tuple) = ", js_hash(tup))
    print("HASH(Tuple.T) = ", js_hash(tup_transpose))
    # --- test cases ---
    print("TEST D01 [HASH(Tuple) != HASH(Tuple.T)] successful = ",
          js_hash(tup) != js_hash(tup_transpose))
    print("TEST D01 [HASH(Matrix) == HASH(Tuple)] successful = ",
          js_hash(mat) == js_hash(tup))
    print("TEST D01 [HASH(Matrix.T) == HASH(Tuple.T)] successful = ",
          js_hash(mat.T) == js_hash(tup_transpose))


def test_js_hash_string():
    """
    Test string input to js_hash function
    """
    # string
    string = "test"
    string_capital_letter = "Test"
    string_space = "test "
    print("====== str ======")
    print("String = ", string)
    print("HASH(String) = ", js_hash(string))
    print("String2 = ", string_capital_letter)
    print("HASH(String2) = ", js_hash(string_capital_letter))
    print("String3 = ", string_space)
    print("HASH(String3) = ", js_hash(string_space))
    # --- test cases ---
    print("TEST E01 [HASH(String) != HASH(String2)] successful = ",
          js_hash(string) != js_hash(string_capital_letter))
    print("TEST E02 [HASH(String) != HASH(String3)] successful = ",
          js_hash(string) != js_hash(string_space))
    print("TEST E03 [HASH(String2) != HASH(String3)] successful = ",
          js_hash(string_capital_letter) != js_hash(string_space))


@pytest.mark.slow
def test_js_hash_matrikelnummer():
    """
    Test hash_matrikelnummer for hash collisions
    """
    # matrikelnummer
    matrikel_nr1 = 12345
    matrikel_nr2 = 12346
    print("M.-Nr.1 = ", matrikel_nr1)
    print("HASH(M.-Nr.1) = ", hash_matrikelnummer(matrikel_nr1))
    print("M.-Nr.2 = ", matrikel_nr2)
    print("HASH(M.-Nr.2) = ", hash_matrikelnummer(matrikel_nr2))
    assert hash_matrikelnummer(matrikel_nr1) != hash_matrikelnummer(matrikel_nr2), \
        "TEST F01 [HASH(M.-Nr.1) != HASH(M.-Nr.2)] successful"
    dif_bitwise = hash_matrikelnummer(matrikel_nr1) ^ \
        hash_matrikelnummer(matrikel_nr2)   # bitwise xor
    sum_dif = 0
    for i in range(32):
        bit = (dif_bitwise >> i) & 0x1
        sum_dif += bit
    print("Bit-Differences(HASH(M.-Nr.1), HASH(M.-Nr.2)) = ", sum_dif)
    assert sum_dif > 10, "TEST F02 [sum(bit-differences) > 10] successful"

    # count conflicts in matrikelnummer range
    print("check conflicts in relevant ranges")
    print("check if conflict probability in last 3 digits"
          " for seminar participants is very low")
    for i in range(1, 15):
        offset = i * 10000
        hash_list = []
        hash_suffix_list = []
        total_conflicts = 0
        total_conflicts_suffix = 0
        total_combinations = 20000 ** 2
        total_participants = 30
        seminar_combinations = total_participants ** 2
        for mnr in range(offset, offset + 20000):
            hash_ = hash_matrikelnummer(mnr)
            if hash_ in hash_list:
                total_conflicts += 1
            else:
                hash_list.append(hash_)
            suffix_3digits = hash_ % 10000
            if suffix_3digits in hash_suffix_list:
                total_conflicts_suffix += 1
            else:
                hash_suffix_list.append(suffix_3digits)
        print("#Conflicts in range [%s, %s) = " % (offset, offset + 20000),
              total_conflicts)
        assert total_conflicts == 0, \
            "TEST F%s [#Conflicts == 0] successful" % (2*i+1)
        conflict_probability = float(total_conflicts_suffix) * \
            float(seminar_combinations) / float(total_combinations)

        print("P(Confl.(3dig.) in seminar) in range [%s, %s) = " %
              (offset, offset + 20000), conflict_probability)
        assert conflict_probability < 0.05, \
            "TEST F%s [P(Confl.(3dig.)) < 0.05] successful" % (2*i+2)
