# -*- coding: utf-8 -*-
"""Inputs and expected outputs for the first particle filter simulation"""
import os

from .particlefilter_utils import Example, RandomNumberSafe

IMG_NAME = "resources/oneDimensionalEnvironment.png"

START_POS = 80

PARTICLE_NUM = 200

RESULT_HASH = 3386008092

_DATA_FILE = os.path.join(os.path.dirname(__file__), "example1.npz")
RANDOM_NUMBERS = RandomNumberSafe.from_npz(_DATA_FILE)

EXAMPLE = Example(IMG_NAME, START_POS, PARTICLE_NUM, RESULT_HASH,
                  RANDOM_NUMBERS, 5, 100)
