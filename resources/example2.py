# -*- coding: utf-8 -*-
"""Inputs and expected outputs for the first particle filter simulation"""
import os

from .particlefilter_utils import Example, RandomNumberSafe

IMG_NAME = "resources/oneDimensionalEnvironment2.png"

START_POS = 600

PARTICLE_NUM = 150

RESULT_HASH = 1991735630

_DATA_FILE = os.path.join(os.path.dirname(__file__), "example2.npz")
RANDOM_NUMBERS = RandomNumberSafe.from_npz(_DATA_FILE)

EXAMPLE = Example(IMG_NAME, START_POS, PARTICLE_NUM, RESULT_HASH,
                  RANDOM_NUMBERS, -5, 75)
