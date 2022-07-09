# -*- coding: utf-8 -*-
"""
###############################################################################
############################ NO IMPLEMENTATION HERE ###########################
###############################################################################
"""
from __future__ import print_function

import os

import numpy as np
HAS_IMREAD = True
try:
    from imageio import imread
except ImportError:
    HAS_IMREAD = False

# pylint: disable=wrong-import-order, wrong-import-position
from ksr_common.hash_function import js_hash
from ksr_common.string_suite import TestSuite

from resources.particlefilter_utils import test_plot_vs as plot_vs, RandomNumberSafe

try:
    from dev_particlefilter import ParticleFilter
except ImportError:
    from student_particlefilter import ParticleFilter

DEV_MODE = False

RED = [255, 0, 4, 255]
GREEN = [138, 242, 4, 255]

INIT_HASH = 3548632916
MOVE_HASH = 4067819392
MOVE_HASH_C_R = 3195571136
MOVE_HASH_C_L = 4062006495
WEIGHT_HASH = 3661711718
RESAMPLE_HASH = 2171753757
_CWD = os.path.dirname(__file__)
_WORLD_IMAGE = os.path.join(_CWD, "resources", "oneDimensionalEnvironment.png")
try:
    WORLD = np.load(_WORLD_IMAGE + ".npy")
except IOError as ex:
    if not HAS_IMREAD:
        # We don't have the precomputed numpy data and now image reading
        # capabilities. The only possibility is to give up here.
        raise ex
    WORLD = imread(_WORLD_IMAGE)
    np.save(_WORLD_IMAGE + ".npy", WORLD)


def test_particlefilter():
    """Test specific aspects of the particle filter implementation"""
    pf = ParticleFilter(
        WORLD.shape[1],
        RandomNumberSafe(
            init=[60, 80], move=[
                [0, -1],
                [0, +1],
                [0, -1],
            ], resample_idx=[0]
        )
    )

    ts = TestSuite("ParticleFilter")
    ts.test_open("Initialization")
    test_result = True

    pf.initialize(2)
    initial_d = [x.deep_copy() for x in pf.particles]
    test_init_h = js_hash(np.around([[p.x, p.w] for p in pf.particles], 3))
    init_cmp = test_init_h == INIT_HASH
    if DEV_MODE:
        print("INIT_HASH = {}".format(test_init_h))

    if not init_cmp:
        if any(p.w == 0 for p in pf.particles):
            print("HINT: It seems that your initial weight is 0. This is likely "
                  "due to the way Python 2 handles divisions. Please check "
                  "https://docs.python.org/2/tutorial/introduction.html#numbers "
                  "for further details.")
        print("{:!^50s}".format(" WARNING "))
        print("The init test went wrong so it is almost certain that all "
              "subsequent test will also fail.")
        print("{:!^50s}".format(" WARNING "))
    ts.test_close(init_cmp, "Init went wrong!")
    test_result = test_result and init_cmp

    ts.test_open("Move")
    pf.move(10)
    test_move_h = js_hash(np.around([[p.x, p.w] for p in pf.particles], 3))
    move_cmp = test_move_h == MOVE_HASH
    if DEV_MODE:
        print("MOVE_HASH = {}".format(test_move_h))
    ts.test_close(move_cmp, "Move went wrong!")
    test_result = test_result and move_cmp

    ts.test_open("Weighting")
    pf.weight(WORLD, RED)
    test_weight_h = js_hash(np.around([[p.x, p.w] for p in pf.particles], 3))
    weight_cmp = test_weight_h == WEIGHT_HASH
    if DEV_MODE:
        print("WEIGHT_HASH = {}".format(test_weight_h))
    ts.test_close(weight_cmp, "Weighting went wrong!")
    test_result = test_result and weight_cmp

    ts.test_open("Resampling")
    pf.resample()
    resample_d = [x.deep_copy() for x in pf.particles]
    test_resample_h = js_hash(np.around([[p.x, p.w] for p in pf.particles], 3))
    resample_cmp = test_resample_h == RESAMPLE_HASH
    if DEV_MODE:
        print("RESAMPLE_HASH = {}".format(test_resample_h))
    ts.test_close(resample_cmp, "Resampling went wrong")
    test_result = test_result and resample_cmp

    ts.test_open("Move - Edge cases")
    # reset particle position
    for i, particle in enumerate(pf.particles, 0):
        particle.x = i*10
    pf.move(WORLD.shape[1])
    test_move_h_r = js_hash(np.around([[p.x, p.w] for p in pf.particles], 3))

    pf.move(-WORLD.shape[1])
    test_move_h_l = js_hash(np.around([[p.x, p.w] for p in pf.particles], 3))
    move_cmp = test_move_h_l == MOVE_HASH_C_L and test_move_h_r == MOVE_HASH_C_R
    if DEV_MODE:
        print("MOVE_HASH_C_R = {}".format(test_move_h_r))
        print("MOVE_HASH_C_L = {}".format(test_move_h_l))
    ts.test_close(move_cmp,
                  "Please check the handling of the edge cases in move! "
                  "Have you considered movement in both directions?")
    test_result = test_result and move_cmp

    if not move_cmp:
        plot_vs(WORLD, 1, initial_d, resample_d)

    ts.suite_close(test_result)
    if __name__ == "test_particlefilter":
        assert test_result


if __name__ == "__main__":
    test_particlefilter()
