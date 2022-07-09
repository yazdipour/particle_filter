# -*- coding: utf-8 -*-
"""
.. codeauthor:: Alexander Vorndran <alexander.vorndran@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke.fischedick@tu-ilmenau.de>

###############################################################################
############################ NO IMPLEMENTATION HERE ###########################
###############################################################################
"""
from __future__ import print_function

import os
import six

try:
    from imageio import imread
    HAS_IMREAD = True
except ImportError:
    HAS_IMREAD = False
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=wrong-import-order
from ksr_common.string_suite import ValidationSuite
from ksr_common.hash_function import numpy_hash_normalized

from resources.example1 import EXAMPLE as example1
from resources.example2 import EXAMPLE as example2

try:
    from dev_particlefilter import ParticleFilter
    from dev_simulation import Simulator
except ImportError:
    from student_particlefilter import ParticleFilter
    from student_simulation import Simulator

DEV_MODE = False


def test_simulation(silent=False):
    """Test the particle filter in two scenarios"""
    examples = [example1, example2]
    results = []

    for idx, example in enumerate(examples):
        if not silent:
            plt.clf()
        suite = ValidationSuite("ParticleFilter Example {}".format(idx+1))

        basepath = os.path.dirname(__file__)
        try:
            img = np.load(os.path.join(basepath, example.img_name)+".npy")
        except IOError as err:
            if HAS_IMREAD:
                img = imread(os.path.join(basepath, example.img_name))
                np.save(os.path.join(basepath, example.img_name)+".npy", img)
            else:
                raise err

        start_pos = example.start_pos
        pf = ParticleFilter(img.shape[1], example.rnd_safe)
        sim = Simulator(img, start_pos, pf)
        sim.process([example.movement for _ in six.moves.range(example.n_iterations)],
                    particle_num=example.particle_num, silent=silent)

        suite.validation_open("Particle weight and position")
        if any(sim.pf.particles):
            particle_hash = numpy_hash_normalized(
                [[p.x, p.w] for p in sim.pf.particles]
            )
            result_cmp = particle_hash == example.result
            if DEV_MODE:
                print(idx, particle_hash)
            suite.validation_close(result_cmp)
        else:
            result_cmp = False

        suite.suite_close(result_cmp)
        results.append(result_cmp)

    if __name__ == "test_simulation":
        assert len(results) == len(examples), \
            "The test failed to collect the result of at least one example input"
        assert all(results), \
            "At least one example could not be validated successfully"


if __name__ == "__main__":
    test_simulation(silent=False)
