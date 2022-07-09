# -*- coding: utf-8 -*-
"""1D simulation of particle filter localization algorithm"""
from __future__ import print_function

import numpy as np

from resources.particlefilter_utils import plot_vs

from student_particlefilter import ParticleFilter


class Simulator(object):
    """Simulates a robot moving through a map making observations.

    Parameter
    ---------
    map_1di : np.ndarray
        The one dimensional map in which we act.
    start_pos : int
        The position in the map_1di
    pf : ParticleFilter
        The particle filter object.
    resolution : float
        The resolution of map_1di to the discrete 1D map we move in.

    Attributes
    ----------
    map_1di : np.ndarray
        The 2D image for our 1D map.
    start_pos : int
        The initial position of the simulated robot
    map_res : float
        The resolution of map_1di to the discrete 1D map we move in.
    current_pos : int
        The current position of the simulated robot
    pf : ParticleFilter
        The particle filter object.

    """

    def __init__(self, map_1di, start_pos, pf, resolution=1.0):
        self.map_1di = map_1di
        self.start_pos = start_pos
        self.map_res = max(min(1.0, resolution), 0.01)
        self.current_pos = start_pos
        self.pf = pf

    def process(self, move, particle_num, silent=False):
        """This function runs as many loops as there are elements in the move
        parameter and simulates for each movement the resulting observation in
        the map. These elements are then fed into the particle filter.

        Parameters
        ----------
        move : List of movements
            The movements which will be executed in the process. The number of
            elements in this list determines the number of iterations of the
            particle filter.
        particle_num : int
            The number of particles used.
        silent : bool
            Determines whether a continuous visualization is shown.
        """
        #   (i)     Initialize the particle filter with the correct number of
        #           particles.
        # TODO: add your code here
        self.pf.initialize(particle_num)

        for mov in move:
            self.current_pos = (self.current_pos + mov) % self.map_1di.shape[1]
            old = [p.deep_copy() for p in self.pf.particles]

            #   (ii)    Apply the movement (mov) to the particles in the particle
            #           filter.
            # TODO: add your code here
            self.pf.move(mov)

            #   (iii)   Weight the particles in the particle filter with the
            #           observation (obs).
            obs = self.map_1di[self.map_1di.shape[0] - 1, self.current_pos]
            # TODO: add your code here
            self.pf.weight(self.map_1di, obs)

            #   (iv)    Resample the particles in the particle filter.
            # TODO: add your code here
            self.pf.resample()

            # copying and other visualization stuff
            new = [p.deep_copy() for p in self.pf.particles]
            if not silent:
                plot_vs(self.map_1di, self.map_res, old, new, self.current_pos)
