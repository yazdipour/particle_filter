# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np

from resources.particlefilter_utils import (gaussian, RandomNumberSafe,
                                            Particle)


class ParticleFilter(object):
    """
    Parameter
    ---------
    max_idx : int
        The maximum x value possible
    rnd_numbers : RandomNumberSafe
        Container for a set of random numbers
    move_sigma : float
        The sigma value for the normal distributed noise while moving.

    Attributes
    ----------
    particles : List of Particle
        A List of all particle objects which are used.
    max_idx : int
        The last index in the position axis.
    move_sigma : float
        The deviation term for motion noise.
    rnd_safe : RandomNumberSafe
        Container for a set of random numbers
    """

    def __init__(self, max_idx, rnd_numbers=None, move_sigma=1):
        self.particles = []
        self.max_idx = max_idx
        self.move_sigma = move_sigma
        if rnd_numbers is not None:
            self.rnd_safe = rnd_numbers
        else:
            self.rnd_safe = RandomNumberSafe()

    def initialize(self, n):
        """Initializes n particles.

        Initializes the internal particle list with uniform distributed
        particles whose weight is also uniform/equal. The position is chosen
        randomly with an uniform distribution, whereas the weight
        (or probability) is determined by the number of particles. The sum of
        all weights should be 1.0.

        Parameter
        ---------
        n : int
            Number of particles.

        Examples
        --------
        The following example shows uniformly distributed positions and
        weights.
        Example only works if RandomNumberSafe is set or random numbers are
        generated instead!

        >>> max_idx = 20
        >>> pf = ParticleFilter(max_idx)
        >>> pf.initialize(5)
        >>> print([str(x.x) for x in pf.particles])  # position in [0, max_idx)
        ['13', '3', '18', '14', '3']
        >>> print([str(x.w) for x in pf.particles])  # probability [0, 1.0]
        ['0.2', '0.2', '0.2', '0.2', '0.2']
        >>> pf.initialize(10)
        >>> print([str(x.x) for x in pf.particles])  # position in [0, max_idx)
        ['19', '8', '16', '17', '10', '8', '4', '2', '14', '14']
        >>> print([str(x.w) for x in pf.particles])  # probability [0, 1.0]
        ['0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1']
        """
        random_positions = self.rnd_safe.get_init_list()

        #   (i)     Compute the initial, uniform weight for all particles.
        # #TODO: add your code here
        w = 1 / n
        #   (ii)    Initialize a list of particles with random positions
        #           and the initial weight.
        # TODO: add your code here
        self.particles = [Particle(x, w) for x in random_positions]

    def move(self, movement):
        """Applies the movement on all particles and adds noise.

        It is assumed that the movement has a noise component that we don't
        know about. It is also known that particles will stick close to a good
        pose hypothesis later on. We counter the unknown noise by adding
        additional noise to the movement of a particle to spread the particles
        further around a good hypothesis. This will ensure a high probability
        of having one or more particles that are right on (or close to) the
        mark and get a high weight in the next step.

        Parameter
        ---------
        movement : int
            movement relative to the current position, where a negative value
            means moving to the left
        """
        #   (iii)  Generate the noisy movement
        #          Hint: self.rnd_safe.get_noise_samples() provides you with
        #                (reproducible) random values to simulate the noise.
        # TODO: add your code here
        noisy_movement = self.rnd_safe.get_noise_samples()

        #   (iv)   Add the noisy movement to the particles
        #         Hint:   The world is circular, which e.g. means that the
        #                  robot will enter the world from the left side again
        #                  after he left it on the right side.
        # TODO: add your code here
        for i, p in enumerate(self.particles):
            p.x = (p.x + movement + noisy_movement[i]) % self.max_idx

    def weight(self, map_1di, observation, obs_prob=1.0):
        """Adjusts the importance weight of each particle

        The weight is computed according to the observation which the
        (simulated) robot made compared to the observation each particle
        predicts. We will be using a binary weight function meaning that we
        either see the observation or we don't see the observation.

        Parameter
        ---------
        map_1di : np.array
            The map which is actually a 2D image, but for this exercise we
            use the bottom line as 'map'.
        observation : [red green blue opacity]
            The color observation made by the (simulated) robot.
        obs_prob : float (0,1)
            The probability that obs_prob is really observed when obs_prob is
            the observation.

        Examples
        --------
        Example only works if RandomNumberSafe is set or random numbers are
        generated instead!

        >>> import numpy as np
        >>> o = [1, 1]
        >>> f = [0, 0]
        >>> my_map = np.array([f, f, o, o, o, f, f, f, o, o, o, f, f, f, f, o, o, o, f, f], ndmin=3)
        >>> print([[x[0] for x in row] for row in my_map])
        [[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]]
        >>> max_idx = 20
        >>> pf = ParticleFilter(max_idx)
        >>> pf.initialize(10)
        >>> print([x.x for x in pf.particles])  # pose or indices
        [3, 3, 5, 8, 18, 10, 19, 14, 13, 16]
        >>> print([ "{:1.3f}".format(x.w) for x in pf.particles])  # initial weight
        ['0.100', '0.100', '0.100', '0.100', '0.100', '0.100', '0.100', '0.100', '0.100', '0.100']
        >>> obs = o
        >>> pf.weight(my_map, obs)
        >>> print(["{:1.3f}".format(x.w) for x in pf.particles])  # new weight
        ['0.499', '0.499', '0.342', '0.499', '0.342', '0.499', '0.342', '0.342', '0.342', '0.499']
        """
        # For each particle ...
        for particle in self.particles:
            #   (v)     Extract the position of the particle.
            particle_pos = particle.x
            # TODO: add your code here

            # Get the observation the particle makes.
            p_observation = map_1di[map_1di.shape[0] - 1, int(particle_pos)]

            #   (vi)    Compare the observation of the robot and the
            #           observation of the particle
            #           Hint:   The observation is a list of four elements:
            #                   [red, green, blue, opacity]
            # TODO: add your code here
            compare = observation[0] == p_observation[0] and observation[1] == p_observation[
                1] and observation[2] == p_observation[2] and observation[3] == p_observation[3]

            # We either observe green, red or blue, and as such we have a
            # binary problem. We do not need to compute a specific weight.
            #   (vii)   If the observations are equal then use
            #           gaussian(0, 0, obs_prob) as the weight of the particle,
            #           else use gaussian(1, 0, obs_prob).
            #           Hint:   gaussian(x, mu, sig) is provided
            # TODO: add your code here
            if compare:
                particle.w = gaussian(0, 0, obs_prob)
            else:
                particle.w = gaussian(1, 0, obs_prob)

    def resample(self):
        """Resamples the particles according to their weight

        This is **not** roulette wheel selection, but stochastic universal
        sampling (SUS). SUS has better properties regarding bias and spread,
        and it also works better for populations where large differences
        between the importance weights exist.

        Examples
        --------
        Example only works if RandomNumberSafe is set or random numbers are
        generated instead!

        >>> import numpy as np
        >>> o = [1, 1]
        >>> f = [0, 0]
        >>> my_map = np.array([f, f, o, o, o, f, f, f, o, o, o, f, f, f, f, o, o, o, f, f], ndmin=3)
        >>> max_idx = 20
        >>> pf = ParticleFilter(max_idx)
        >>> pf.initialize(10)
        >>> obs = o
        >>> pf.weight(my_map, obs)
        >>> print([x.x for x in pf.particles])  # current position
        [16, 13, 18, 13, 3, 0, 12, 3, 1, 1]
        >>> pf.resample()
        >>> print([x.x for x in pf.particles])  # new position
        [13, 0, 3, 3, 3, 16, 16, 13, 13, 0]

        See also
        --------
        https://en.wikipedia.org/wiki/Stochastic_universal_sampling
        """
        cum_weights = np.cumsum([p.w for p in self.particles])
        total = 0.0 + cum_weights[-1]
        n_samples = len(self.particles)
        interval_width = total / n_samples

        start = self.rnd_safe.get_next_resample_idx()
        pointers = np.mod(start + interval_width*np.arange(n_samples), total)
        indices = np.searchsorted(cum_weights, pointers)
        self.particles = [self.particles[i].deep_copy() for i in indices]

        # normalizing (only for visualization)
        weight_sum = sum(p.w for p in self.particles)
        for p in self.particles:
            p.w /= weight_sum
