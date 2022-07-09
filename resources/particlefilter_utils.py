# -*- coding: utf-8 -*-
"""
Datatypes and helper functions used to implement a simple particle filter
"""
from __future__ import print_function

from six.moves.queue import Queue

import numpy as np
import matplotlib.pyplot as plt


class RandomNumberSafe(object):
    """The container class for saving sets of once random generated numbers for
    the purpose of reaching a comparable result when using the particle filter.
    This way the result will be the same. Otherwise the result may differ in a
    way that a comparison is impossible while the result may be valid.

    Parameter
    ----------
    init : list of int
        The precomputed random positions.
    move : list of list of float
        The precomputed noisy movements
    resample_idx : list of int
        The precomputed list of indices.

    Attributes
    ----------
    init_list : list of int
        The list of uniform distributed positions.
    move_queue : Queue of List of float
        The queue which contains the noisy movements for all particles for all
        iterations.
    resample_idx_queue : Queue of int
        The randomly chosen start index values for the resampling process for
        each iteration.

    Examples
    -----------
    >>> RND_INIT_LIST = [622, 631, 172, 316]
    >>> RND_MOVE = [[5.0, 6.0, 6.0, 5.0], [6.0, 5.0, 6.0, 6.0]]
    >>> RND_RESAMPLE_IDX = [80, 12]
    >>> r = RandomNumberSafe(RND_INIT_LIST, RND_MOVE, RND_RESAMPLE_IDX)
    >>> print(r)
    RND_INIT_LIST = [622, 631, 172, 316]
    RND_MOVE = [
    [5.0, 6.0, 6.0, 5.0],
    [6.0, 5.0, 6.0, 6.0]]
    RND_RESAMPLE_IDX = [80, 12]
    """
    def __init__(self, init=None, move=None, resample_idx=None):
        self.init_list = []
        self.move_queue = Queue()
        self.resample_idx_queue = Queue()
        if init is not None:
            self.init_list = init
        if move is not None:
            for row in move:
                self.move_queue.put(row)
        if resample_idx is not None:
            for index in resample_idx:
                self.resample_idx_queue.put(index)

    @classmethod
    def from_npz(cls, filename):
        """Load the relevant data from an .npz file"""
        data = np.load(filename)
        return cls(data["RND_INIT_LIST"], data["RND_MOVE"],
                   data["RND_RESAMPLE_IDX"])

    def get_init_list(self):
        """
        Returns
        --------
        list of float
            The list with initial particle positions.

        Examples
        --------
        >>> RND_INIT_LIST = [622, 631, 172, 316]
        >>> RND_MOVE = [[5.0, 6.0, 6.0, 5.0], [6.0, 5.0, 6.0, 6.0]]
        >>> RND_RESAMPLE_IDX = [80, 12]
        >>> r = RandomNumberSafe(RND_INIT_LIST, RND_MOVE, RND_RESAMPLE_IDX)
        >>> print(r)
        RND_INIT_LIST = [622, 631, 172, 316]
        RND_MOVE = [
        [5.0, 6.0, 6.0, 5.0],
        [6.0, 5.0, 6.0, 6.0]]
        RND_RESAMPLE_IDX = [80, 12]
        >>> print(r.get_init_list())
        [622, 631, 172, 316]
        """
        return self.init_list

    def get_noise_samples(self):
        """
        Returns
        --------
        list of float
            list of random noise values for each individual particle.

        Examples
        --------
        >>> RND_INIT_LIST = [622, 631, 172, 316]
        >>> RND_MOVE = [[5.0, 6.0, 6.0, 5.0], [6.0, 5.0, 6.0, 6.0]]
        >>> RND_RESAMPLE_IDX = [80, 12]
        >>> r = RandomNumberSafe(RND_INIT_LIST, RND_MOVE, RND_RESAMPLE_IDX)
        >>> print(r)
        RND_INIT_LIST = [622, 631, 172, 316]
        RND_MOVE = [
        [5.0, 6.0, 6.0, 5.0],
        [6.0, 5.0, 6.0, 6.0]]
        RND_RESAMPLE_IDX = [80, 12]
        >>> print(r.get_noise_samples())
        [5.0, 6.0, 6.0, 5.0]
        """
        return self.move_queue.get(block=False)

    def get_next_resample_idx(self):
        """
        Returns
        --------
        int
            The index at which the resampling (wheel) starts.

        Examples
        --------
        >>> RND_INIT_LIST = [622, 631, 172, 316]
        >>> RND_MOVE = [[5.0, 6.0, 6.0, 5.0], [6.0, 5.0, 6.0, 6.0]]
        >>> RND_RESAMPLE_IDX = [80, 12]
        >>> r = RandomNumberSafe(RND_INIT_LIST, RND_MOVE, RND_RESAMPLE_IDX)
        >>> print(r)
        RND_INIT_LIST = [622, 631, 172, 316]
        RND_MOVE = [
        [5.0, 6.0, 6.0, 5.0],
        [6.0, 5.0, 6.0, 6.0]]
        RND_RESAMPLE_IDX = [80, 12]
        >>> print(r.get_next_resample_idx())
        80
        """
        return self.resample_idx_queue.get(block=False)

    def __str__(self):
        """
        Returns
        --------
        str
            The string representation for collecting the precomputed values
            from one test run.
        """
        this_str = "\nRND_INIT_LIST = "
        this_str += "[" + ", ".join(str(x) for x in self.init_list) + "]\n"
        this_str += "\nRND_MOVE = "
        this_str += "["
        for idx, row in enumerate(self.move_queue.queue):
            if idx == 0:
                this_str += "\n[" + ", ".join(str(x) for x in np.around(row, 3)) + "]"
            else:
                this_str += ",\n[" + ", ".join(str(x) for x in np.around(row, 3)) + "]"
        this_str += "]\n"
        this_str += "\nRND_RESAMPLE_IDX = "
        this_str += "["
        this_str += ", ".join(str(x) for x in self.resample_idx_queue.queue)
        this_str += "]\n"
        return this_str

    def __repr__(self):
        return str(self)


class Example(object):
    """
    Container for an example

    Parameters
    ----------
    img_name : str
        Name of the image which represents the map.
    start_pos : int
        The real position of the robot.
    particle_num : int
        The number of particles which are used.
    result_hash : int
        Hashwert of the expected result.
    rnd_nmr_safe : RandomNumberSafe
        Container for fixed precomputed random numbers.

    Attributes
    ----------
    img_name : str
        Name of the image which represents the map.
    start_pos : int
        The real position of the robot.
    particle_num : int
        The number of particles which are used.
    result : int
        Hashwert of the expected result.
    rnd_safe : RandomNumberSafe
        Container for fixed precomputed random numbers.
    """
    def __init__(self, img_name, start_pos, particle_num, result_hash,
                 rnd_nmr_safe, movement, n_iterations=100):
        self.img_name = img_name
        self.start_pos = start_pos
        self.particle_num = particle_num
        self.result = result_hash
        self.rnd_safe = rnd_nmr_safe
        self.movement = movement
        self.n_iterations = n_iterations


def gaussian(x, mu, sig):
    """
    Simple gaussian function

    Parameter
    ---------
    x : float
        The parameter for which we compute the gaussian.
    mu : float
        The center of the distribution.
    sig : float
        The standard deviation sigma.

    Returns
    --------
    float
        The value of the gaussian function at x.

    Examples
    --------
    >>> mu=0
    >>> sig=1
    >>> x=0
    >>> print(gaussian(x, mu, sig))
    0.398942280401
    >>> x=1
    >>> print(gaussian(x, mu, sig))
    0.241970724519
    >>> x=-1
    >>> print(gaussian(x, mu, sig))
    0.241970724519
    """
    return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


class Particle(object):
    """
    Container class for particles.

    Parameters
    ----------
    x_pos : int
        The initial position of the robot.
    weight : float
        The initial weight of the particle.

    Attributes
    -----------
    x : int
        The position of the particle.
    w : float
        The weight of the particle.
    """

    def __init__(self, x_pos, weight):
        self.x = x_pos
        self.w = weight

    def __str__(self):
        return "x: {}, w: {}".format(self.x, self.w)

    def __repr__(self):
        return "Particle(x={}, w={})".format(self.x, self.w)

    def deep_copy(self):
        """
        Returns
        -------
        Particle
            An actual copy of the particle and not a reference.
        """
        return Particle(self.x, self.w)


def plot_img(img):
    """Simple wrapper around matplotlib.pyplot.imshow"""
    plt.imshow(img)
    plt.axis('off')


def plot(img, img_res, particles, current_pos=None):
    """
    Parameters
    ----------
    img : np.ndarray
        Picture of the environment.
    img_res : float  (0, 1]
        Resolution of the image in relation to the particle filter.
    particles : np.array of particles
        The current set of localization particles.
    current_pos : int
        The position of the (simulated) robot.
    """

    plt.clf()
    upper = plt.subplot(2, 1, 1)
    if current_pos is not None:
        upper.bar(current_pos, 260, 1, color="black",
                  label="current robot position",
                  edgecolor="none", linewidth=2)
        upper.legend()
    upper.imshow(img)
    upper.axis('off')
    resolution = max(min(1.0, img_res), 0.01)
    lower_xlim_max = resolution * img.shape[1]
    lower = plt.subplot(2, 1, 2)
    lower.bar([p.x for p in particles], [p.w for p in particles], 1,
              color="red", edgecolor="none")
    #    if current_pos is not None:
    #        lower.bar(current_pos, 1, 1, color="red")
    lower.set_xlim(0, lower_xlim_max)
    lower.set_ylim(0.0, 0.05)
    plt.pause(0.02)


def plot_vs(img, img_res, old_p, new_p, current_pos=None):
    """
    Parameters
    ----------
    img : np.ndarray
        Picture of the environment.
    img_res :float  (0, 1]
        Resolution of the image in relation to the particle filter.
    old_p : List of Particles
        The particle distribution before the new movement is accounted for.
    new_p : List of Particles
        The particle distribution after one step in the particle filter and
        with processed movement.
    current_pos : int
        The position of the (simulated) robot.
    """
    # some computations
    resolution = max(min(1.0, img_res), 0.01)
    xlim_max = np.rint(resolution * img.shape[1])
    ylim_max = 0.05
    bins = np.linspace(0, xlim_max, int(xlim_max/5))

    hist_value, bins = np.histogram([p.x for p in new_p], bins=bins)
    hist_value = hist_value.astype(float) / float(hist_value.sum())
    bin_pos = 0.5*(bins[1:] + bins[:-1])
    bin_width = 1.0 * (bins[1] - bins[0])
    # plot
    fig = plt.gcf()
    if not fig.get_axes():
        upper = fig.add_subplot(211)
        lower = fig.add_subplot(212, sharex=upper)
    else:
        upper, lower = fig.get_axes()

    if not upper.images:
        upper.set_title("map")
        upper.imshow(img)
        upper.axis('off')
    if current_pos is not None:
        if upper.lines:
            upper.lines[0].set_xdata(current_pos)
        else:
            upper.axvline(current_pos,# ymin=0, ymax=img.shape[0],
                          color="black", label="current robot position",
                          linewidth=3)
            upper.legend()

    resolution = max(min(1.0, img_res), 0.01)
    xlim_max = resolution * img.shape[1]
    lower.set_title("old vs new particle")
    # matplotlib does not store bar plots as line-objects, but as a collection
    # of rectangular patches instead. In order to update them, one has to
    # iterate over them and update their positions and heights manually.
    if lower.patches:
        # update histogram patches they are stationary so just update
        # their height
        for i, height in enumerate(hist_value):
            lower.patches[i].set_height(height)
        # update old particle positions. As they move, position and height
        # (showing weight) must be updated
        for i, particle in enumerate(old_p, i+1):
            lower.patches[i].set_x(particle.x)
            lower.patches[i].set_height(particle.w)
        # update new particle positions using the same procedure as for the
        # old particles
        for i, particle in enumerate(new_p, i+1):
            lower.patches[i].set_x(particle.x)
            lower.patches[i].set_height(particle.w)
    elif any(old_p) and any(new_p):
        lower.bar(bin_pos, hist_value, alpha=0.3,
                  label='new particle histogram', width=bin_width)
        lower.bar([p.x for p in old_p], [p.w for p in old_p], 1,
                  color="black", edgecolor="none", alpha=0.5, label='old particle')
        lower.bar([p.x for p in new_p], [p.w for p in new_p], 1,
                  color="red", edgecolor="none", label='new particle')
        lower.set_xlim(0, xlim_max)
        lower.set_ylim(0.0, ylim_max)
        lower.set_ylabel('p(S)')
        lower.set_xlabel('position (circular world)')
        lower.legend()
        plt.tight_layout()
    plt.pause(0.01)


def test_plot_vs(img, img_res, old_p, new_p, current_pos=None):
    """
    Parameters
    ----------
    img : np.ndarray
        Picture of the environment.
    img_res :float  (0, 1]
        Resolution of the image in relation to the particle filter.
    old_p : List of Particles
        The particle distribution before the new movement is accounted for.
    new_p : List of Particles
        The particle distribution after one step in the particle filter and
        with processed movement.
    current_pos : int
        The position of the (simulated) robot.
    """
    # some computations
    resolution = max(min(1.0, img_res), 0.01)
    xlim_max = resolution * img.shape[1]

    # plot
    plt.clf()
    upper = plt.subplot(2, 1, 1)
    upper.imshow(img)
    upper.axis('off')
    if current_pos is not None:
        upper.bar(current_pos, img.shape[0], 1, color="black",
                  label="current robot position",
                  edgecolor="none", linewidth=3)
        upper.legend()
    lower = plt.subplot(2, 1, 2, sharex=upper)
    lower.set_title("old vs new particle")
    lower.bar([p.x for p in old_p], [p.w for p in old_p], 1,
              color="black", edgecolor="none", alpha=0.3, label='old particle')
    lower.bar([p.x for p in new_p], [p.w for p in new_p], 1,
              color="red", edgecolor="none", label='new particle')
    lower.set_xlim(0, xlim_max)
    lower.set_ylim(0.0, 1)
    lower.set_ylabel('p(S)')
    lower.set_xlabel('Position (circular world)')
    lower.legend()
    plt.tight_layout()
    plt.pause(0.02)


def plot_all(img, img_res, current, move, weight, current_pos=None):
    """
    Parameters
    ----------
    img : np.ndarray
        Picture of the environment
    img_res : float  (0, 1]
        Resolution of the image in relation to the particle filter
    current_pos : int
        The position of the (simulated) robot.
    """

    plt.clf()
    upper = plt.subplot(4, 1, 1)
    if current_pos is not None:
        upper.bar(current_pos, 260, width=2, color="black",
                  label="current robot position",
                  edgecolor="none")
        upper.legend()
    upper.imshow(img)
    upper.axis('off')
    resolution = max(min(1.0, img_res), 0.01)
    xlim_max = resolution * img.shape[1]

    lower1 = plt.subplot(4, 1, 2, sharex=upper)
    lower1.set_title("Initial Distribution/ Last Distribution")
    lower1.bar([p.x for p in current], [p.weight for p in current], 1,
               color="red", edgecolor="none")
    lower1.set_xlim(0, xlim_max)
    lower1.set_ylim(0.0, 0.05)

    lower2 = plt.subplot(4, 1, 3, sharex=upper)
    lower2.set_title("Distribution after applying the current movement")
    lower2.bar([p.x for p in current], [p.weight for p in current], 1,
               color="black", edgecolor="none", alpha=0.1)
    lower2.bar([p.x for p in move], [p.weight for p in move], 1,
               color="red", edgecolor="none")
    lower2.set_xlim(0, xlim_max)
    lower2.set_ylim(0.0, 0.05)

    lower3 = plt.subplot(4, 1, 4, sharex=upper)
    lower3.set_title("Distribution after weighting")
    lower3.bar([p.x for p in weight], [p.weight for p in weight], 1,
               color="red", edgecolor="none")
    lower3.set_xlim(0, xlim_max)
    lower3.set_ylim(0.0, 1)
    plt.tight_layout()
    plt.pause(0.02)
