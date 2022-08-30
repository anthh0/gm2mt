import numpy as np
import scipy.stats
import math
import pathlib

import gm2mt.auxiliary as aux

class Distribution:
    def __init__(self):
        pass

class _NoneDist(Distribution):
    def __init__(self):
        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}
    def select(self, number, rng):
        raise SyntaxError("This should not be happening!")
    def params_txt(self):
        return "None"
    def params_xl(self):
        return ["nonedist", 0, 0]

class Gaussian(Distribution):
    def __init__(
        self,
        mean,
        std
    ):
        if isinstance(mean, (int, float)):
            self.mean = [mean]
        elif isinstance(mean, (list, np.ndarray)):
            self.mean = list(mean)
        if isinstance(std, (int, float)):
            self.std = [std]
        elif isinstance(std, (list, np.ndarray)):
            self.std = list(std)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.mean) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["mean=" + str(i) for i in self.mean]
        if len(self.std) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["std=" + str(i) for i in self.std]

        self._MPID['loops'] = len(self.mean) * len(self.std)
    
    def select(self, number, rng):
        number = int(number)
        selection = rng.normal(loc = self.mean, scale = self.std, size = number)
        return selection
    
    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.normal(loc = self.mean, scale = self.std, size = (number, loops)).transpose()
        return selection

    
    def params_txt(self):
        return f"Gaussian distribution (mean {self.mean}, std {self.std})"
    
    def params_xl(self):
        return ["gaussian", self.mean, self.std]

class Uniform(Distribution):
    def __init__(
        self,
        min,
        max
    ):
        if isinstance(min, (int, float)):
            self.min = [min]
        elif isinstance(min, (list, np.ndarray)):
            self.min = list(min)
        if isinstance(max, (int, float)):
            self.max = [max]
        elif isinstance(max, (list, np.ndarray)):
            self.max = list(max)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.min) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["min=" + str(i) for i in self.min]
        if len(self.max) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["max=" + str(i) for i in self.max]

        self._MPID['loops'] = len(self.min) * len(self.max)

    def select(self, number, rng):
        number = int(number)
        selection = rng.uniform(low = self.min, high = self.max, size = number)
        return selection

    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.uniform(low = self.min, high = self.max, size = (number, loops)).transpose()
        return selection
    
    def params_txt(self):
        return f"uniform distribution ({self.min}-{self.max})"

    def params_xl(self):
        return ["uniform", self.min, self.max]

class Single(Distribution):
    def __init__(
        self,
        value
    ):
        if isinstance(value, (int, float)):
            self.value = [value]
        elif isinstance(value, (list, np.ndarray)):
            self.value = list(value)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.value) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["value=" + str(i) for i in self.value]

        self._MPID['loops'] = len(self.value)
    
    def select(self, number, rng):
        number = int(number)
        return np.full(shape = number, fill_value = self.value)
    
    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.uniform(low = self.value, high = self.value, size = (number, loops)).transpose()
        return selection
    
    def params_txt(self):
        return self.value
    
    def params_xl(self):
        return ["single", self.value, 0]

class Triangle(Distribution):
    def __init__(
        self,
        min,
        max
    ):
        if isinstance(min, (int, float)):
            self.min = [min]
        elif isinstance(min, (list, np.ndarray)):
            self.min = list(min)

        if isinstance(max, (int, float)):
            self.max = [max]
        elif isinstance(max, (list, np.ndarray)):
            self.max = list(max)
        
        self.mode = [(i+j)/2 for i in self.min for j in self.max]       #this might break if both min and max are multivalued
        
        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.min) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["value=" + str(i) for i in self.min]
        
        if len(self.max) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["value=" + str(i) for i in self.max]

        self._MPID['loops'] = len(self.min)*len(self.max)

    def select(self, number, rng):
        number = int(number)
        selection = rng.triangular(left = self.min, mode = self.mode, right = self.max, size = number)
        return selection

    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.triangular(left = self.min, mode = self.mode, right = self.max, size = (number, loops)).transpose()
        return selection
    
    def params_txt(self):
        return f"triangle distribution ({self.min}-{self.max})"

    def params_xl(self):
        return ['triangle', self.min, self.max]

class Custom(Distribution):
    def __init__(
        self,
        dir,
        zero
    ):
        if isinstance(dir, str):
            self.dir = [dir]
        elif isinstance(dir, (np.ndarray, list)):
            self.dir = list(dir)
        if isinstance(zero, (int, float)):
            self.zero = [zero]
        elif isinstance(zero, (np.ndarray, list)):
            self.zero = list(zero)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.dir) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["dir=" + str(i) for i in self.dir]
        if len(self.zero) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["zero=" + str(i) for i in self.zero]

        self._MPID['loops'] = len(self.dir) * len(self.zero)
        
        dir_list = np.broadcast_to(dir, self._MPID['loops'])
        zero_list = np.broadcast_to(zero, self._MPID['loops'])
        self.pairs = list(range(self._MPID['loops']))
        self.distributions = list(range(self._MPID['loops']))
        for idx, (dir, zero) in enumerate(zip(dir_list, zero_list)):
            source = str(pathlib.Path(__file__).parent.absolute()) + "/injection_profiles" + dir + ".dat"
            bin_centers, bin_heights = np.loadtxt(fname = source, skiprows = 5, usecols = (0, 1), unpack = True)
            bin_heights_normed = Custom._normalize(bin_heights[bin_heights >= 0])
            bin_centers = bin_centers[bin_heights >= 0] # cut the bin centers array as well
    
            if zero == "avg":
                avg = np.average(bin_centers, weights = bin_heights_normed)
                bin_centers -= avg
            elif zero == "max":
                bin_centers -= bin_centers[np.argmax(bin_heights_normed)]
            elif isinstance(zero, (int, float, np.int64, np.float64)):
                bin_centers -= zero
            else:
                raise ValueError("Your zero parameter is not recognized.")

            bin_edges = np.empty(shape = len(bin_centers) + 1)
            bin_edges[0] = bin_centers[0] - 0.5
            for i in range(1, len(bin_centers)):
                bin_edges[i] = (bin_centers[i - 1] + bin_centers[i]) / 2
            bin_edges[-1] = bin_centers[-1] + 0.5

            self.pairs[idx] = (bin_heights_normed, bin_edges)
            self.distributions[idx] = scipy.stats.rv_histogram(histogram = (bin_heights_normed, bin_edges))

    def select(self, number, rng):
        number = int(number)
        distribution = self.distributions[0]
        selection = distribution.rvs(size = number, random_state = rng)
        return selection
    
    def mselect(self, number, rng, loops):
        number = int(number)

        selection = np.zeros(shape = (loops, number))
        distributions = np.broadcast_to(self.distributions, shape = loops)
        for i in range(loops):
            selection[i] = distributions[i].rvs(size = number, random_state = rng)
        return selection

    def params_txt(self):
        return f"custom distribution ({self.dir}.dat, zero: {self.zero})"
    
    def params_xl(self):
        return ["custom", self.dir, self.zero]

    @staticmethod
    def _normalize(weights):
        return weights / weights.sum()

class Custom2D(Distribution):                           # UNDER CONSTRUCTION #
    def __init__(
        self,
        dir
    ):
        if isinstance(dir, str):
            self.dir = [dir]
        elif isinstance(dir, list):
            self.dir = dir
        else:
            raise TypeError("Your x_alpha_profile parameter was not recognized")

        self.bins = 100
        self.range = 10
        
        self._MPID = {'MPs': 0, 'loops': 0, 'labels': 0}
        if len(self.dir) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["dir=" + str(i) for i in self.dir]
        self._MPID['loops'] = len(self.dir)
        
        # FOR NOW ONLY CONSIDERING ONE DIR PARAMETER #
        source = str(pathlib.Path(__file__).parent.absolute()) + self.dir[0] + ".dat"
        
        x, px, pz = np.loadtxt(fname = source, skiprows = 1, usecols = (3, 4, 8), unpack = True)
        x = 1E3*x - aux.r_inj_offset        # convert (ring) offset from [m] to (inflector) offset [mm]
        Px = px*aux.p_magic                 # convert from phase space momentum to real momentum
        Pz = pz*aux.p_magic + aux.p_magic
        alpha = 1E3*np.arctan(Px / Pz)      # calculate inflector angle [mrad]

        x_hist1d_normed, x_edges_1 = np.histogram(x, bins = self.bins, range = [-self.range, self.range], density = True)
        x_alpha_hist2d, x_edges_2, a_edges_2 = np.histogram2d(x, alpha, bins = self.bins, range = [[-self.range, self.range], [-self.range, self.range]])

        self.x_dist = scipy.stats.rv_histogram(histogram = (x_hist1d_normed, x_edges_1))            # normed histogram for choosing x_offset
        self.a_dists = []
        for i in range(self.bins):
            a_hist1d_normed = Custom2D._normalize(x_alpha_hist2d[i])
            self.a_dists.append(scipy.stats.rv_histogram(histogram = (a_hist1d_normed, a_edges_2)))    # array of normed histograms for choosing alpha given an x selection
        
    def select(self, number, rng):
        number = int(number)

        x_selections = self.x_dist.rvs(size = number, random_state = rng)               # generate a 1d array of x values
        a_selections = np.zeros(shape = number)
        for idx, x_val in enumerate(x_selections):
            x_bin = math.floor((x_val + self.range)/(2*self.range)*self.bins)
            a_selections[idx] = self.a_dists[x_bin].rvs(size = 1, random_state = rng)   # for each x_val, generate an alpha value from the respective alpha dist
        
        x_selections += aux.r_inj_offset                                                # convert from inflector offset to ring offset
        return (x_selections, a_selections)

    def mselect(self, number, rng, loops):
        number = int(number)
        x_selections = self.x_dist.rvs(size = (loops, number), random_state = rng)
        a_selections = np.zeros(shape = (loops, number))
        for i, row in enumerate(x_selections):
            for j, x_val in enumerate(row):
                x_bin = math.floor((x_val + self.range)/(2*self.range)*self.bins)
                a_selections[i][j] = self.a_dists[x_bin].rvs(size = 1, random_state = rng)

        # print(f"x vals: {x_selections}, a vals: {a_selections}")
        x_selections += aux.r_inj_offset
        return (x_selections, a_selections)

    def params_txt(self):
        return f"custom 2d distribution ({self.dir}.dat)"

    def params_xl(self):
        return ["custom 2d", self.dir, 0]

    @staticmethod
    def _compare_arrays(a1, a2):            # not really needed anymore, was helpful during testing
        if len(a1) == len(a2):
            allTrue = True
            for idx in range(len(a1)):
                if a1[idx] != a2[idx]:
                    allTrue = False
            return allTrue
        else:
            return False

    @staticmethod
    def _normalize(weights):
        if weights.sum != 0:
            return weights / weights.sum()
        else:
            return weights



