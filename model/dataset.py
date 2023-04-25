import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
from pyDOE2 import lhs
from sobol_seq import i4_sobol_generate
import ghalton

class pinn_collect_dataset():
    '''
    A Dataset that generates training data for PINN
    '''
    def __init__(self, num_collect=None, geom=None, time_dim=None,
                 space_distribution='uniform', time_distribution='uniform', given_data=False,
                 adaptive=None, **kwargs):

        '''
        :param num_collect: int or a list of int, number(s) of collection data
            - If the distribution is 'uniform', then it requires a list of int to describe the number of data in each dimension
            - If the distribution is a random one, then it requires an int to describe the number in the whole space.
            - If the :param adaptive is not None, then it describes the initialized number of data
        :param geom: a list of tuples, where each tuple represents the (min, max) value of a dimension in an order of (x, y, z, (t))
            -If time_dim is True, then the last dimension is the time dimension
        :param time_dim: bool, if there's a time domain
        :param space_distribution: data distribution method in the spacial dimension(s)
        :param time_distribution: data distribution method in the time dimension
        :param given_data: bool, use designated data from outside
            If it's True, it will ignore these:
                :param num_collect, :param num_bc, :param num_ic, :param space_distribution, :param time_distribution
            and also requires additional parameters:
                :param kwargs['collect_data']: a ndarray of collection data points in [x, y, z, (t)] format
                :param kwargs['boundary_data']: a ndarray or a list of ndarrays of boundary data points in
                    [x, (y), (z), (t)] format
                :param kwargs['initial_data']: a ndarray of initial data points of initial data points in
                    [x, (y), (z), t0] format, only required if :param time_dim is True
        :param adaptive: None or str, use adaptive data collection method. All the prepared data or given data will be
            the initialized data
        '''


        self.num_collect = num_collect
        self.geom = geom
        self.time_dim = time_dim
        self.space_distribution = space_distribution
        self.time_distribution = time_distribution
        self.given_data = given_data
        self.adaptive = adaptive

        self.collect_data = None

        if self.given_data is True:
            self.collect_data = kwargs['collect_data']
            assert self.collect_data.shape[1] == len(self.geom)


    def prepare_collection_data(self):
        if self.given_data is True:
            return self.collect_data #directly
        else:
            if self.space_distribution == 'uniform':
                assert isinstance(self.num_collect, list), "If the space_distribution is 'uniform', a list of int is required"
                assert len(self.num_collect) == len(self.geom), "The length of the list should be equal to the number of spatial dimensions"

                grid = np.meshgrid(
                    *[np.linspace(start, end, num) for (start, end), num in zip(self.geom, self.num_collect)])
                collection_data = np.stack([g.ravel() for g in grid], axis=-1)
                self.collect_data = torch.tensor(collection_data, dtype=torch.float32)

            elif self.space_distribution in ['random', 'LHS', 'Halton', 'Hammersley', 'Sobol']:
                assert isinstance(self.num_collect,
                                  int), "For random, LHS, Halton, Hammersley, and Sobol distributions, an int is required"

                if self.space_distribution == 'random':
                    collection_data = np.random.uniform(0, 1, (self.num_collect, len(self.geom)))

                elif self.space_distribution == 'LHS':
                    collection_data = lhs(len(self.geom), self.num_collect)

                elif self.space_distribution == 'Halton':
                    sequencer = ghalton.Halton(len(self.geom))
                    collection_data = np.array(sequencer.get(self.num_collect))

                elif self.space_distribution == 'Hammersley':
                    def van_der_corput(n, base):
                        vdc, denom = 0, 1
                        while n:
                            denom *= base
                            n, remainder = divmod(n, base)
                            vdc += remainder / denom
                        return vdc

                    d = len(self.geom)
                    collection_data = np.empty((self.num_collect, d))
                    collection_data[:, 0] = np.arange(self.num_collect) / self.num_collect
                    for j in range(1, d):
                        collection_data[:, j] = np.array([van_der_corput(i, j + 1) for i in range(self.num_collect)])

                elif self.space_distribution == 'Sobol':
                    collection_data = i4_sobol_generate(len(self.geom), self.num_collect)

                # Scale the collected data based on the space_dim
                for i, (start, end) in enumerate(self.geom):
                    collection_data[:, i] = collection_data[:, i] * (end - start) + start

                self.collect_data = torch.tensor(collection_data, dtype=torch.float32)

            else:
                raise NotImplementedError(
                    "The space_distribution '{}' is not implemented.".format(self.space_distribution))

            return self.collect_data

