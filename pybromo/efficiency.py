#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Efficiency module to compute FRET efficiency and max emission rate
# based on 1-d Langevin simulated dye-dye distance trajectories
#

"""
This module contains the core classes and functions to perform the
 efficiency and max emission rates.
"""

import os
import hashlib
import itertools
from pathlib import Path
from time import ctime
import json

import numpy as np
from numpy import array, sqrt
import tables

from .storage import FRETStore, ExistingArrayError
from .iter_chunks import iter_chunksize, iter_chunk_index

def em_rates_from_E_DA(em_rate_tot, E_values):
    """Donor and Acceptor emission rates from total emission rate and E (FRET).
    """
    #E_values = np.asarray(E_values)
    em_rates_a = E_values * em_rate_tot
    em_rates_d = em_rate_tot - em_rates_a
    return em_rates_d, em_rates_a


def em_rates_from_E_unique(em_rate_tot, E_values):
    """Array of unique emission rates for given total emission and E (FRET).
    """
    em_rates_d, em_rates_a = em_rates_from_E_DA(em_rate_tot, E_values)
    return np.unique(np.hstack([em_rates_d, em_rates_a]))


def em_rates_from_E_DA_mix(em_rates_tot, E_values):
    """D and A emission rates for two populations.
    """
    em_rates_d, em_rates_a = [], []
    for em_rate_tot, E_value in zip(em_rates_tot, E_values):
        em_rate_di, em_rate_ai = em_rates_from_E_DA(em_rate_tot, E_value)
        em_rates_d.append(em_rate_di)
        em_rates_a.append(em_rate_ai)
    return em_rates_d, em_rates_a



class FRETEfficiency(object):
    """
    Class for calculating and storing FRET efficiencies and max emission rates
    """
    _PREFIX_FRET = "E"

    __DOCS_STORE_ARGS___ = """
            prefix (string): file-name prefix for the HDF5 file.
            path (string): a folder where simulation data is saved.
            chunksize (int): chunk size used for the on-disk arrays saved
                during the brownian motion simulation. Does not apply to
                the timestamps arrays (see :method:``).
            chunkslice ('times' or 'bytes'): if 'bytes' (default) the chunksize
                is taken as the size in bytes of the chunks. Else, if 'times'
                chunksize is the size of the last dimension. In this latter
                case 2-D or 3-D arrays have bigger chunks than 1-D arrays.
            overwrite (bool): if True, overwrite the file if already exists.
                All the previously stored data in that file will be lost.
        """[1:]
    
    def compact_name(self, hashsize=6, t_max=False):
        """Compact representation of simulation params (no ID, EID and t_max)
        """
        name = f"{self.E_method}_R0_{self.R0}"
        if hashsize > 0:
            name = self.S.hash()[:hashsize] + '_' + name
        return name
    
    def _open_store(self, store, prefix='', path='./', mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Low level method used to implement different stores.

        Arguments:
            store (one of storage.Store classes): the store class to use.
        """ + self.__DOCS_STORE_ARGS___ + """
        Returns:
            Store object.
        """
        store_fname = '%s_%s.hdf5' % (prefix, self.compact_name())
        attr_params = dict(E_method=self.E_method, R0=self.R0, max_rates=self.em_rates)
        kwargs = dict(path=path, #nparams=self.numeric_params,
                      attr_params=attr_params, mode=mode)
        store_obj = store(store_fname, **kwargs)
        return store_obj

    def open_store_efficiency(self, chunksize,chunkslice="times", path=None, mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Arguments:
        """ + self.__DOCS_STORE_ARGS___
        if hasattr(self, 'fret_store'):
            return
        if path is None:
            if hasattr(self, 'store'):
                # Use same folder of the trajectory file
                path = self.store.filepath.parent
            else:
                # No trajectory file, use current folder
                path = '.'
        self.fret_store = self._open_store(FRETStore,
                                         prefix=FRETEfficiency._PREFIX_FRET,
                                         path=path,
                                         mode=mode)
        self.fret_group = self.fret_store.h5file.root.fret
        #self.fret_stor.v_attrs['R0'] = self.R0
        #self.fret_stor.v_attrs['E_method'] = self.E_method
        name = f"E_{self.hash}_{self.E_method}_R0_{self.R0}_{self.em_rates}.hd5"
        kwargs = dict(num_particles = self.num_particles, chunksize=chunksize, chunkslice=chunkslice)
        #kwargs = dict(name=name, num_particles = self.num_particles, chunksize=chunksize, chunkslice=chunkslice)

        #self.fret_group = self.fret_store.h5file.root.fret
        self.efficiency, self.max_em_rate_d, self.max_em_rate_a = self.fret_store.add_efficiency(**kwargs)
        #Open max rate stores
        #self.em_rates_d = self.fret_group.max_rates_d
        #self.em_rates_a = self.fret_group.max_rates_a
        #self.max_em_rate_a = self.fret_store.add_max_rates_a(max_rates=self.max_rates,**kwargs)
        #self.max_em_rate_d = self.fret_store.add_max_rates_d(**kwargs)
        #self.max_em_rate_a = self.fret_store.add_max_rates_a(**kwargs)
    
    @staticmethod
    def empirical_efficiency(R,R0=56.):
        return 1/(1. + 0.975*(R/R0)**(2.65))
    
    @staticmethod
    def theoretical_efficiency(R,R0=56.):
        return 1/(1. + (R/R0)**6.)
    
    def _get_E(self,R):
        if self.E_method == "theoretical":
            E = self.theoretical_efficiency(R,self.R0)
        elif self.E_method == "emirical":
            E = self.empirical_efficiency(R,self.R0)
        return E

    def add_efficiency_max_rates(self,t_chunksize=None,chunkslice='bytes',timeslice=None,path='./',):
        """
        A function to add the the donor/acceptor emission rates based on simulated efficiencies
        em_rates: the maximum emission rates from the timestamp simulation
        populations: population slices for emission rates
        """
        if t_chunksize is None:
            t_chunksize = self.S.emission.chunkshape[1]
        timeslice_size = self.S.n_samples
        if timeslice is not None:
            timeslice_size = timeslice // self.S.t_step

        self.open_store_efficiency(chunksize=t_chunksize, chunkslice=chunkslice, path=path)
        EFF = self.efficiency
        EM_A = self.max_em_rate_a
        EM_D = self.max_em_rate_d
        #dd_arr = np.zeros((self.num_particles,t_chunksize),dtype=np.float32)
        for pop,em in zip(self.populations,self.em_rates) :      
            #Calculate the donor/acceptor max rates and store in hdf5
            for i_start, i_end in iter_chunk_index(timeslice_size, t_chunksize):
                    d_i = self.S.dye_distance[pop, i_start:i_end]
                    #dd_arr = d_i
                    E_i = self._get_E(R=d_i)
                    em_rates_d_i, em_rates_a_i = em_rates_from_E_DA(em,E_i)
                    EFF.append(E_i)
                    EM_D.append(em_rates_d_i)
                    EM_A.append(em_rates_a_i) 

    def __init__(self,S,populations,em_rates,timeslice=None,E_method="theoretical",R0=56.):
            """
            Initialize parameters for calculating efficiency.
            """
            if timeslice is None:
                timeslice = S.t_max
            assert len(populations) == len(em_rates)
            assert sum(populations) <= S.num_particles
            assert timeslice <= S.t_max
            self.em_rates = em_rates
            self.R0 = R0
            self.E_method = E_method
            self.populations = S.particles.num_particles_to_slices(populations)
            self.num_particles = S.num_particles
            self.hash = S.hash()[:6]
            self.S = S
    

