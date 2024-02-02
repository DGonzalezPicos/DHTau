import os
os.environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import pathlib

from parameters import Parameters
from spectrum import DataSpectrum
from log_likelihood import LogLikelihood
from pRT_model import pRT_model



class Retrieval:
    
    
    # Pymultinest default parameters
    n_live_points = 100
    evidence_tolerance = 0.5
    n_iter_before_update = 500
    
    def __init__(self, parameters, d_spec, pRT_model):
        self.parameters = parameters
        self.d_spec = d_spec
        self.pRT_model = pRT_model
        
        # Initialize the pRT model
        # self.pRT_model.setup(self.parameters.params)
        
        # Initialize the log-likelihood
        self.loglike = LogLikelihood(d_spec, self.parameters.n_params, scale_flux=True)
        
    def PMN_lnL_func(self, cube=None, ndim=None, nparams=None):
        
        # Transform the cube to the parameter space
        sample = self.parameters(cube) # the attribute self.parameters.params is updated
        # self.parameters.add_sample(sample)
        
        m_spec = self.pRT_model(self.parameters.params)
        lnL = self.loglike(m_spec)
        return lnL
    
    def PMN_run(self):
        # Pause the process to not overload memory on start-up
        time.sleep(1.5*rank)

        # Run the MultiNest retrieval
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.parameters, 
            n_dims=self.parameters.n_params, 
            outputfiles_basename='pmn_', 
            resume=True, 
            verbose=True, 
            const_efficiency_mode=True, 
            sampling_efficiency=0.05, 
            n_live_points=self.n_live_points, 
            evidence_tolerance=self.evidence_tolerance, 
            n_iter_before_update=self.n_iter_before_update, 
            )
        
        
if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    ## Define parameters
    free_params = {
        # general properties
        # 'Rp'    : ([0.5, 2.0], r'$R_p$ [R$_J$]'),
        # 'log_g' : ([3.0, 5.5], r'$\log(g)$ [cm/s$^2$]'),
        'vsini' : ([1.0, 20.0], r'$v \sin(i)$ [km/s]'),
        'RV'    : ([-30.0, 30.0], r'RV [km/s]'),
        
        # chemistry
        'log12CO' : ([-12, -2], r'$\log$(CO)'),
        'logH2O'  : ([-12, -2], r'$\log$(H$_2$O)'),
        'logNa'   : ([-12, -2], r'$\log$(Na)'),
        
        # temperature profile
        'T1' : ([2000, 20000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
        'T2' : ([1000, 20000], r'$T_2$ [K]'),
        'T3' : ([300,  10000], r'$T_3$ [K]'),
        'T4' : ([300,  5000],  r'$T_4$ [K]'),
    }

    constant_params = {
        'Rp'    : 1.0,
        'log_g' : 4.0,
    }
    parameters = Parameters(free_params, constant_params)
    
    ## Load data
    file_data = 'data/DHTauA.dat'
    d_spec = DataSpectrum(file_target=file_data, slit='w_0.4', flux_units='erg/s/cm2/cm')
    d_spec.preprocess(
        file_transm='data/DHTauA_molecfit_transm.dat',
        tell_threshold=0.4,
        n_edge_pixels=30,
        sigma_clip=3,
        sigma_clip_window=11,
        )
    # output shape of wave, flux, err = (n_orders, n_dets, n_wave) = (7, 3, 2048)
    
    ## Load pRT model
    pRT_model = pRT_model().pickle_load('data/testing_atm.pickle')
    
    ret = Retrieval(parameters, d_spec, pRT_model)