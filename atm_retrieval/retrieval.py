import numpy as np
import pathlib

from parameters import Parameters
from spectrum import DataSpectrum
from log_likelihood import LogLikelihood

class Retrieval:
    
    def __init__(self, params, d_spec, pRT_model):
        self.params = params
        self.d_spec = d_spec
        self.pRT_model = pRT_model
        
        # Initialize the log-likelihood
        self.log_likelihood = LogLikelihood(d_spec, params.n_params)
        
        # Initialize the pRT model
        self.pRT_model.setup(self.params.params)
        
        
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
    file_data = '../data/crires_example_spectrum.dat'
    d_spec = DataSpectrum(file_target=file_data, slit='w_0.4', flux_units='erg/s/cm2/cm')
    
    ## Load pRT model
    
    
    ret = Retrieval(parameters, d_spec, pRT_model)