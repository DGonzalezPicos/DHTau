import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from parameters import Parameters
from spectrum import DataSpectrum, ModelSpectrum
from pRT_model import pRT_model
from log_likelihood import LogLikelihood


path = pathlib.Path('atm_retrieval/') # run file as --> python atm_retrieval/prior_predictive_check.py
# path = pathlib.Path('./') # use this if you run this script from the atm_retrieval/ folder

free_params = {
        # general properties
        # 'Rp'    : ([0.5, 2.0], r'$R_p$ [R$_J$]'),
        # 'log_g' : ([3.0, 5.5], r'$\log(g)$ [cm/s$^2$]'),
        'vsini' : ([1.0, 12.0], r'$v \sin(i)$ [km/s]'),
        # 'rv'    : ([-30.0, 30.0], r'RV [km/s]'),
        
        # chemistry
        'log12CO' : ([-8, -2], r'$\log$(CO)'),
        'logH2O'  : ([-8, -2], r'$\log$(H$_2$O)'),
        'logNa'   : ([-12, -3], r'$\log$(Na)'),
        
        # temperature profile
        'T1' : ([4000, 10000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
        'T2' : ([2000, 5000], r'$T_2$ [K]'),
        'T3' : ([1000,  3000], r'$T_3$ [K]'),
        'T4' : ([300,  2000],  r'$T_4$ [K]'),
    }

constant_params = {
    'R_p'    : 3.0, # [R_jup]
    'parallax' : 50., # [mas]
    'epsilon_limb' : 0.5, # 
    'log_g' : 4.0,
    'log_P_knots': [2, 1, -1, -5], # [log(bar)]
    'rv' : 30.0, # [km/s]
}
# Define parameters with given priors
parameters = Parameters(free_params, constant_params)

## Load CRIRES data
file_data = 'data/DHTauA.dat'
d_spec = DataSpectrum(file_target=file_data, slit='w_0.4', flux_units='photons')
# Load Telluric model (fitted to the data with Molecfit)
molecfit_spec = DataSpectrum(file_target='data/DHTauA_molecfit_transm.dat', slit='w_0.4', flux_units='')

# Divide by the molecfit spectrum 
throughput = molecfit_spec.err # read as the third column (fix name)  
transm = (molecfit_spec.flux * throughput)
zeros = transm <= 0.01
d_spec.flux = np.divide(d_spec.flux, transm, where=np.logical_not(zeros))
d_spec.err = np.divide(d_spec.err, transm, where=np.logical_not(zeros))

# mask regions with deep telluric lines
tell_threshold = 0.7
tell_mask = molecfit_spec.flux < tell_threshold
d_spec.flux[tell_mask] = np.nan
d_spec.update_isfinite_mask()
d_spec.clip_det_edges(30)


# Flatten arrays for plotting purposes
d_spec.reshape_orders_dets()
# convert to erg/s/cm2/cm (quick manual fix for lack of flux calibration..)
d_spec.flux *= 1.8e-10
d_spec.err *= 1.8e-10
d_spec.flux_units = 'erg/s/cm2/nm' # update flux units

## Load precomputed model (from pRT_model.py)
assert pathlib.Path('data/testing_atm.pickle').exists(), 'Run pRT_model.py first'
pRT = pRT_model().pickle_load('data/testing_atm.pickle')



# Generate and plot models at the lower and upper edge of the prior (and center)
def prior_predictive_check(parameters, pRT):

    fig, ax = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 1.], 'wspace':0.05, 'top':0.85})
    

    colors = ['C0', 'C1', 'C2']
    theta = [0.0, 0.5, 1.0] # lower edge, center, upper edge
    for i, theta_i in enumerate(theta):
        cube = theta_i * np.ones(parameters.ndim)
        sample = parameters(cube) # transform the cube to the parameter space
        parameters.add_sample(sample) # add the sample to the parameters (create dictionary)
    
        m_spec = pRT(parameters.params) # generate the model spectrum
        # calculate the likelihood
        # Prepare likelihood
        # COMMENT: check what happens when `scale_flux = False`
        # The flux scaling is applied to all order-detectors except the first one
        # that way we estimate the planet radius from the first order-detector
        loglike = LogLikelihood(d_spec, n_params=parameters.n_params, scale_flux=True)
        
        log_L = loglike(m_spec)

        # plot model spectrum and PT profile
        dict_str = [f'{key_i} = {parameters.params[key_i]}' for key_i in parameters.param_keys]
        ax[0].text(0.0, 1.16 - 0.05*i, '   '.join(dict_str), transform=ax[0].transAxes, fontsize=12, color=colors[i])

        m_spec.wave = d_spec.wave.flatten()
        m_spec.flux = (m_spec.flux * loglike.f[:,:,None]).flatten()
        label = r'$\log \mathcal{L}$ = ' + f'{log_L:.4e}'
        m_spec.plot(ax=ax[0], color=colors[i], label=label, alpha=0.8)

        pRT.PT.plot(ax=ax[1], color=colors[i])


    # plot data
    d_spec.flatten() # flatten the data spectrum (7, 3, 2048) -- > (7*3*2048,)
    d_spec.plot(ax=ax[0], color='black', alpha=0.4, label='data')
    ax[0].fill_between(d_spec.wave, d_spec.flux - d_spec.err, d_spec.flux + d_spec.err, color='black', alpha=0.1)
    
    
    ax[0].legend()
    ax[0].set(xlabel=r'Wavelength [nm]')
    # set y-axis to the right-hand side
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    plt.show()
    return fig, ax

prior_predictive_check(parameters, pRT)