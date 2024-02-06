import numpy as np
import pathlib

import argparse

from atm_retrieval.parameters import Parameters
from atm_retrieval.spectrum import DataSpectrum
from atm_retrieval.pRT_model import pRT_model
from atm_retrieval.retrieval import Retrieval
from atm_retrieval.utils import pickle_load, pickle_save


run = 'testing_004'
run_dir = pathlib.Path(f'retrieval_outputs/{run}')
run_dir.mkdir(parents=True, exist_ok=True)

plots_dir = run_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
    
    
# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
parser.add_argument('--prior_check', '-c', action='store_true', default=False)
parser.add_argument('--retrieval', '-r', action='store_true', default=False)
args = parser.parse_args()

cache = True
## Define parameters
free_params = {
    # GPs
    'log_a': [(-1,0.5), r'$\log\ a$'], 
    'log_l': [(-2,-0.8), r'$\log\ l$'], 
    
    
    'log_g': [(2.5,5.5), r'$\log\ g$'], 

    'vsini' : ([2.0, 16.0], r'$v \sin(i)$ [km/s]'),
    'rv'    : ([-30.0, 30.0], r'RV [km/s]'),
    
    # chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 

    'log_H2O'  : ([-12, -2], r'$\log$(H$_2$O)'),
    'log_Na'   : ([-12, -2], r'$\log$(Na)'),
    
    # temperature profile
    'T1' : ([5000, 8000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
    'T2' : ([3000, 6000], r'$T_2$ [K]'),
    'T3' : ([1000, 3000], r'$T_3$ [K]'),
    'T4' : ([200,  2000],  r'$T_4$ [K]'),
}

constant_params = {
    'log_P_knots': [2, 1, -1, -5], # [log(bar)]
    'R_p'    : 1.0,
    'distance': 133.3, # [pc] Gaia EDR3 parallactic distance from Bailer-Jones et al. (2021)
    # 'log_g' : 4.0,
    'epsilon_limb' : 0.5,
    
}
parameters = Parameters(free_params, constant_params)

if args.pre_processing:
    print('Pre-processing...')
    # Run the pre-processing

    ## Load data
    file_data = 'data/DHTauA.dat'
    d_spec = DataSpectrum(file_target=file_data, 
                          slit='w_0.2', 
                          flux_units='photons',
                          wave_range=[2200, 2480])
    d_spec.preprocess(
        file_transm='data/DHTauA_molecfit_transm.dat',
        tell_threshold=0.4,
        n_edge_pixels=30,
        sigma_clip=3,
        sigma_clip_window=11,
        ra=67.422516,
        dec=26.54998,
        mjd=59945.15094260,
        fig_name=plots_dir / 'preprocessed_spec.pdf'
        )
    d_spec.pickle_save(run_dir / 'd_spec.pickle')
    print(f' Preprocessed spectrum saved to {run_dir / "d_spec.pickle"}')
    # output shape of wave, flux, err = (n_orders, n_dets, n_wave) = (7, 3, 2048)
    
    ## Prepare pRT model
    if (run_dir / 'atm.pickle').exists() and cache:
        print('Loading precomputed pRT model...')
        pRT = pRT_model().pickle_load(run_dir / 'atm.pickle')
    else:
        line_species_dict = {
            
            'H2O': 'H2O_pokazatel_main_iso',
            '12CO': 'CO_high',
            'Na': 'Na_allard',
        }
        pRT = pRT_model(line_species_dict=line_species_dict,
                        d_spec=d_spec,
                        mode='lbl',
                        lbl_opacity_sampling=5,
                        rayleigh_species=['H2', 'He'],
                        continuum_opacities=['H2-H2', 'H2-He'],
                        log_P_range=(-5,2),
                        n_atm_layers=30,
                        rv_range=(-50,50))
        

        # Load opacities and prepare a Radtrans instance for every order-detector
        pRT.get_atmospheres()
        pRT.pickle_save(run_dir / 'atm.pickle')
        print(f' pRT model saved to {run_dir / "atm.pickle"}')
        
        sample = parameters.random_sample
        parameters.add_sample(sample)
        m_spec = pRT(parameters.params)
    

if args.prior_check:
    import matplotlib.pyplot as plt
    print('Prior predictive check...')
    
    d_spec = pickle_load(run_dir / 'd_spec.pickle')
    pRT = pickle_load(run_dir / 'atm.pickle')
    ret = Retrieval(parameters, d_spec, pRT)
    
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), 
                           gridspec_kw={'width_ratios': [3, 1.], 'wspace':0.05, 'top':0.85})
    

    colors = ['C0', 'C1', 'C2']
    theta = [0.0, 0.5, 1.0] # lower edge, center, upper edge
    for i, theta_i in enumerate(theta):
        cube = theta_i * np.ones(ret.parameters.ndim)
        sample = ret.parameters(cube) # transform the cube to the parameter space
        ret.parameters.add_sample(sample) # add the sample to the parameters (create dictionary)
    
        m_spec = ret.pRT_model(ret.parameters.params) # generate the model spectrum
        
        log_L = ret.loglike(m_spec)

        # plot model spectrum and PT profile
        dict_str = [f'{key_i} = {ret.parameters.params[key_i]}' for key_i in ret.parameters.param_keys]
        ax[0].text(0.0, 1.16 - 0.05*i, '   '.join(dict_str), transform=ax[0].transAxes, fontsize=12, color=colors[i])

        m_spec.wave = ret.d_spec.wave
        
        for order in range(m_spec.n_orders):
            for det in range(m_spec.n_dets):
                m_spec.flux[order, det] *= ret.loglike.f[order, det]
                label = r'$\log \mathcal{L}$ = ' + f'{log_L:.4e}' if (order+det)==0 else None

                ax[0].plot(m_spec.wave[order, det], m_spec.flux[order, det], color=colors[i], alpha=0.8, label=label)
            
        ret.pRT_model.PT.plot(ax=ax[1], color=colors[i])


    # plot data
    ret.d_spec.flatten() # flatten the data spectrum (7, 3, 2048) -- > (7*3*2048,)
    ret.d_spec.plot(ax=ax[0], color='black', alpha=0.4, label='data')
    ax[0].fill_between(ret.d_spec.wave, ret.d_spec.flux - ret.d_spec.err, ret.d_spec.flux + ret.d_spec.err, color='black', alpha=0.1)
    
    
    ax[0].legend()
    ax[0].set(xlabel=r'Wavelength [nm]')
    # set y-axis to the right-hand side
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    # plt.show()
    fig.savefig(plots_dir / 'prior_predictive_check.pdf', bbox_inches='tight')
    print(f' Prior predictive check saved to {plots_dir / "prior_predictive_check.pdf"}')
    plt.close(fig)
    
    
    
    

if args.retrieval:
    print('Retrieval...')
    ### Init retrieval object
    d_spec = pickle_load(run_dir / 'd_spec.pickle')
    pRT = pickle_load(run_dir / 'atm.pickle')
    ret = Retrieval(parameters, d_spec, pRT)

    # uncomment line below to run the retrieval
    ret.PMN_run(run=run)
    
    # call this file from the command line (with modules loaded and environment activated) as follows:
    # replace 64 by the number of CPU cores available
    # mpiexec -np 64 python run_retrieval.py -r