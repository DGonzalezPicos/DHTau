import numpy as np
import pathlib
import time

import argparse

from atm_retrieval.parameters import Parameters
from atm_retrieval.spectrum import DataSpectrum
from atm_retrieval.pRT_model import pRT_model
from atm_retrieval.retrieval import Retrieval
from atm_retrieval.utils import pickle_load, pickle_save


run = 'night2_veiling'
run_dir = pathlib.Path(f'retrieval_outputs/{run}')
run_dir.mkdir(parents=True, exist_ok=True)

plots_dir = run_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
    
# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', '-p', action='store_true', default=False)
parser.add_argument('--prior_check', '-c', action='store_true', default=False)
parser.add_argument('--retrieval', '-r', action='store_true', default=False)
parser.add_argument('--evaluation', '-e', action='store_true', default=False)
args = parser.parse_args()

## Define parameters
free_params = {
    # GPs
    'log_a': [(-1,0.5), r'$\log\ a$'], 
    'log_l': [(-2,-1.2), r'$\log\ l$'], 

    'log_g': [(2.0,5.0), r'$\log\ g$'], 
    # 'r_k'  : [(0.0, 3.0), r'$r_k$'], # veiling factor (0 for no veiling, can be > 1)

    'vsini' : ([2.0, 16.0], r'$v \sin(i)$ [km/s]'),
    # 'vsini' : ([5.2, 5.4], r'$v \sin(i)$ [km/s]'),
    'rv'    : ([-30.0, 30.0], r'RV [km/s]'),
    # 'rv' : ([16.0, 16.1], r'RV [km/s]'),
    
    # chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 

    'log_H2O'  : ([-12, -2], r'$\log$(H$_2$O)'),
    
    # isotologue of water with 18O
    'log_H2O_181': ([-12, -2], r'$\log\ \mathrm{H_2^{18}O}$'),
    'log_Na'   : ([-12, -2], r'$\log$\ Na'),
    'log_HF'   : ([-12, -2], r'$\log$\ HF'),
    'log_Ca'   : ([-12, -2], r'$\log$\ Ca'),
    'log_Ti'   : ([-12, -2], r'$\log$\ Ti'),
    'log_CN'   : ([-12, -2], r'$\log$\ CN'),
    # 'log_13CN' : ([-12, -2], r'$\log\ \mathrm{^{13}CN}$'),
    #'log_Mg'   : ([-12, -2], r'$\log$(Mg)'),
    #'log_Fe'   : ([-12, -2], r'$\log$(Fe)'),
    #'log_Al'   : ([-12, -2], r'$\log$(Al)'),
    
    
    # temperature profile
    # 'T1' : ([5000, 8000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
    # 'T2' : ([1000, 6000], r'$T_2$ [K]'),
    # 'T3' : ([600,  6000],  r'$T_3$ [K]'),
    # 'T4' : ([600, 3000], r'$T_4$ [K]'),
    # 'T5' : ([200,  2000],  r'$T_5$ [K]'),
    
    # temperature gradients
    'T1': ([3000, 10000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
    'dlnT_dlnP_1': [(0.04,  0.40), r'$\nabla T_1$'],
    'dlnT_dlnP_2': [(0.04,  0.40), r'$\nabla T_2$'],
    'dlnT_dlnP_3': [(-0.04, 0.40), r'$\nabla T_3$'],
    'dlnT_dlnP_4': [(-0.10, 0.40), r'$\nabla T_4$'],
    'dlnT_dlnP_5': [(-0.10, 0.40), r'$\nabla T_5$'],
    'dlnT_dlnP_6': [(-0.10, 0.40), r'$\nabla T_4$'],
    'dlnT_dlnP_7': [(-0.10, 0.40), r'$\nabla T_5$'],
}

constant_params = {
    'log_P_knots': [2, 1, 0, -1, -2, -3, -5], # [log(bar)]
    'R_p'    : 1.0,
    'distance': 133.3, # [pc] Gaia EDR3 parallactic distance from Bailer-Jones et al. (2021)
    # 'log_g' : 4.0,
    'epsilon_limb' : 0.5,
    'N_knots': 1, # 2048/N_knots high-pass filter
    'N_veiling': 5, # number of veiling components
    'cov_mode': 'GP', # 'GP' or 'covariance'
    
}

parameters_file = run_dir / 'parameters.json'
cache = False

if parameters_file.exists() and cache:
    print(f'--> Loading parameters from file {parameters_file}...')
    parameters = Parameters.load(parameters_file)
    
else:
    parameters = Parameters(free_params, constant_params)
    parameters.save(parameters_file)

    
if args.pre_processing:
    print('--> Pre-processing...')
    
    ## Load data
    # DGP (2024-04-21) run retrieval on two nights
    # file_data = [f'data/VDHTauA+Bcenter_PRIMARY_CRIRES_SPEC1D_night1.dat',
    #              f'data/VDHTauA+Bcenter_PRIMARY_CRIRES_SPEC1D_night2.dat']
    
    # run retrieval on one night (choose night 1 or 2)
    file_data = [f'data/VDHTauA+Bcenter_PRIMARY_CRIRES_SPEC1D_night2.dat']
    
    assert isinstance(file_data, list), 'file_data must be a list of strings (even if it has only one element)'
    
    d_spec_list = []
    for i, file in enumerate(file_data):
        d_spec = DataSpectrum(file_target=file, 
                            slit='w_0.4', 
                            flux_units='photons',
                            wave_range=[1990, 2480],
                            night=(i+1) if len(file_data) > 1 else None,)
        
        
        d_spec.preprocess(
            # file_transm='data/DHTauA_molecfit_transm.dat',
            file_transm=None, # included in `file_target` now
            tell_threshold=0.50,
            tell_grow_mask=31,
            n_edge_pixels=40,
            sigma_clip=4,
            sigma_clip_window=11,
            ra=67.422516,
            dec=26.54998,
            mjd=59945.15094260,
            # fig_name=plots_dir / 'preprocessed_spec.pdf' # deprecated
            fig_dir = plots_dir
            )
        d_spec_list.append(d_spec)
        
    # Add data from two different nights (new shape of wave, flux, err = (n_orders, n_dets, n_wave) = (7, 3*2, 2048)
    d_spec = d_spec_list[0].add_dataset(d_spec_list[1]) if len(d_spec_list) > 1 else d_spec_list[0]
        
        
    d_spec.pickle_save(run_dir / 'd_spec.pickle')
    print(f' Preprocessed spectrum saved to {run_dir / "d_spec.pickle"}')
    # output shape of wave, flux, err = (n_orders, n_dets, n_wave) = (7, 3, 2048)
    
    ## Prepare pRT model
    if (run_dir / 'atm.pickle').exists() and cache:
        print(' Loading precomputed pRT model...')
        pRT = pRT_model().pickle_load(run_dir / 'atm.pickle')
    else:
        # ls /net/lem/data2/pRT_input_data/opacities/lines/line_by_line/
        line_species_dict = {
            
            'H2O': 'H2O_pokazatel_main_iso',
            # 'H2O_181': 'H2O_181',
            'H2O_181': 'H2O_181_HotWat78',
            '12CO': 'CO_high',
            '13CO': 'CO_36_high',
            'Na': 'Na_allard',
            'HF': 'HF_main_iso',
            'Ca': 'Ca',
            'Ti': 'Ti',
            'CN': 'CN_high', # DGP (2024-04-15), new linelist up to 4000 K
            # '13CN': 'CN_34_high',
            #'Mg': 'Mg',
            #'Fe': 'Fe',
            #'Al': 'Al'
        }
        pRT = pRT_model(line_species_dict=line_species_dict,
                        d_spec=d_spec,
                        mode='lbl',
                        # WARNING: setting `lbl_opacity_sampling = 10` underestimates vsini
                        # and can lead to wrong log_g and PT profiles
                        lbl_opacity_sampling=5, # set to 5 for speed, 3 for accuracy
                        rayleigh_species=['H2', 'He'],
                        continuum_opacities=['H2-H2', 'H2-He' ], #, 'H-'],
                        log_P_range=(-5,2),
                        n_atm_layers=30, # set to 20 for speed, 30 for accuracy
                        rv_range=(-50,50))
        

        # Load opacities and prepare a Radtrans instance for every order-detector
        pRT.get_atmospheres()
        pRT.pickle_save(run_dir / 'atm.pickle')
        print(f' pRT model saved to {run_dir / "atm.pickle"}')
    

if args.prior_check:
    import matplotlib.pyplot as plt
    print('--> Prior predictive check...')
    
    d_spec = pickle_load(run_dir / 'd_spec.pickle')
    pRT = pickle_load(run_dir / 'atm.pickle')
    ret = Retrieval(parameters, d_spec, pRT, run=run)
    ret.prior_check() # new function to plot the prior predictive check
    

if args.retrieval:
    print('--> Retrieval...')
    ret_start_time = time.time()

    ### Init retrieval object
    d_spec = pickle_load(run_dir / 'd_spec.pickle')
    pRT = pickle_load(run_dir / 'atm.pickle')
    ret = Retrieval(parameters, d_spec, pRT, run=run)
    ret.n_live_points = 100
    # ret.n_iter_before_update = 1
    # uncomment line below to run the retrieval
    ret.PMN_run()

    ret_end_time = time.time()
    ret_duration = (ret_end_time - ret_start_time)
    print('Retrieval time:', ret_duration, 'seconds')
    with open(run_dir / 'retrieval_time.txt', 'w') as f:
        f.write('{}'.format(ret_duration))
    
    # call this file from the command line (with modules loaded and environment activated) as follows:
    # replace 64 by the number of CPU cores available
    # mpiexec -np 64 python run_retrieval.py -r
    # mpiexec --use-hwthread-cpus --bind-to none -np 80 python run_retrieval.py -r
    
if args.evaluation:
    print('--> Evaluation...')

    # Load the retrieval object
    d_spec = pickle_load(run_dir / 'd_spec.pickle')
    pRT = pickle_load(run_dir / 'atm.pickle')
    ret = Retrieval(parameters, d_spec, pRT, run=run)
    
    # print(d_spec)
    
    ret.evaluation = True
    ret.PMN_callback(
            n_samples=None, 
            n_live=None, 
            n_params=None, 
            live_points=None, 
            posterior=None, 
            stats=None,
            max_ln_L=None, 
            ln_Z=None, 
            ln_Z_err=None, 
            nullcontext=None
            )