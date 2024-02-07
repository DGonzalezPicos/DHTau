import os
os.environ['OMP_NUM_THREADS'] = '1'
from mpi4py import MPI
import time
import pickle
import corner 
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import pathlib
import pymultinest

from atm_retrieval.parameters import Parameters
from atm_retrieval.spectrum import DataSpectrum
from atm_retrieval.log_likelihood import LogLikelihood
from atm_retrieval.pRT_model import pRT_model

from atm_retrieval.utils import quantiles

class Retrieval:
    
    
    # Pymultinest default parameters
    output_dir = 'retrieval_outputs'
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
        
        # ensure the output directory exists
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # default
        self.evaluation = False
        self.cb_count = 0
        
        
    def PMN_lnL_func(self, cube=None, ndim=None, nparams=None):
        
        # Transform the cube to the parameter space
        # sample = self.parameters(cube) # the attribute self.parameters.params is updated
        # self.parameters.add_sample(sample)
        
        self.m_spec = self.pRT_model(self.parameters.params)
        lnL = self.loglike(self.m_spec)
        return lnL
    
    def PMN_run(self, run='testing'):
        
        run_dir = f'{self.output_dir}/{run}'
        output_dir = pathlib.Path(run_dir) / 'output'
        
        # Pause the process to not overload memory on start-up
        time.sleep(1.5*rank)

        # Run the MultiNest retrieval
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.parameters, 
            n_dims=self.parameters.n_params, 
            outputfiles_basename=f'{self.output_dir}/pmn_', 
            resume=False, 
            verbose=True, 
            const_efficiency_mode=True, 
            sampling_efficiency=0.05, 
            n_live_points=self.n_live_points, 
            evidence_tolerance=self.evidence_tolerance, 
            n_iter_before_update=self.n_iter_before_update, 
            dump_callback=self.PMN_callback,
            )
        
    def pickle_save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        return None
    
    def pickle_load(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
    def PMN_callback(self, 
                          n_samples, 
                          n_live, 
                          n_params, 
                          live_points, 
                          posterior, 
                          stats,
                          max_ln_L, 
                          ln_Z, 
                          ln_Z_err, 
                          nullcontext
                          ):
        # self.CB.active = True

        if self.evaluation:

            # Set-up analyzer object
            analyzer = pymultinest.Analyzer(
                n_params=self.parameters.n_params, 
                outputfiles_basename=f'{self.output_dir}/pmn_',
                )
            stats = analyzer.get_stats()

            # Load the equally-weighted posterior distribution
            posterior = analyzer.get_equal_weighted_posterior()
            posterior = posterior[:,:-1]

            # Read the parameters of the best-fitting model
            bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])
            
            # PT envelopes
            temperature_env = []
            for sample in posterior:
                self.parameters.params.add_sample(sample)
                self.pRT_model.get_temperature(self.parameters.params)
                temperature_env.append(self.pRT_model.temperature)

        else:

            # Read the parameters of the best-fitting model
            bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2]

            # Remove the last 2 columns
            posterior = posterior[:,:-2]

        if rank != 0:
            return

        # Evaluate the model with best-fitting parameters
        # for i, key_i in enumerate(self.parameters.param_keys):
        #     # Update the Parameters instance
        #     self.parameters.params[key_i] = bestfit_params[i]
        #     if key_i.startswith('log_'):
        #         self.parameters.params = self.parameters.log_to_linear(self.Param.params, key_i)

        # # Update the parameters
        # # self.parameters.read_PT_params(cube=None)
        # # self.parameters.read_uncertainty_params()
        # # self.parameters.read_chemistry_params()

        # if self.evaluation:
        #     # Get each species' contribution to the spectrum
        #     self.get_species_contribution()

        # # Update class instances with best-fitting parameters
        # self.PMN_lnL_func()
        # self.CB.active = False
        
        # # TODO: implement function below...
        # # self.CB = CallBack(...)
        
        # simple cornerplot
        print(f'Best-fitting parameters: {bestfit_params}')
        
        labels = np.array(list(self.parameters.param_mathtext.values()))
        
        # get quantiles for ranges
        Q = np.array(
                [quantiles(posterior[:,i], q=[0.16,0.5,0.84]) \
                for i in range(posterior.shape[1])]
                )
            
        ranges = np.array(
            [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
                for q_i in Q]
            )
        
        fontsize = 12
        color = 'green'
        smooth = 1.0
        # plot cornerplot
        fig = corner.corner(posterior, 
                            labels=labels, 
                            title_kwargs={'fontsize': fontsize},
                            labelpad=0.25*posterior.shape[0]/17,
                            bins=20,
                            max_n_ticks=3,
                            show_titles=True,
                            range=ranges,
                            
                            quantiles=[0.16,0.84],
                            title_quantiles=[0.16,0.5,0.84],
                            
                            color=color,
                            linewidths=0.5,
                            hist_kwargs={'color':color,
                                            'linewidth':0.5,
                                            'density':True,
                                            'histtype':'stepfilled',
                                            'alpha':0.5,
                                            },
                            
                            fill_contours=True,
                            smooth=smooth,
                            )
        corner.overplot_lines(fig, bestfit_params, color='green', lw=0.5)
        fig_label = 'final' if self.evaluation else f'{self.cb_count}'

        outfig = f'{self.output_dir}/corner_{fig_label}.png'
        fig.savefig(outfig)
        print(f'Saved {outfig}\n')
        plt.close(fig)
        

        
        
        
if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    run = 'testing_000'
    run_dir = pathlib.Path(f'retrieval_outputs/{run}')
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    ## Define parameters
    free_params = {
        # general properties
        # 'Rp'    : ([0.5, 2.0], r'$R_p$ [R$_J$]'),
        # 'log_g' : ([3.0, 5.5], r'$\log(g)$ [cm/s$^2$]'),
        'log_a': [(-1,0.5), r'$\log\ a$'], 
        'log_l': [(-2,-0.8), r'$\log\ l$'], 
        
        
        'log_g': [(2.5,5.5), r'$\log\ g$'], 

        'vsini' : ([1.0, 20.0], r'$v \sin(i)$ [km/s]'),
        'rv'    : ([-30.0, 30.0], r'RV [km/s]'),
        
        # chemistry
        'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
        'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 

        'logH2O'  : ([-12, -2], r'$\log$(H$_2$O)'),
        'logNa'   : ([-12, -2], r'$\log$(Na)'),
        
        # temperature profile
        'T1' : ([2000, 20000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
        'T2' : ([1000, 20000], r'$T_2$ [K]'),
        'T3' : ([300,  10000], r'$T_3$ [K]'),
        'T4' : ([300,  5000],  r'$T_4$ [K]'),
    }

    constant_params = {
        'log_P_knots': [2, 1, -1, -5], # [log(bar)]
        'Rp'    : 1.0,
        'log_g' : 4.0,
        'epsilon_limb' : 0.5,
        
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
        ra=67.422516,
        dec=26.54998,
        mjd=59945.15094260,
        )
    # output shape of wave, flux, err = (n_orders, n_dets, n_wave) = (7, 3, 2048)
    
    ## Load pRT model
    pRT_model = pRT_model().pickle_load('data/testing_atm.pickle')
    
    ### Init retrieval object
    ret = Retrieval(parameters, d_spec, pRT_model)
    ret.pickle_save(run_dir / 'retrieval.pickle') # save the retrieval object
    
    cube = np.random.rand(parameters.n_params)
    print(f'Testing the log-likelihood function on a random sample...')
    lnL = ret.PMN_lnL_func(cube)
    print(f'lnL = {lnL:.4e}')
    
    
    # uncomment line below to run the retrieval
    ret.PMN_run(run='testing_000')
    
    ret.evaluation = True