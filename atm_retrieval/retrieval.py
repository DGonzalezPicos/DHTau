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
import warnings
# ignore numpy mean of empty slice
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


from atm_retrieval.parameters import Parameters
from atm_retrieval.spectrum import DataSpectrum
from atm_retrieval.log_likelihood import LogLikelihood
from atm_retrieval.pRT_model import pRT_model
from atm_retrieval.covariance import get_Covariance_class
import atm_retrieval.figures as figs

from atm_retrieval.utils import quantiles

class Retrieval:
    
    
    # Pymultinest default parameters
    output_dir = 'retrieval_outputs'
    n_live_points = 100 # >200 for accuracy, 100 for speed
    evidence_tolerance = 5.0 # 0.5 for accuracy, 5.0 for speed
    n_iter_before_update = int(n_live_points * 3)
    sampling_efficiency = 0.10 # 0.05 for accuracy, 0.10 for speed
    
    bestfit_color = 'green'
    
    def __init__(self, parameters, d_spec, pRT_model, run='testing', prepare_for_covariance=True):
        self.parameters = parameters
        self.d_spec = d_spec
        if prepare_for_covariance: # required matrices for covariance calculation
            self.d_spec.prepare_for_covariance()
        self.pRT_model = pRT_model
        
        # Initialize the log-likelihood
        self.loglike = LogLikelihood(d_spec, self.parameters.n_params, scale_flux=True)
        
        # Initialize the Covariance matrix for each order-detector
        self.Cov = np.empty((self.d_spec.n_orders, self.d_spec.n_dets), dtype=object)
        self.mask_ij = np.ones((self.d_spec.n_orders, self.d_spec.n_dets, self.d_spec.n_pixels), dtype=bool)
        cov_kwargs = dict(
                        trunc_dist   = 3, 
                        scale_GP_amp = True, 
                        max_separation = 20, 
        )
        if 'log_l' in self.parameters.params.keys():
            cov_kwargs['max_separation'] = cov_kwargs['trunc_dist'] * 10**self.parameters.param_priors['log_l'][1]

        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):
                
                # Select only the finite pixels
                mask_ij = np.isfinite(self.d_spec.flux[i,j])
                self.mask_ij[i,j] = mask_ij # store for later use in `log_likelihood.py`
                if not mask_ij.any():
                    print(f'Skipping order {i}, detector {j} with no valid pixels')
                    continue

                self.Cov[i,j] = get_Covariance_class(
                    self.d_spec.err[i,j,mask_ij], 
                    self.parameters.params['cov_mode'], 
                    separation=self.d_spec.separation[i,j], 
                    err_eff=self.d_spec.err_eff[i,j], 
                    **cov_kwargs
                    )
        # delete the unnecessary attributes, now stored in self.Cov objects
        del self.d_spec.separation, 
        del self.d_spec.err_eff, 
        del self.d_spec.err
        # ensure the output directory exists
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.run_dir = pathlib.Path(f'{self.output_dir}/{run}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # default
        self.evaluation = False
        self.cb_count = 0
        
        
    def PMN_lnL_func(self, cube=None, ndim=None, nparams=None):
                
        self.m_spec = self.pRT_model(self.parameters.params, get_contr=self.evaluation)
        if self.d_spec.normalized:
            # normalize the model spectrum in the same way as the data
            self.m_spec.normalize_flux_per_order(**self.d_spec.normalize_args)
        
        # generate spline model for flux decomposition
        self.m_spec.N_knots = self.parameters.params.get('N_knots', 1)
            
        lnL = self.loglike(self.m_spec, self.Cov)
        if np.isfinite(lnL):
            return lnL
        else:
            return -np.inf
    
    def PMN_run(self):
        
        # Pause the process to not overload memory on start-up
        time.sleep(2)

        # Run the MultiNest retrieval
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.parameters, 
            n_dims=self.parameters.n_params, 
            outputfiles_basename=f'{self.run_dir}/pmn_', 
            resume=False, 
            verbose=True, 
            const_efficiency_mode=True, 
            sampling_efficiency=self.sampling_efficiency, 
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
        
    def PMN_analyzer(self):
        # Set-up analyzer object
        analyzer = pymultinest.Analyzer(
            n_params=self.parameters.n_params, 
            outputfiles_basename=f'{self.run_dir}/pmn_',
            )
        stats = analyzer.get_stats()

        # Load the equally-weighted posterior distribution
        posterior = analyzer.get_equal_weighted_posterior()
        posterior = posterior[:,:-1]

        # Read the parameters of the best-fitting model
        bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])
        return posterior, bestfit_params
        
        
        
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
        self.cb_count += 1
        print(f' - Callback {self.cb_count}')
        
        if self.evaluation:
            posterior, bestfit_params = self.PMN_analyzer()
            
        else:
            # Read the parameters of the best-fitting model
            bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2]
            # Remove the last 2 columns
            posterior = posterior[:,:-2]

        if rank != 0:
            return

        # Evaluate the model with best-fitting parameters
        self.parameters.add_sample(bestfit_params)
        self.PMN_lnL_func()

        labels = np.array(list(self.parameters.param_mathtext.values()))
        
        fig_label = 'final' if self.evaluation else f'{self.cb_count}'

            
        fig = figs.simple_cornerplot(posterior,
                                labels, 
                                bestfit_params=bestfit_params)
        l, b, w, h = [0.32,3.42,0.65,0.20]

        ax_res_dim  = [l, b*(h+0.03), w, 0.97*h/5]
        ax_spec_dim = [l, ax_res_dim[1]+ax_res_dim[3], w, 4*0.97*h/5]


        ax_spec = fig.add_axes(ax_spec_dim)
        ax_res = fig.add_axes(ax_res_dim)
        
        ax_spec, ax_res = figs.fig_bestfit_model(
            self.d_spec, 
            self.m_spec,
            self.loglike,
            Cov=getattr(self, 'Cov', None),
            xlabel=r'Wavelength [nm]',
            ylabel=r'Flux [erg/s/cm$^2$/cm]',
            ax_spec=ax_spec,
            ax_res=ax_res,
            bestfit_color=self.bestfit_color,
            # fig_name=self.run_dir / f'plots/retrieval_bestfit_model.pdf',
            )
            
        
        l, b, w, h = [0.69,0.45,0.28,0.28]
        ax_PT = fig.add_axes([l,b,h,h])
        # Plot the best-fitting PT profile
        x1, x2 = np.min(self.pRT_model.PT.temperature), np.max(self.pRT_model.PT.temperature)
        x_pad = 0.05*(x2-x1)
        x1 -= x_pad
        x2 += x_pad
        
        # if self.evaluation
        # PT envelopes
        temperature_samples = []
        for sample in posterior:
            self.parameters.add_sample(sample)
            self.pRT_model.get_temperature(self.parameters.params)
            temperature_samples.append(self.pRT_model.temperature)
            
        # Convert profiles to 1, 2, 3-sigma equivalent and median
        q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
            0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
            ]  
        self.pRT_model.PT.temperature_envelopes = quantiles(np.array(temperature_samples), q=q, axis=0)
        
        if hasattr(self.pRT_model, 'int_contr_em'):
            if np.sum(np.isnan(self.pRT_model.int_contr_em)) == 0:
                print(f'Copying integrated contribution emission from pRT_atm to PT')
                self.pRT_model.PT.int_contr_em = self.pRT_model.int_contr_em
        ax_PT = figs.fig_PT(
            PT=self.pRT_model.PT, 
            ax=ax_PT, 
            fig=fig,
            # xlim=(x1,x2), 
            bestfit_color=self.bestfit_color,
            envelopes_color=self.bestfit_color,
            int_contr_em_color='red',
            # fig_name=self.run_dir / f'plots/retrieval_PT_profile.pdf',
            )
        
        fig.savefig(self.run_dir / f'plots/retrieval_summary_{fig_label}.pdf')
        print(f' - Saved {self.run_dir / f"plots/retrieval_summary_{fig_label}.pdf"}')
            
        if self.evaluation:
                       
            figs.fig_bestfit_model(
                self.d_spec, 
                self.m_spec,
                self.loglike,
                Cov=getattr(self, 'Cov', None),
                xlabel=r'Wavelength [nm]',
                ylabel=r'Flux [erg/s/cm$^2$/cm]',
                bestfit_color=self.bestfit_color,
                fig_name=self.run_dir / f'plots/retrieval_bestfit_model_{fig_label}.pdf',
                )
            figs.fig_PT(
                    PT=self.pRT_model.PT, 
                    # xlim=(x1,x2), 
                    bestfit_color=self.bestfit_color,
                    envelopes_color=self.bestfit_color,
                    int_contr_em_color='red',
                    fig_name=self.run_dir / f'plots/retrieval_PT_profile_{fig_label}.pdf',
                    )
            
    def prior_check(self):
        
        # Check the prior space 
        figs.fig_prior_check(self, fig_name=self.run_dir / 'plots/prior_check.pdf')
        return self
        
        
        
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