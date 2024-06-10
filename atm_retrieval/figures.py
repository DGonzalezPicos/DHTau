import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter1d

import os
import copy
import corner
import pathlib
import petitRADTRANS.nat_cst as nc

from atm_retrieval.utils import quantiles, weigh_alpha
from atm_retrieval.spline_model import SplineModel

# make borders thicker
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5


def fig_order_subplots(n_orders, ylabel, xlabel=r'Wavelength (nm)'):

    fig, ax = plt.subplots(
        figsize=(10,2.8*n_orders), nrows=n_orders, 
        gridspec_kw={'hspace':0.22, 'left':0.1, 'right':0.95, 
                     'top':(1-0.02*7/n_orders), 'bottom':0.035*7/n_orders, 
                     }
        )
    if n_orders == 1:
        ax = np.array([ax])

    ax[n_orders//2].set(ylabel=ylabel)
    ax[-1].set(xlabel=xlabel)

    return fig, ax

def fig_spec_to_fit(d_spec, prefix=None, w_set='', overplot_array=None, fig_name=None):

    ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
   
    if len(d_spec.flux.shape) == 1:
        # reshape to match d_spec.fig_flux_calib_2MASS
        d_spec.reshape_orders_dets()

    
    if overplot_array is not None:
        if np.shape(overplot_array) != (d_spec.n_orders, d_spec.n_dets):
            # reshape to match d_spec.fig_flux_calib_2MASS
            overplot_array = np.reshape(overplot_array, d_spec.flux.shape)
            
            
    fig, ax = fig_order_subplots(d_spec.n_orders, ylabel=ylabel)

    for i in range(d_spec.n_orders):
        ax[i].axhline(1.0, c='k', ls='-', lw=0.5, alpha=0.5)
        for j in range(d_spec.n_dets):
            ax[i].plot(d_spec.wave[i,j], d_spec.flux[i,j], c='k', lw=0.5)
            if overplot_array is not None:
                ax[i].plot(d_spec.wave[i,j], overplot_array[i,j], c='brown', lw=1, alpha=0.3)
        
        ax[i].set(xlim=(d_spec.order_wlen_ranges[i].min()-0.5, 
                        d_spec.order_wlen_ranges[i].max()+0.5)
                  )

    
    fig_name = fig_name if fig_name is not None else prefix+f'plots/spec_to_fit_{w_set}.pdf'
    plt.savefig(fig_name)
    print(f' Figure saved as {fig_name}')
    #plt.show()
    plt.close(fig)
    
def fig_sigma_clip(d_spec, clip_mask, fig_name=None):
    
    assert d_spec.flux.shape == clip_mask.shape, 'Shapes of d_spec.flux and flux_clip do not match'
    ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    fig, ax = fig_order_subplots(d_spec.n_orders, ylabel=ylabel)
    lw = 0.8
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            
            mask = clip_mask[i,j]
            f_clip = np.where(mask, d_spec.flux[i,j], np.nan)
            f_clean  = np.where(~mask, d_spec.flux[i,j], np.nan) 
            ax[i].plot(d_spec.wave[i,j], f_clip, c='r', lw=lw)
            # if overplot_array is not None:
            ax[i].plot(d_spec.wave[i,j], f_clean, c='k', lw=lw, alpha=0.9)
        
        ax[i].set(xlim=(d_spec.order_wlen_ranges[i].min()-0.5, 
                        d_spec.order_wlen_ranges[i].max()+0.5)
                  )

    if fig_name is not None:
        fig_name = fig_name if fig_name is not None else prefix+f'plots/spec_to_fit_{w_set}.pdf'
        plt.savefig(fig_name)
        print(f' Figure saved as {fig_name}')
        #plt.show()
        plt.close(fig)
    return fig, ax

def fig_PT(PT,
            ax=None, 
            fig=None,
            xlim=None, 
            bestfit_color='C0',
            envelopes_color='C0',
            int_contr_em_color='red',
            fig_name=None,
    ):

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))
        
    # assert hasattr(PT, 'temperature_envelopes'), 'No temperature envelopes found'
    
    p = PT.pressure
    if hasattr(PT, 'int_contr_em'):
        if np.max(PT.int_contr_em) > 0.0: # additional check
            # Plot the integrated contribution emission
            ax_twin = ax.twiny()
            ax_twin.plot(
                PT.int_contr_em, p, 
                c='red', lw=1, alpha=0.4,
                )
            # weigh_alpha(PT.int_contr_em, p, np.linspace(0,10000,p.size), ax, alpha_min=0.5, plot=True)
            # define photosphere as region where PT.int_contr_em > np.quantile(PT.int_contr_em, 0.9)
            photosphere = PT.int_contr_em > np.quantile(PT.int_contr_em, 0.95)
            P_phot = np.mean(p[photosphere])
            T_phot = np.mean(PT.temperature_envelopes[3][photosphere])
            T_phot_err = np.std(PT.temperature_envelopes[3][photosphere])
            # print(f' - Photospheric temperature: {T_phot:.1f} +- {T_phot_err:.1f} K')
            # make empty marker
            ax.scatter(T_phot, P_phot, c='red',
                        marker='o', 
                        s=50, 
                        alpha=0.5,
                        zorder=10,
                        label=f'T$_\mathrm{{phot}}$ = {T_phot:.0f} $\pm$ {T_phot_err:.0f} K')
            
            
            # remove xticks
            ax_twin.set_xticks([])
            ax_twin.spines['top'].set_visible(False)
            ax_twin.spines['bottom'].set_visible(False)
            ax_twin.set(
                # xlabel='Integrated contribution emission',
                xlim=(0,np.max(PT.int_contr_em)*1.1),
                )
    if hasattr(PT, 'temperature_envelopes'):
        # Plot the PT confidence envelopes
        for i in range(3):
            ax.fill_betweenx(
                y=p, x1=PT.temperature_envelopes[i], 
                x2=PT.temperature_envelopes[-i-1], 
                color=envelopes_color, ec='none', 
                alpha=0.3,
                )

        # Plot the median PT
        ax.plot(
            PT.temperature_envelopes[3], p, 
            c=bestfit_color, lw=2,
    )

        xlim = (0, PT.temperature_envelopes[-1].max()*1.02) if xlim is None else xlim
    else:
        ax.plot(PT.temperature, p, c=bestfit_color, lw=2)
        xlim = (0, PT.temperature.max()*1.02) if xlim is None else xlim
    ax.set(xlabel='Temperature (K)', ylabel='Pressure (bar)',
            ylim=(p.max(), p.min()), yscale='log',
            xlim=xlim,
            )
    #ax.legend(loc='upper right', fontsize=12)
    
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f' - Saved {fig_name}')
        plt.close(fig)
   
    return fig, ax

def fig_PT_phoenix(PT,
            ax=None, 
            fig=None,
            xlim=None, 
            bestfit_color='black',
            envelopes_color='black',
            int_contr_em_color='red',
            fig_name=None,
    ):

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))
        
    # assert hasattr(PT, 'temperature_envelopes'), 'No temperature envelopes found'
    
    p = PT.pressure
    if hasattr(PT, 'int_contr_em'):
        if np.max(PT.int_contr_em) > 0.0: # additional check
            # Plot the integrated contribution emission
            ax_twin = ax.twiny()
            ax_twin.plot(
                PT.int_contr_em, p, 
                c='red', lw=1, alpha=0.4,
                )
            # weigh_alpha(PT.int_contr_em, p, np.linspace(0,10000,p.size), ax, alpha_min=0.5, plot=True)
            # define photosphere as region where PT.int_contr_em > np.quantile(PT.int_contr_em, 0.9)
            photosphere = PT.int_contr_em > np.quantile(PT.int_contr_em, 0.95)
            P_phot = np.mean(p[photosphere])
            T_phot = np.mean(PT.temperature_envelopes[3][photosphere])
            T_phot_err = np.std(PT.temperature_envelopes[3][photosphere])
            # print(f' - Photospheric temperature: {T_phot:.1f} +- {T_phot_err:.1f} K')
            # make empty marker
            
            #ax.scatter(T_phot, P_phot, c='red',
             #           marker='o', 
              #          s=50, 
               #         alpha=0.5,
                #        zorder=10,
                 #       label=f'T$_\mathrm{{phot}}$ = {T_phot:.0f} $\pm$ {T_phot_err:.0f} K')
            
            
            # remove xticks
            ax_twin.set_xticks([])
            ax_twin.spines['top'].set_visible(False)
            ax_twin.spines['bottom'].set_visible(False)
            ax_twin.set(
                # xlabel='Integrated contribution emission',
                xlim=(0,np.max(PT.int_contr_em)*1.1),
                )
    if hasattr(PT, 'temperature_envelopes'):
        # Plot the PT confidence envelopes
        for i in range(3):
            ax.fill_betweenx(
                y=p, x1=PT.temperature_envelopes[i], 
                x2=PT.temperature_envelopes[-i-1], 
                color=envelopes_color, ec='none', 
                alpha=0.3,
                )

        # Plot the median PT
        ax.plot(
            PT.temperature_envelopes[3], p, 
            c=bestfit_color, lw=2, label='retrieved'
    )
        

        xlim = (0, PT.temperature_envelopes[-1].max()*1.02) if xlim is None else xlim
    else:
        ax.plot(PT.temperature, p, c=bestfit_color, lw=2,label='retrieved')
        xlim = (0, PT.temperature.max()*1.02) if xlim is None else xlim
    ax.set(xlabel='Temperature (K)', ylabel='Pressure (bar)',
            ylim=(p.max(), p.min()), yscale='log',
            xlim=xlim,
            )
    #ax.legend(loc='upper right', fontsize=12)

    ### PHOENIX PART ###

    from phoenix import Phoenix

    # plot a range of PT profiles for different temperatures
    temperatures = np.arange(2800, 4400, 200)
    loggs = [4.0, 5.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(temperatures)))
    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i, T in enumerate(temperatures):
        for j, logg in enumerate(loggs):
            ph = Phoenix(T, logg)
            
            label = f'logg = {logg}' if i == 0 else None
            if j == 0:
                ls = '-' 
            if j == 1:
                ls = '--'  
            if j == 2:
                ls = '-.'  
            ph.plot_PT(ax=ax, color=colors[i], ls=ls, label=label, alpha=0.7, lw=2.)

    # create colorbar
    cbar_im = ax.scatter([], [], c=[], cmap='plasma', vmin=temperatures.min(), vmax=temperatures.max())
    cbar = fig.colorbar(cbar_im, ax=ax, label='Temperature [K]')
    #ax.set_ylim(1e2, 1e-5) # region we are interested in
            
    #SPHINX PART

    def load_sphinx_model(Teff=3100.0, log_g=4.0, logZ=0.0, C_O=0.50):
        
        path = pathlib.Path('Sphinx files/')
        sign = '+' if logZ >= 0 else '-'
    
        # PT profile
        file = path / f'Teff_{Teff:.1f}_logg_{log_g}_logZ_{sign}{abs(logZ)}_CtoO_{C_O}_atms.txt'
        assert file.exists(), f'File {file} does not exist.'
        t, p = np.loadtxt(file, unpack=True)
    
        # VMRs
        file_chem = path / file.name.replace('atms', 'mixing_ratios')
    
        with open(file_chem, 'r') as f:
            header = f.readline()
        
        header = header.split(',')
        header[0] = 'pressure'
        # remove spaces
        header = [h.strip() for h in header]
        VMRs = np.loadtxt(file_chem, unpack=True)
        VMRs = {k:v for k, v in zip(header, VMRs)}
    
        return t, p, VMRs, file

    t_sphinx,p_sphinx, VMRs_sphinx, file_sphinx = load_sphinx_model()

    ax.plot(t_sphinx,p_sphinx,color='green', label='Sphinx T=3100K, logg=4.0')

    ax.legend()
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f' - Saved {fig_name}')
        plt.close(fig)
   
    return fig, ax





def simple_cornerplot(posterior, 
                      labels, 
                      bestfit_params=None,
                      fig=None, 
                      fig_name=None):
        
        
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
                            fig=fig,
                            )
        if bestfit_params is not None:
            corner.overplot_lines(fig, bestfit_params, color='green', lw=0.5)
        if fig_name is not None:
            fig.savefig(fig_name)
            print(f' - Saved {fig_name}')
            plt.close(fig)
            
        return fig
    
def fig_bestfit_model(d_spec,
                      m_spec,
                      LogLike,
                      Cov=None,
                      xlabel=r'Wavelength (nm)',
                      bestfit_color='C0',
                      ax_spec=None,
                      ax_res=None,
                      flux_factor=1.0,
                      fig_name=None,
                      **kwargs):
    
    if (ax_spec is None) and (ax_res is None):
        # Create a new figure
        is_new_fig = True
        n_orders = d_spec.n_orders

        fig, ax = plt.subplots(
            figsize=(10,2.5*n_orders*2), nrows=n_orders*3, 
            gridspec_kw={'hspace':0, 'height_ratios':[1,1/3,1/5]*n_orders, 
                        'left':0.1, 'right':0.95, 
                        'top':(1-0.02*7/(n_orders*3)), 
                        'bottom':0.035*7/(n_orders*3), 
                        }
            )
    else:
        is_new_fig = False
    if flux_factor == 1.0:
        ylabel_spec = r'$F_\lambda$'+'\n'+r'$(\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    else:
        ylabel_spec = r'$F_\lambda$'+'\n'+ f'$(\mathrm{{erg\ s^{{-1}}\ cm^{{-2}}\ nm^{{-1}}}}\cdot 10^{{-{np.log10(flux_factor):.0f}}})$'
   
    # Use the same ylim, also for multiple axes
    ylim_spec = (np.nanmean(d_spec.flux)-4*np.nanstd(d_spec.flux), 
                 np.nanmean(d_spec.flux)+4*np.nanstd(d_spec.flux)
                )
    ylim_res = (1/3*(ylim_spec[0]-np.nanmean(d_spec.flux)), 
                1/3*(ylim_spec[1]-np.nanmean(d_spec.flux))
                )
    # apply flux factor
    ylim_spec = (ylim_spec[0]*flux_factor, ylim_spec[1]*flux_factor)
    ylim_res = (ylim_res[0]*flux_factor, ylim_res[1]*flux_factor)


    lw = kwargs.get('lw', 0.5)
    for i in range(d_spec.n_orders):

        if is_new_fig:
            # Spectrum and residual axes
            ax_spec = ax[i*3]
            ax_res  = ax[i*3+1]

            # Remove the temporary axis
            ax[i*3+2].remove()

            # Use a different xlim for the separate figures
            xlim = (d_spec.wave[i,:].min()-0.5, 
                    d_spec.wave[i,:].max()+0.5)
        else:
            xlim = (d_spec.wave.min()-0.5, 
                    d_spec.wave.max()+0.5)

        ax_spec.set(xlim=xlim, xticks=[], 
                    # ylim=ylim_spec,
                    )
        ax_res.set(xlim=xlim, ylim=ylim_res)

        for j in range(d_spec.n_dets):
        
            x = d_spec.wave[i,j]
            mask_ij = d_spec.mask_isfinite[i,j]
            
            if np.sum(mask_ij) == 0:
                continue
            # if mask_ij.any():
            # Show the observed and model spectra
            # Cov = None
            if Cov is not None:
                if hasattr(Cov, 'diag'):
                    err = np.sqrt(Cov.diag[i,j]) * LogLike.beta[i,j] * flux_factor
                else:
                    # print(f' - Calculating error from dense covariance matrix order, det = {i}, {j}')
                    err = np.sqrt(np.diag(Cov[i,j].get_dense_cov())) * LogLike.beta[i,j] * flux_factor #
                    # print(f' shape Cov[i,j].cov = {Cov[i,j].cov.shape}')
                    # print(f' shape Cov[i,j].cov[0] = {Cov[i,j].cov[0].shape}')  
                    # err = np.sqrt(Cov[i,j].cov[0]) * LogLike.beta[i,j] * flux_factor
                    
            else:
                # err = d_spec.err[i,j,mask_ij] * LogLike.beta[i,j] * flux_factor
                err = np.ones_like(d_spec.flux[i,j,mask_ij]) * 0.01 * flux_factor # avoid calculating error for speed
            
            wave = d_spec.wave[i,j,mask_ij]
            flux = d_spec.flux[i,j,mask_ij] * flux_factor
            flux_full = d_spec.flux[i,j,:] * flux_factor
            err_full = np.nan * np.ones_like(flux_full)
                        
            mean_err = np.mean(err)
            err_full[mask_ij] = err
            
            m_flux_ij = m_spec.flux[i,j,mask_ij]

            
            ax_spec.plot(
                x, flux_full,
                c='k', lw=lw, label='Data'
                )
            ax_spec.fill_between(
                x, flux_full-err_full, flux_full+err_full, 
                fc='k', alpha=0.4, ec='none',
                )

            if hasattr(LogLike, 'chi_squared_red'):
                label = 'Best-fit model ' + \
                        r'$(\chi^2_\mathrm{red}$$=' + \
                        '{:.2f}'.format(LogLike.chi_squared_red) + \
                        r')$'
            else:
                label = 'Best-fit model'
                    
            f = LogLike.f[:,i,j]
            
            # M = LogLike.M[i,j]
            # linear_model = M * f[:,None] * flux_factor
            # set ~mask to np.nan on last axis
            # linear_model[:,~mask_ij] = np.nan
            
            # model = np.sum(linear_model, axis=0) # full fitted linear model
            # model = f * m_spec.flux[i,j] * flux_factor
            # model = f @ m_spec.flux_spline[:,i,j] if m_spec.N_knots > 1 else f * m_spec.flux[i,j]
            # model *= flux_factor
            model = LogLike.m[i,j,:] * flux_factor
            # model_spec = np.sum(LogLike.f[:N_knots,i,j] * LogLike.M[:N_knots,i,j,:], axis=0) # TODO:
            # model_veiling = np.sum(LogLike.f[-1,i,j] * LogLike.M[-1,i,j,:], axis=0)
            if m_spec.N_knots > 1:
                    
                # m_flux_ij_spline = SplineModel(N_knots=m_spec.N_knots, spline_degree=3)(m_flux_ij)
                
                # replace single-component matrix with multi-component matrix
                M_ij = SplineModel(N_knots=m_spec.N_knots, spline_degree=3)(m_flux_ij)
            else:
                M_ij = m_spec.flux[i,j,mask_ij][np.newaxis,:]
            
            model[~mask_ij] = np.nan
            ax_spec.plot(x, model, lw=lw, label=label, color=bestfit_color)
            N_veiling = getattr(m_spec, 'N_veiling', 0)
            if N_veiling > 0:
                # build linear model with veiling components
                N_pRT = len(f) - N_veiling
                M_ij = np.concatenate([M_ij, m_spec.M_veiling[:,mask_ij]], axis=0) # add veiling components
                
                # print(f' - N_pRT = {N_pRT}, N_veiling = {N_veiling}')
                # print(f' M_ij.shape = {M_ij.shape}, f.shape = {f.shape}')
                m_veiling, m_pRT = (np.nan * np.ones_like(x) for _ in range(2))
                # m_veiling[mask_ij] = f[N_pRT:] @ M_ij[N_pRT:]
                # m_pRT[mask_ij] = f[:N_pRT] @ M_ij[:N_pRT]
                m_pRT[mask_ij] = f[:N_pRT] @ M_ij[:N_pRT]
                m_veiling[mask_ij] = f[N_pRT:] @ M_ij[N_pRT:]
                ax_spec.plot(x, m_veiling, lw=lw, label='Veiling', color='magenta')
                ax_spec.plot(x, m_pRT, lw=lw, label='pRT', color='navy')
            if hasattr(m_spec, 'veiling_model'):
                try: # testing...
                    ax_spec.plot(x, m_spec.veiling_model[i,j,:], lw=lw, ls=':', 
                                 label='Veiling model' if (i==0 and j==0) else None,
                                 color='magenta')
                except:
                    pass

            # Plot the residuals
            res_ij = flux_full - model
            res_ij[~mask_ij] = np.nan
            ax_res.plot(x, res_ij, c='k', lw=lw)
            ax_res.plot(
                [x.min(), x.max()], 
                [0,0], c=bestfit_color, lw=1
            )

            ax_res.errorbar(
                wave.min()-0.2, 0, yerr=1*mean_err, 
                fmt='none', lw=1, ecolor='k', capsize=2, color='k', 
                )

            if i==0 and j==0:
                ax_spec.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )

    # Set the labels for the final axis
    ax_spec.set(ylabel=ylabel_spec)
    ax_res.set(xlabel=xlabel, ylabel='Res.')

    if fig_name is not None:
        plt.savefig(fig_name)
        print(f' - Saved {fig_name}')
        plt.close(fig)
        
    # else:
    return ax_spec, ax_res


def fig_prior_check(ret, fig_name='prior_check.pdf'):
    
    fig_PT_prior, ax_PT = plt.subplots(1,1,figsize=(7,7),tight_layout=True)

    theta = [0.0, 0.5, 1.0] # lower edge, center, upper edge
    colors = ['limegreen', 'b', 'r']

    f = []
    m = []
    m_veiling = []
    for i, theta_i in enumerate(theta):
        cube = theta_i * np.ones(ret.parameters.ndim)
        sample = ret.parameters(cube) # transform the cube to the parameter space
        ret.parameters.add_sample(sample) # add the sample to the parameters (create dictionary)

        
        # m_spec = ret.pRT_model(ret.parameters.params) # generate the model spectrum
        # m_spec.N_knots = ret.parameters.params.get('N_knots', 1)
        log_L = ret.PMN_lnL_func()

        # log_L = ret.loglike(m_spec, ret.Cov)
        sample_dict = dict(zip(ret.parameters.param_keys, sample))
        print(sample_dict)
        print(f' - log_L = {log_L:.1e}\n')
        m.append(ret.loglike.m) # store model
        if hasattr(ret.m_spec, 'veiling_model'):
            m_veiling.append(ret.m_spec.veiling_model) # store veiling model
        
        
        fig_PT_prior, ax_PT = fig_PT(ret.pRT_model.PT, ax_PT, fig=fig_PT_prior,
                       bestfit_color=colors[i],)
    if isinstance(fig_name, str):
        fig_name = pathlib.Path(fig_name)
    outfig_PT = fig_name.parent / (fig_name.stem + '_PT.pdf')
    fig_PT_prior.savefig(outfig_PT)
    print(f' - Saved {outfig_PT}')
    fig, ax = plt.subplots(2,1, figsize=(10,5), sharex=True, 
                            gridspec_kw={'height_ratios':[3,1],
                                        'top':0.97, 'bottom':0.1, 
                                        'hspace':0.05,
                                        'left':0.07, 
                                        'right':0.98})
    ax[0].set(ylabel='Normalized flux')
    ax[1].set(ylabel='Residuals', xlabel='Wavelength / nm')
    with PdfPages(fig_name) as pdf:
        for order in range(ret.d_spec.n_orders):
            
            ax[0].set(xlim=(ret.d_spec.wave[order].min()-0.1, 
                            ret.d_spec.wave[order].max()+0.1)
                        )
            lw = 1.0
            ax[1].axhline(0, c='k', lw=0.5)
            for det in range(ret.d_spec.n_dets):

                wave = ret.d_spec.wave[order,det]
                flux = ret.d_spec.flux[order,det]
                err = np.nan * np.ones_like(flux)
                mask = ret.d_spec.mask_isfinite[order,det]
                if np.sum(mask) == 0: # skip if no valid data points
                    continue
                err[mask] = ret.Cov[order,det].err
                
                ax[0].plot(wave, flux, c='k', lw=lw)
                ax[0].fill_between(wave, flux-err, flux+err, fc='k', alpha=0.2)
                ax[1].fill_between(wave, -err, err, fc='k', alpha=0.2)
                for i, theta_i in enumerate(theta):
                    print(i,order,det)
                    print(f)
                    print(f[i][:,order,det])
                    print()
                    m_spec = m[i][order,det]
                    ax[0].plot(wave, m_spec, c=colors[i], lw=lw, alpha=0.8)
                    
                    residuals = flux - m_spec
                    ax[1].plot(wave, residuals, c=colors[i], lw=lw, alpha=0.8)
                    if len(m_veiling) > 0:
                        ax[0].plot(wave, m_veiling[i][order,det], c=colors[i], ls=':', lw=lw, alpha=0.8)
                        
                    
            pdf.savefig(fig)
        # clear the axes
        ax[0].clear()
        ax[1].clear()
    plt.close(fig_PT_prior)
    plt.close(fig)
    
    print(f' - Saved {fig_name}')
    return fig_PT_prior, fig