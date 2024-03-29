import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter1d

import os
import copy
import corner

import petitRADTRANS.nat_cst as nc

from atm_retrieval.utils import quantiles, weigh_alpha

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

    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            
            mask = clip_mask[i,j]
            f_clip = np.where(mask, d_spec.flux[i,j], np.nan)
            f_clean  = np.where(~mask, d_spec.flux[i,j], np.nan) 
            ax[i].plot(d_spec.wave[i,j], f_clip, c='r', lw=0.5)
            # if overplot_array is not None:
            ax[i].plot(d_spec.wave[i,j], f_clean, c='k', lw=1, alpha=0.9)
        
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
        
    assert hasattr(PT, 'temperature_envelopes'), 'No temperature envelopes found'
    
    p = PT.pressure
    if hasattr(PT, 'int_contr_em'):
        if np.max(PT.int_contr_em) > 0.0: # additional check
            # Plot the integrated contribution emission
            ax_twin = ax.twiny()
            ax_twin.plot(
                PT.int_contr_em, p, 
                c='red', lw=1, alpha=0.4,
                )
            weigh_alpha(PT.int_contr_em, p, np.linspace(0,10000,p.size), ax, alpha_min=0.5, plot=True)
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
        
    ax.set(xlabel='Temperature (K)', ylabel='Pressure (bar)',
            ylim=(p.max(), p.min()), yscale='log',
            xlim=xlim,
            )
    #ax.legend(loc='upper right', fontsize=12)
    
    if fig_name is not None:
        fig.savefig(fig_name)
        print(f' - Saved {fig_name}')
    # fig.savefig(self.prefix+f'plots/{out_fig_prefix}_PT_profile_{out_fig_suffix}.pdf')
    # print(f' - Saved {self.prefix}plots/{out_fig_prefix}_PT_profile_{out_fig_suffix}.pdf')
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
    ylim_res = (1/5*(ylim_spec[0]-np.nanmean(d_spec.flux)), 
                1/5*(ylim_spec[1]-np.nanmean(d_spec.flux))
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
            if mask_ij.any():
                # Show the observed and model spectra
            
                if Cov is not None:
                    cov = Cov[i,j].get_dense_cov()
                    err = np.sqrt(np.diag(cov)) * LogLike.beta[i,j] * flux_factor
                else:
                    err = d_spec.err[i,j,mask_ij] * LogLike.beta[i,j] * flux_factor
                
                wave = d_spec.wave[i,j,mask_ij]
                flux = d_spec.flux[i,j,mask_ij] * flux_factor
                flux_full = d_spec.flux[i,j,:] * flux_factor
                err_full = np.nan * np.ones_like(flux_full)
                            
                mean_err = np.mean(err)
                err_full[mask_ij] = err

                
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
                    
            f = LogLike.f[i,j]
            # M = LogLike.M[i,j]
            # linear_model = M * f[:,None] * flux_factor
            # set ~mask to np.nan on last axis
            # linear_model[:,~mask_ij] = np.nan
           
            # model = np.sum(linear_model, axis=0) # full fitted linear model
            model = f * m_spec.flux[i,j] * flux_factor
            model[~mask_ij] = np.nan
            ax_spec.plot(x, model, lw=lw, label=label, color=bestfit_color)

            if mask_ij.any():

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
