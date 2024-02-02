import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter1d

import os
import copy
import corner

import petitRADTRANS.nat_cst as nc

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
                ax[i].plot(d_spec.wave[i,j], overplot_array[i,j], c='magenta', lw=1)
        
        ax[i].set(xlim=(d_spec.order_wlen_ranges[i].min()-0.5, 
                        d_spec.order_wlen_ranges[i].max()+0.5)
                  )

    
    fig_name = fig_name if fig_name is not None else prefix+f'plots/spec_to_fit_{w_set}.pdf'
    plt.savefig(fig_name)
    print(f' Figure saved as {fig_name}')
    #plt.show()
    plt.close(fig)