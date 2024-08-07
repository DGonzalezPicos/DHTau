import numpy as np
import pathlib
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages

from atm_retrieval.parameters import Parameters
from atm_retrieval.retrieval import Retrieval
from atm_retrieval.utils import pickle_load, pickle_save


run = 'testing_030'
run_dir = pathlib.Path(f'retrieval_outputs/{run}')

parameters_file = run_dir / 'parameters.json'
print(f'--> Loading parameters from file {parameters_file}...')
parameters = Parameters.load(parameters_file)

# Load the retrieval object
d_spec = pickle_load(run_dir / 'd_spec.pickle')
pRT = pickle_load(run_dir / 'atm.pickle')
ret = Retrieval(parameters, d_spec, pRT, run=run)

# read retrieval results
posterior, bestfit_params = ret.PMN_analyzer()
keys = ret.parameters.param_priors.keys()
bestfit_params_dict = dict(zip(keys, bestfit_params))
# print(bestfit_params_dict)

# print()
# CO_ratio = np.exp(bestfit_params_dict['log_12CO'])/np.exp(bestfit_params_dict['log_13CO'])
# print(CO_ratio)
# print()



ret.parameters.add_sample(bestfit_params)
ret.PMN_lnL_func()
m_spec_0 =ret.m_spec

# now generate model without a given species
species = 'Ca'
# species = 'g'

params_dict = bestfit_params_dict.copy()
if species == 'g':
    params_dict['log_g'] = 5.0
else:
    params_dict[f'log_{species}'] -= 14.0
# generate the model

ret.parameters.add_sample(list(params_dict.values()))
ret.PMN_lnL_func()
m_spec_1 = ret.m_spec

n_orders, n_dets = d_spec.flux.shape[:2]

def plot_order_det(order, det, species, m_spec_0, m_spec_1, d_spec, out_fig):
    wave = d_spec.wave[order, det]
    m_0 = m_spec_0.flux[order, det] / np.nanmean(m_spec_0.flux[order, det])
    m_1 = m_spec_1.flux[order, det] / np.nanmean(m_spec_1.flux[order, det])
    d  = d_spec.flux[order, det] / np.nanmean(d_spec.flux[order, det])

    fig, ax = plt.subplots(3, 1, figsize=(10, 4),
                        sharex=True,
                        gridspec_kw={'left': 0.08, 
                                        'right': 0.98,
                                        'top': 0.97,
                                        'bottom': 0.13, 
                                        'hspace': 0.1,})

    # top panel: data and model
    ax[0].plot(wave, d, label='data', color='k', alpha=0.9, lw=2)
    ax[0].plot(wave, m_0, label='model', color='g', alpha=0.7, lw=2)
    ax[-1].plot(wave, d - m_0, label='data - model', color='g', alpha=0.7, lw=2)


    if np.allclose(m_0, m_1, rtol=1e-2):
        # print('Model without species is equal to model with species')
        return None
    ax[0].plot(wave, m_1, label=f'model without {species}', color='b', alpha=0.7, lw=2)

    # middle panel: model of species only
    ax[1].plot(wave, m_0 - m_1, label=f'{species}', color='r', alpha=0.9, lw=2)
    ax[1].legend()

    # bottom panel: residuals
    ax[-1].plot(wave, d - m_1, label=f'data - model without {species}', color='b', alpha=0.7, lw=2)
    ax[-1].axhline(0, color='k', ls='-', alpha=0.7)

    ax[0].set_ylabel('Normalized flux')
    ax[-1].set(xlabel='Wavelength [nm]', ylabel='Residuals')
    ax[0].legend()
    return None

def plot_spec_order_det(order, det, species, m_spec, d_spec, out_fig):

    # print(d_spec.wave[order, 0])
    # print(d_spec.wave[order, :])


    if det == ':':
        wave = d_spec.wave[order, :].flatten()
        m_0 = (m_spec.flux[order, :] / np.nanmean(m_spec.flux[order, :])).flatten()
        # m_1 = m_spec_1.flux[order, det] / np.nanmean(m_spec_1.flux[order, det])
        d  = (d_spec.flux[order, :] / np.nanmean(d_spec.flux[order, :])).flatten()
    else:
        wave = d_spec.wave[order, det]
        m_0 = m_spec.flux[order, det] / np.nanmean(m_spec.flux[order, det])
        # m_1 = m_spec_1.flux[order, det] / np.nanmean(m_spec_1.flux[order, det])
        d  = d_spec.flux[order, det] / np.nanmean(d_spec.flux[order, det])

    fig, ax = plt.subplots(2, 1, figsize=(14, 6),
                        sharex=True,
                        gridspec_kw={'left': 0.06, 
                                        'right': 0.98,
                                        'top': 0.97,
                                        'bottom': 0.1, 
                                        'hspace': 0.1,
                                        'height_ratios':[3/4,1/4]})

    # top panel: data and model
    ax[0].plot(wave, d, label='data', color='k', alpha=0.9, lw=2)
    ax[0].plot(wave, m_0, label='model', color='g', alpha=0.7, lw=2)
    ax[-1].plot(wave, d - m_0, label='data - model', color='g', alpha=0.7, lw=2)


    # if np.allclose(m_0, m_1, rtol=1e-2):
    #     # print('Model without species is equal to model with species')
    #     return None
    # ax[0].plot(wave, m_1, label=f'model without {species}', color='b', alpha=0.7, lw=2)

    # # middle panel: model of species only
    # ax[1].plot(wave, m_0 - m_1, label=f'{species}', color='r', alpha=0.9, lw=2)
    # ax[1].legend()

    # bottom panel: residuals
    # ax[-1].plot(wave, d - m_0, label=f'data - model', color='b', alpha=0.7, lw=2)
    ax[-1].axhline(0, color='k', ls='-', alpha=0.7)

    ax[0].set_ylabel('Normalized flux')
    ax[-1].set(xlabel='Wavelength [nm]', ylabel='Residuals')
    ax[0].legend()
    return None


# out_fig = run_dir / f'plots/species_contribution_{species}.pdf'
out_fig = run_dir / f'plots/bestfit_spec_4.pdf'

with PdfPages(out_fig) as pdf:\

    plot_spec_order_det(4, ':', species, m_spec_0, d_spec, out_fig)
    pdf.savefig()
    plt.close()
    
    # for i in range(n_orders):
    #     for j in range(n_dets):
    #         plot_spec_order_det(i, j, species, m_spec_0, d_spec, out_fig)
    #         pdf.savefig()
    #         plt.close()


# 3,2 three broadened peaks
# 1,1 first Ca peak

print(f'Plots saved to {out_fig}')


