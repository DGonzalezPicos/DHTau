import numpy as np
import matplotlib.pyplot as plt
import pathlib

from petitRADTRANS import Radtrans

from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening

def load_sphinx_model(Teff=3100.0, log_g=4.0, logZ=0.0, C_O=0.50):
        
    path = pathlib.Path('data/')
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

def calculate_mass_fracs(pressure,chemistry,line_lists):
    # calculating the VMR's
    species = ['H2H2', 'H2He', 'H2O', 'CO', 'Na'
                #   'Na', 'Mg',
                #   'Ca', 'Ti','Fe' # ignore for now... to keep it simple
                ]

    VMRs = VMRs = {k:v for k, v in chemistry.items() if k in species}
    atomic_mass = dict(H=1.00794, He=4.002602, C=12.0107, O=15.9994, Na=22.9898)

    VMRs['H2'] = VMRs.pop('H2H2')
    VMRs['He'] = VMRs.pop('H2He')
    VMRs['12CO'] = VMRs.pop('CO')

    # Determining the mass fractions
    mass_fractions = vmr_to_mass_fractions(VMRs)

    # rename mass fractions with the line-list names
    for key, value in line_lists.items():
        mass_fractions[value] = mass_fractions.pop(key)

    return mass_fractions

def plot_pt_chem(t,p,chemistry,figname='plots/pt_chem.png'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t, p, color='brown', lw=3.)

    ax_vmr = ax.twiny()

    my_species = ['H2H2', 'H2He', 'H2O', 'CO', 'Na']
    for key in chemistry:
        if key == 'pressure':
            continue

        if key not in my_species:
            continue
        ax_vmr.plot(chemistry[key], chemistry['pressure'], label=key,
                    lw=3., alpha=0.5)

    ax.set(yscale='log', ylim=(p.max(), p.min()), 
           xlabel='Temperature [K]', ylabel='Pressure [bar]')
    ax_vmr.set(xscale='log', yscale='log',
               ylim=(p.max(), p.min()),
               xlim=(1e-8, 1e0),
                  xlabel='Volume Mixing Ratio', ylabel='Pressure [bar]')
    ax_vmr.legend()
    plt.savefig(figname)
    plt.show()

def vmr_to_mass_fractions(vmr):
    
    atomic_mass = {'H':1.00794, 'He':4.002602, 'C':12.0107, 'O':15.9994,
                   'Na':22.98976928, 'Ca':40.078, 'Mg':24.305, 'Ti':47.867, 'Fe':55.845,
                   'H2':2.0, 'H2O':18.01528, '12CO':28.0101
                   }
    mass_fractions = {}
    for key in vmr:
        mass_fractions[key] = vmr[key] * atomic_mass[key]
    
    # Mean Molecular Weight
    mass_fractions['MMW'] = np.sum([mass_fractions[key] for key in mass_fractions], axis=0)
    return mass_fractions

def rot_broadening(wave, flux, vsini, epsilon_limb=0.0):
    
    # Make the wavelength grid even (requirement for `fastRotBroad`)
    wave_even = np.linspace(wave.min(), wave.max(), wave.size)
    flux_even = np.interp(wave_even, wave, flux) # linear interpolation
    
    # Rotational broadening of the model spectrum
    flux_rot_broad = pyasl.fastRotBroad(wave_even, flux_even, 
                                        epsilon=epsilon_limb, 
                                        vsini=vsini
                                        )
    return wave_even, flux_rot_broad

def rebin(wave, flux, wave_new):
    return np.interp(wave_new, wave, flux)

def instr_broadening(wave, flux, out_res=1e6, in_res=1e6):

    # Delta lambda of resolution element is FWHM of the LSF's standard deviation
    sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / \
                (2*np.sqrt(2*np.log(2)))

    spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    
    # Apply gaussian filter to broaden with the spectral resolution
    # LSF = Line Spread Function
    flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, 
                                mode='nearest'
                                )
    return flux_LSF

def scale_flux(flux, R_p, parallax_mas):
    '''Scale the flux to observation by scaling with planetary radius and parallax (distance)
    
    Parameters
    ----------
        flux : array
            flux array
        R_p : float
            planetary radius in Jupiter radii
        parallax_mas : float
            parallax in milliarcseconds'''
            
    pc = 3.085677581e18 # 1 parsec in cm
    r_jup = 7.1492e9 # Jupiter radius in cm
    
    distance = 1. /(parallax_mas * 1e-3) # distance in pc
    
    flux *= ((R_p*r_jup) / (distance*pc))**2
    return flux








