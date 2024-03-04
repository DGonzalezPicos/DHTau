''' Useful functions for manipulating spectra and pRT models'''
import numpy as np

from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening


### SPECTRUM MANIPULATION FUNCTIONS ###
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

def rebin(wave, flux, wave_new):
    return np.interp(wave_new, wave, flux)

def CRIRES_grid():
    '''Architechture of the CRIRES+ spectral orders and detectors
    shape = (n_orders, n_detectors) = (7, 3))'''
    wave_grid = np.array([
        [[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
        [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
        [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
        [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
        [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
        [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
        [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]],
        ])
    return wave_grid


### pRT MODEL ###


def spectrally_weighted_integration(wave, flux, array):

    # Integrate and weigh the array by the spectrum
    integral1 = np.trapz(wave*flux*array, wave)
    integral2 = np.trapz(wave*flux, wave)

    return integral1/integral2

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