import numpy as np
import matplotlib.pyplot as plt

from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening

## Load petitRADTRANS spectrum from `00_pRT_quickstart.py`
prt = np.load('data/prt_spectrum.npy')
wave = prt[:, 0]
flux = prt[:, 1]

## Doppler shift the spectrum
# Define the radial velocity in km/s (sum of barycentric and systemic velocity)
v_rad = 20.0 # km/s
# Shift the spectrum
wave_shifted = wave * (1.0 - v_rad/2.99792458e5)

## Rotational broadening
# the projected spin of the object is `vsini` (v * sin(i)) where `i`` is the inclination
# the linear limb darkening parameter is `epsilon_limb` ---> ignore for now
vsini = 10.0 # km/s
# epsilon_limb = 0.65 # between 0 and 1
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

# Apply rotational broadening
wave_even, flux_rotbroad_even = rot_broadening(wave, flux, vsini)
# Rebin the spectrum to the original wavelength grid 
flux_rotbroad = rebin(wave_even, flux_rotbroad_even, wave)

## Instrumental broadening
# The instrumental broadening is a Gaussian with a FWHM matching the spectral resolution
# of the instrument

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

# The input resolution `in_res` is 1e6 / `lbl_opacity_sampling`
in_res = 1e6 / 5 # 5 is the `lbl_opacity_sampling` used in `00D_pRT_quickstart.py`
crires_resolution = 1e5 # CRIRES+ resolution
flux_LSF = instr_broadening(wave, flux_rotbroad, out_res=crires_resolution, in_res=in_res)


## Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(wave, flux, label='Original', lw=2, alpha=0.4, color='k')
ax.plot(wave_shifted, flux, label=f'Shifted (RV={v_rad:.1f} km/s)', lw=2, alpha=0.6, color='b')
ax.plot(wave, flux_rotbroad, label=f'RotBroad (vsini = {vsini:.1f} km/s)', lw=2, alpha=1.0, color='g')
ax.plot(wave, flux_LSF, label=f'InstrBroad (R = {crires_resolution:.0f})', lw=2, alpha=1.0, color='r', ls='--')

ax.set(xlabel='Wavelength [um]', ylabel='Flux []', 
       xlim=np.percentile(wave, [48., 52.]))
ax.legend()
plt.show()

## Additional comments:
# * The instrumental broadening of CRIRES+ is very small compared to the rotational broadening when vsini > 10 km/s
# * Important to apply rotational broadening before instrumental broadening
# * In retrievals we fit for [v_rad, vsini, epsilon_limb]