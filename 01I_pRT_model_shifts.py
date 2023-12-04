import numpy as np
import matplotlib.pyplot as plt
import pathlib

from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening

prt = np.load('data/prt_spectrum.npy')
wave = prt[:, 0]
flux = prt[:, 1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(wave, flux/np.mean(flux), color='black', lw=3.)
ax.set(xlabel='Wavelengt [micron]', ylabel='Flux [erg s^-1 cm^-3]',
       xlim=(2.342, 2.346),
       )

#rotational broadening
v_sini = 5.0 # km/s
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

wave_even, flux_rotbroad_even = rot_broadening(wave, flux, v_sini)

def rebin(wave, flux, wave_new):
    return np.interp(wave_new, wave, flux)

# Apply rotational broadening
wave_even, flux_rotbroad_even = rot_broadening(wave, flux, v_sini)
# Rebin the spectrum to the original wavelength grid 
flux_rotbroad = rebin(wave_even, flux_rotbroad_even, wave)

in_res = 1e6 / 5 # 5 is the `lbl_opacity_sampling` used in `00D_pRT_quickstart.py`
crires_resolution = 1e5 # CRIRES+ resolution
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

flux_LSF = instr_broadening(wave, flux_rotbroad, out_res=crires_resolution, in_res=in_res)

ax.plot(wave, flux_LSF/np.mean(flux_LSF), color='red', lw=3.)
plt.savefig('plots/spectrum_shift.png')
plt.show()

save_spectrum = True
outdir = pathlib.Path('data')
outdir.mkdir(exist_ok=True)

if save_spectrum:
    # save output as .npy file
    np.save('data/prt_spectrum_shifted.npy', np.array([wave, flux]).T)
