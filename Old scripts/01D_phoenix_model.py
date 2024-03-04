import numpy as np
import matplotlib.pyplot as plt

from expecto import get_spectrum


## Load petitRADTRANS spectrum from `00_pRT_quickstart.py`
prt = np.load('data/prt_spectrum.npy')
prt_wave = prt[:, 0]
prt_flux = prt[:, 1]

## PHOENIX spectrum
# Define stellar parameters for M0-type star
Teff = 3700.
log_g = 4.0
# Get PHOENIX spectrum
phoenix = get_spectrum(Teff, log_g, cache=True)
phoenix_wave = phoenix.spectral_axis.to('um').value
mask = (phoenix_wave > prt_wave.min()) & (phoenix_wave < prt_wave.max())
phoenix_wave = phoenix_wave[mask]
phoenix_flux = phoenix.flux.value[mask]

## Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(prt_wave, prt_flux / np.mean(prt_flux), label='PRT', lw=2)
ax.plot(phoenix_wave, phoenix_flux / np.mean(phoenix_flux), label='PHOENIX', lw=2)
ax.legend()
ax.set(xlabel='Wavelength [um]', ylabel='Flux [erg/s/cm2/um]')
plt.show()

## Do they match?
# Not really? --> Why?
# * Different chemistry --> Equilibrium chemistry
# * Additional opacity sources --> Atoms (Na, Ti, Fe, Si, etc)
# * Different temperature structure --> Radiative-convective equilibrium