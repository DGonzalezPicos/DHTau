import numpy as np
import matplotlib.pyplot as plt
import pathlib


prt = np.load('data/prt_spectrum_shifted.npy')
wave = prt[:, 0]
flux = prt[:, 1]

wave_crires_full, flux_crires, err_crires = np.loadtxt('data/crires_example_spectrum.dat').T

# let's use only one (order,detector) for simplicity (order 5, detector 2)
min_wave_crires, max_wave_crires = 2338.68, 2353.964
wave_crires = wave_crires_full[(wave_crires_full >= min_wave_crires) & (wave_crires_full <= max_wave_crires)]
wave_crires *= 1e-3 # convert to micron
assert wave_crires.size == 2048, 'CRIRES+ wavelength grid has changed!'

def rebin(wave, flux, wave_new):
    return np.interp(wave_new, wave, flux)

flux_LSF = rebin(wave, flux, wave_crires)

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

radius = 10.0 # Rjup
parallax_mas = 500.

flux_LSF = scale_flux(flux_LSF, radius, parallax_mas)

# add Gaussian noise (SNR = 100)
SNR = 50
# err = np.random.normal(0, 1/SNR, size=wave_crires.size)
mean_err = np.mean(flux_LSF) / SNR
err = np.random.normal(0, mean_err, size=wave_crires.size)
# synthetic observed spectrum
flux_syn = flux_LSF + np.random.normal(0, mean_err, size=wave_crires.size)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(wave_crires, flux_syn/np.mean(flux_syn), color='black', lw=2)
ax.set(xlabel='Wavelengt [micron]', ylabel='Flux [erg s^-1 cm^-3]',
       #xlim=(2.342, 2.346),
       )
plt.savefig('plots/fake_spectrum.png')
plt.show()

outdir = pathlib.Path('data')
outdir.mkdir(exist_ok=True)

header = 'wave [micron], flux [erg/s/cm2/cm], err [erg/s/cm2/cm], flux_model [erg/s/cm2/cm]'
array_out = np.array([wave_crires, flux_syn, err, flux_LSF])
np.savetxt('data/prt_fake_spectrum.txt', array_out.T, header=header)
print(f'- Saved {outdir}/prt_fake_spectrum.txt')