import numpy as np
import matplotlib.pyplot as plt
import pathlib

from petitRADTRANS import Radtrans

from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening

import I_functions as I

"""
Variables:
Teff = 3900.0
log_g = 4.0
logZ = 0.0
C_O = 0.5
v_sini = 5.0
in_res = 1e6 / 5
crires_res = 1e5
min_wave_crires, max_wave_crires = 2338.68, 2353.964
radius = 10.0 # Rjup
parallax_mas = 500.
SNR = 50.0

wave_crires.size == 2048 ??
line_lists??
"""
c = 2.99792458e5 # speed of light in km/s

# defining sphinx variables
Teff = 3900.0 # K
log_g = 4.0
logZ = 0.0
C_O = 0.5

# setting 'fake' parameters
v_sini = 5.0 # km/s
radius = 10.0 # Rjup
parallax_mas = 500.
SNR = 50.0

# Defining crires variables
in_res = 1e6 / 5 # 5 is the `lbl_opacity_sampling` used in `00D_pRT_quickstart.py`
crires_resolution = 1e5 # CRIRES+ resolution
min_wave_crires, max_wave_crires = 2338.68, 2353.964

# loading the pt profile and chemical abundances
t, p, chemistry,file = I.load_sphinx_model(Teff=Teff, log_g=log_g, logZ=logZ, C_O=C_O)

#I.plot_pt_chem(t,p,chemistry,'plots/pt_chem.png')

pressure = chemistry['pressure']
temperature = np.interp(pressure, p, t)

# calculating the mass fractions
line_lists = {'H2O': 'H2O_pokazatel_main_iso',
                    '12CO': 'CO_high',
                    'Na':'Na_allard'
                    }
mass_fractions = I.calculate_mass_fracs(pressure,chemistry,line_lists)

# creating the prt spectrum
radtrans = Radtrans(line_species=list(line_lists.values()),
                    rayleigh_species=['H2', 'He'], 
                    continuum_opacities=['H2-H2', 'H2-He'],
                    mode='lbl',
                    lbl_opacity_sampling=5,
                    wlen_bords_micron=[2.32, 2.37],
                    )

radtrans.setup_opa_structure(pressure)

radtrans.calc_flux(temperature, mass_fractions, 10.0**(log_g), mmw=mass_fractions['MMW'],
                   contribution=True)

contribution_function = radtrans.contr_em # shape (n_layers, n_wavelengths)

# Calculating the wavelenghts and fluxes
wave = c / (radtrans.freq/1e9) # wavelength in micron
flux = radtrans.flux * c / (wave **2)

# Applying rotational broadening
wave_even, flux_rotbroad_even = I.rot_broadening(wave, flux, v_sini)
# Rebin the spectrum to the original wavelength grid 
flux_rotbroad = I.rebin(wave_even, flux_rotbroad_even, wave)

# Applying intrumental broadening
flux_LSF = I.instr_broadening(wave, flux_rotbroad, out_res=crires_resolution, in_res=in_res)

# Adjusting the grid to fit match crires data 
wave_crires_full, flux_crires, err_crires = np.loadtxt('data/crires_example_spectrum.dat').T
# let's use only one (order,detector) for simplicity (order 5, detector 2)
wave_crires = wave_crires_full[(wave_crires_full >= min_wave_crires) & (wave_crires_full <= max_wave_crires)]
wave_crires *= 1e-3 # convert to micron
assert wave_crires.size == 2048, 'CRIRES+ wavelength grid has changed!'
flux_LSF = I.rebin(wave, flux, wave_crires)

flux_LSF = I.scale_flux(flux_LSF, radius, parallax_mas)

# add Gaussian noise (SNR = 50)
# err = np.random.normal(0, 1/SNR, size=wave_crires.size)
mean_err = np.mean(flux_LSF) / SNR
err = np.random.normal(0, mean_err, size=wave_crires.size)
# synthetic observed spectrum
flux_syn = flux_LSF + np.random.normal(0, mean_err, size=wave_crires.size)

# plotting the spectrum
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(wave_crires, flux_syn/np.mean(flux_syn), color='black', lw=2)
ax.set(xlabel='Wavelengt [micron]', ylabel='Flux [erg s^-1 cm^-3]',
       #xlim=(2.342, 2.346),
       )
plt.savefig('plots/sim_spectrum.png')
plt.show()

#saving the spectrum
outdir = pathlib.Path('data')
outdir.mkdir(exist_ok=True)
header = 'wave [micron], flux [erg/s/cm2/cm], err [erg/s/cm2/cm], flux_model [erg/s/cm2/cm]'
array_out = np.array([wave_crires, flux_syn, err, flux_LSF])
np.savetxt('data/prt_sim_spectrum.txt', array_out.T, header=header)
print(f'- Saved {outdir}/prt_sim_spectrum.txt')






