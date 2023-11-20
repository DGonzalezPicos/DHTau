import numpy as np
import matplotlib.pyplot as plt
# plotting style
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.linewidth'] = 2.5
import pathlib
from scipy.interpolate import splrep, splev

from astropy import units as u
from astropy import constants as const

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import nat_cst as nc

# %% [markdown]
# ### 1. Pressure-Temperature profile

# %%
# Structure of the atmosphere
logP_max = 2.0
logP_min = -6.0
n_layers = 30 # plane-parallel layers
pressure = np.logspace(logP_min, logP_max, n_layers) # from top to bottom

# Knots for the spline interpolation
logP_knots = np.linspace(logP_max, logP_min, 4) # bottom to top
sort = np.argsort(logP_knots) 

# Pressure knots must be in increasing order
P_knots = 10.0**logP_knots
T_knots = np.array([6000., 2500., 1900., 1400.]) # bottom to top
# Spline decomposition and evaluation
knots, coeffs, deg = splrep(logP_knots[sort], np.log10(T_knots[sort]))
temperature = 10**splev(np.log10(pressure), (knots, coeffs, deg), der=0)

# TODO: Load SPHINX PT-profile


# Plot PT-profile (y-axis from bottom to top of the atmosphere)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(temperature, pressure, color='brown', lw=3.)
ax.plot(T_knots, P_knots, 'o', markersize=8, color='red')
ax.set(yscale='log', ylim=(pressure.max(), pressure.min()), 
       xlabel='Temperature [K]', ylabel='Pressure [bar]')
plt.show()

# %% [markdown]
# ### 2. Chemistry

# TODO: Load SPHINX chemistry (H2O, CO, Na)

# %%
# Define Volume-Mixing-Ratios (VMRs) of the line-species
VMRs = {}
VMRs['H2O'] = np.ones_like(pressure) * 1e-4
VMRs['12CO'] = np.ones_like(pressure) * 1.5e-4
VMR_He = np.ones_like(pressure) * 0.1
VMR_wo_H2 = VMR_He + VMRs['H2O'] + VMRs['12CO']

def vmr_to_mass_fractions(VMRs):
    """Converts VMRs to mass fractions"""
    mass_fractions = {}
    atomic_mass = dict(H=1.00794, He=4.002602, C=12.0107, O=15.9994)
    mass_fractions['H2O'] = VMRs['H2O'] * (atomic_mass['H'] * 2.0 + atomic_mass['O'])
    mass_fractions['12CO'] = VMRs['12CO'] * (atomic_mass['C'] + atomic_mass['O'])
    mass_fractions['He'] = VMR_He * atomic_mass['He']
    mass_fractions['H2'] = (1.0 - VMR_wo_H2) * atomic_mass['H'] * 2.0
    mass_fractions['H'] = atomic_mass['H'] * mass_fractions['H2'] + 2. * mass_fractions['H2O']

    # Mean Molecular Weight
    mass_fractions['MMW'] = np.sum([mass_fractions[key] for key in mass_fractions], axis=0)
    return mass_fractions

# Convert to mass fractions (required for petitRADTRANS)
mass_fractions = vmr_to_mass_fractions(VMRs)

# PLot vertical profiles of the VMRs (Free-Chemistry --> constant-with-altitude VMRs)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(temperature, pressure, color='brown', lw=3., alpha=0.2)
ax.plot(T_knots, P_knots, 'o', markersize=8, color='red', alpha=0.2)

ax_vmr = ax.twiny()
for key in mass_fractions:
    ax_vmr.plot(mass_fractions[key], pressure, label=key, lw=4., alpha=0.7,
                ls='--' if key == 'MMW' else '-')
ax_vmr.legend()
ax_vmr.set(yscale='log', xscale='log', ylim=(pressure.max(), pressure.min()), 
           xlim=(1e-6, 1e1), xlabel='Mass Fraction', ylabel='Pressure [bar]')


ax.set(yscale='log', ylim=(pressure.max(), pressure.min()), 
       xlabel='Temperature [K]', ylabel='Pressure [bar]')
plt.show()


# %% [markdown]
# ### 3. Radiative Transfer object
# * Load opacities of line-species
# * Calculate emission spectrum

# %%
# Define the line-list of each species
line_lists = {'H2O': 'H2O_pokazatel_main_iso',
              '12CO': 'CO_high',
              'Na':'Na_allard'}
# rename mass fractions with the line-list names
mass_fractions['H2O_pokazatel_main_iso'] = mass_fractions.pop('H2O')
mass_fractions['CO_high'] = mass_fractions.pop('12CO')


# %%
radtrans = Radtrans(line_species=list(line_lists.values()),
                    rayleigh_species=['H2', 'He'], 
                    continuum_opacities=['H2-H2', 'H2-He'],
                    mode='lbl',
                    lbl_opacity_sampling=5,
                    wlen_bords_micron=[2.32, 2.37],
                    )
# Pressure and temperature structure
radtrans.setup_opa_structure(pressure)

# %%
# Surface gravity in cgs units
gravity = 10.0**(4.0) # logg = 4.0 

radtrans.calc_flux(temperature, mass_fractions, gravity, mmw=mass_fractions['MMW'],
                   contribution=True)

contribution_function = radtrans.contr_em # shape (n_layers, n_wavelengths)

c = 2.99792458e5 # speed of light in km/s
wave = c / (radtrans.freq/1e9) # wavelength in micron
flux = radtrans.flux * c / (wave **2)

save_spectrum = True
outdir = pathlib.Path('data')
outdir.mkdir(exist_ok=True)

if save_spectrum:
    # save output as .npy file
    np.save('data/prt_spectrum.npy', np.array([wave, flux]).T)