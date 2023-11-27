import numpy as np
import matplotlib.pyplot as plt
import pathlib

outdir = pathlib.Path('data')
outdir.mkdir(exist_ok=True)

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


t, p, VMRs_sphinx, file = load_sphinx_model(Teff=3100.0, log_g=4.0, logZ=0.0, C_O=0.50)



my_species = ['H2H2', 'H2He', 'H2O', 'CO', 
            #   'Na', 'Mg',
            #   'Ca', 'Ti','Fe' # ignore for now... to keep it simple
              ]
pressure = VMRs_sphinx['pressure']

# Save the volume mixing ratios in format required by petitRADTRANS
# change H2H2 to H2 and H2He to He
VMRs = {k:v for k, v in VMRs_sphinx.items() if k in my_species}
VMRs['H2'] = VMRs.pop('H2H2')
VMRs['He'] = VMRs.pop('H2He')
VMRs['12CO'] = VMRs.pop('CO')

## The PT profile and the VMRs have different pressure grids.
# --> We interpolate the PT profile to the VMR pressure grid.

temperature = np.interp(pressure, p, t)
# save file with interpolated PT profile and VMRs   
outfile = file.parent / file.name.replace('atms', 'model')
header = 'temperature [K], pressure [bar], '

array_out = np.array([temperature, pressure])
for key in VMRs:
    array_out = np.vstack([array_out, VMRs[key]])
    header += f'{key}, '
header = header[:-2] # remove last comma
np.savetxt(outfile, array_out.T, header=header)
print(f'- Saved {outfile}')

## Generate radiative transfer model
line_lists = {'H2O': 'H2O_pokazatel_main_iso',
              '12CO': 'CO_high',
            #   'Na':'Na_allard',
            #   'Mg':'Mg',
            #   'Ca':'Ca',
            #   'Ti':'Ti',
            #   'Fe':'Fe'
              }
# convert VMRs to mass fractions
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

mass_fractions = vmr_to_mass_fractions(VMRs)
# rename mass fractions with the line-list names
for key, value in line_lists.items():
    mass_fractions[value] = mass_fractions.pop(key)
        
        
        
from petitRADTRANS import Radtrans
radtrans = Radtrans(line_species=list(line_lists.values()),
                    rayleigh_species=['H2', 'He'], 
                    continuum_opacities=['H2-H2', 'H2-He'],
                    mode='lbl',
                    lbl_opacity_sampling=5,
                    wlen_bords_micron=[2.338, 2.355],
                    )
# Pressure and temperature structure
radtrans.setup_opa_structure(pressure)

# Surface gravity in cgs units
gravity = 10.0**(4.0) # logg = 4.0 

radtrans.calc_flux(temperature, mass_fractions, gravity, mmw=mass_fractions['MMW'],
                   contribution=True)

contribution_function = radtrans.contr_em # shape (n_layers, n_wavelengths)
# spectrally integrated contribution function
def spectrally_weighted_integration(wave, flux, array):

    # Integrate and weigh the array by the spectrum
    integral1 = np.trapz(wave*flux*array, wave)
    integral2 = np.trapz(wave*flux, wave)

    return integral1/integral2

c = 2.99792458e5 # speed of light in km/s
wave = c / (radtrans.freq/1e9) # wavelength in micron
flux = radtrans.flux * c / (wave **2)

int_contr = spectrally_weighted_integration(wave, flux, contribution_function)

### convert into a synthetic (=fake) observed spectrum
from D_functions import rot_broadening, instr_broadening, rebin, scale_flux

wave_crires_full, flux_crires, err_crires = np.loadtxt('data/crires_example_spectrum.dat').T

# let's use only one (order,detector) for simplicity (order 5, detector 2)
min_wave_crires, max_wave_crires = 2338.68, 2353.964
wave_crires = wave_crires_full[(wave_crires_full >= min_wave_crires) & (wave_crires_full <= max_wave_crires)]
wave_crires *= 1e-3 # convert to micron
assert wave_crires.size == 2048, 'CRIRES+ wavelength grid has changed!'

# 1) Rotational Broadening
vsini = 5.0 # km/s
wave_even, flux_rotbroad_even = rot_broadening(wave, flux, vsini)
# Instrumental broadening
# The input resolution `in_res` is 1e6 / `lbl_opacity_sampling`
in_res = 1e6 / 5 # 5 is the `lbl_opacity_sampling` used in `00D_pRT_quickstart.py`
crires_resolution = 1e5 # CRIRES+ resolution
flux_LSF = instr_broadening(wave, flux_rotbroad_even, out_res=crires_resolution, in_res=in_res)
# Rebin the spectrum to the *observed* wavelength grid
flux_LSF = rebin(wave_even, flux_LSF, wave_crires)

# Scale the spectrum to the observed spectrum
radius = 10.0 # Rjup
# distance = 50.0 # pc
# parallax in miliseconds of arc

parallax_mas = 500.
flux_LSF = scale_flux(flux_LSF, radius, parallax_mas)


# add Gaussian noise (SNR = 100)
SNR = 50
# err = np.random.normal(0, 1/SNR, size=wave_crires.size)
mean_err = np.mean(flux_LSF) / SNR
err = np.random.normal(0, mean_err, size=wave_crires.size)
# synthetic observed spectrum
flux_syn = flux_LSF + np.random.normal(0, mean_err, size=wave_crires.size)


## Plot spectrum, PT and VMRs

fig, ax = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 1.5]})

flux_scaled = scale_flux(flux, radius, parallax_mas)
ax[0].plot(wave, flux_scaled, lw=2, alpha=0.4, color='k') # output of pRT (no broadening, no noise)
ax[0].plot(wave_crires, flux_syn, lw=2, alpha=0.6, color='g') # with noise
ax[0].fill_between(wave_crires, flux_syn-err, flux_syn+err, color='g', alpha=0.2)

ax[1].plot(temperature, pressure, color='brown', lw=3., alpha=0.9)
# plot filled contribution function
ax_contr = ax[1].twiny()
ax_contr.plot(int_contr, pressure, color='k', lw=3., alpha=0.7)
ax_contr.fill_betweenx(pressure, int_contr, 0, color='k', alpha=0.1)

ax_vmr = ax[1].twiny()

for key in VMRs:
    
    ax_vmr.plot(VMRs[key], pressure, label=key,lw=3., alpha=0.5)
    

ax[0].set(xlabel='Wavelength [micron]', ylabel='Flux [erg/s/cm$^2$/cm]',
          xlim=(wave.min(), wave.max()))
ax[1].set(yscale='log', ylim=(p.max(), p.min()), 
       xlabel='Temperature [K]', ylabel='Pressure [bar]')
ax_vmr.set(xscale='log', yscale='log',
           ylim=(p.max(), p.min()),
           xlim=(1e-8, 1e0),
              xlabel='Volume Mixing Ratio', ylabel='Pressure [bar]')
ax_vmr.legend()

fig.savefig('data/prt_sphinx_spectrum.png', bbox_inches='tight', dpi=200)
plt.show()

# save_spectrum = True
# if save_spectrum:
#     # save output as .npy file
#     np.save('data/prt_sphinx_spectrum.npy', np.array([wave, flux]).T)
#     print(f'- Saved {outdir}/prt_sphinx_spectrum.npy')
# save generated spectrum as .txt with columns (wave, flux, err, wave_full, flux_LSF)
header = 'wave [micron], flux [erg/s/cm2/cm], err [erg/s/cm2/cm], flux_model [erg/s/cm2/cm]'
array_out = np.array([wave_crires, flux_syn, err, flux_LSF])
np.savetxt('data/prt_sphinx_spectrum.txt', array_out.T, header=header)
print(f'- Saved {outdir}/prt_sphinx_spectrum.txt')