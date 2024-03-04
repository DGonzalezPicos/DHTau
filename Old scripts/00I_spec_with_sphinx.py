import numpy as np
import matplotlib.pyplot as plt
import pathlib

from petitRADTRANS import Radtrans

#from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening

#loading the sphinx model fot the pt chemistry
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

t, p, chemistry,file = load_sphinx_model(Teff=3900.0, log_g=4.0, logZ=0.0, C_O=0.50)

#plotting the pt chemistry
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
plot_pt_chem(t,p,chemistry,'plots/pt_chem.png')

species = ['H2H2', 'H2He', 'H2O', 'CO', 
            #   'Na', 'Mg',
            #   'Ca', 'Ti','Fe' # ignore for now... to keep it simple
            ]
pressure = chemistry['pressure']

VMRs = VMRs = {k:v for k, v in chemistry.items() if k in species}
atomic_mass = dict(H=1.00794, He=4.002602, C=12.0107, O=15.9994, Na=22.9898)

VMRs['H2'] = VMRs.pop('H2H2')
VMRs['He'] = VMRs.pop('H2He')
VMRs['12CO'] = VMRs.pop('CO')

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


line_lists = {'H2O': 'H2O_pokazatel_main_iso',
              '12CO': 'CO_high',
            #   'Na':'Na_allard'
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

radtrans = Radtrans(line_species=list(line_lists.values()),
                    rayleigh_species=['H2', 'He'], 
                    continuum_opacities=['H2-H2', 'H2-He'],
                    mode='lbl',
                    lbl_opacity_sampling=5,
                    wlen_bords_micron=[2.32, 2.37],
                    )

radtrans.setup_opa_structure(pressure)
gravity = 10.0**(4.0) 

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

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(wave, flux/np.mean(flux), color='black', lw=3.)
ax.set(xlabel='Wavelengt [micron]', ylabel='Flux [erg s^-1 cm^-3]',
       xlim=(2.342, 2.346),
       )
plt.savefig('plots/spectrum.png')
plt.show()


"""
mass_fracs = {}
mass_fracs['H2O'] = chemistry['H2O'] * (atomic_mass['H'] * 2. + atomic_mass['O'])
mass_fracs['12CO'] = chemistry['CO'] * (atomic_mass['C'] + atomic_mass['O'])

mass_fracs['H2H2'] = chemistry['H2H2'] * atomic_mass['H'] * 4.
mass_fracs['H2He'] = chemistry['H2He'] * (atomic_mass['He'] + atomic_mass['H'] * 2.)
mass_fracs['Na'] = chemistry['Na'] * atomic_mass['Na']

mass_fracs['MMW'] = np.sum([mass_fracs[key] for key in mass_fracs], axis=0)

print(len(chemistry['pressure']))

# PLot vertical profiles of the VMRs (Free-Chemistry --> constant-with-altitude VMRs)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t, p, color='brown', lw=3., alpha=0.2)

ax_vmr = ax.twiny()
for key in mass_fracs:
    ax_vmr.plot(mass_fracs[key], p, label=key, lw=4., alpha=0.7,
                ls='--' if key == 'MMW' else '-')
ax_vmr.legend()
ax_vmr.set(yscale='log', xscale='log', ylim=(p.max(), p.min()), 
           xlim=(1e-6, 1e1), xlabel='Mass Fraction', ylabel='Pressure [bar]')


ax.set(yscale='log', ylim=(p.max(), p.min()), 
       xlabel='Temperature [K]', ylabel='Pressure [bar]')
plt.savefig('plots/pt_massfracs.png')
plt.show()







#loading the prt spectrum
prt = np.load('data/prt_spectrum.npy')
wave = prt[:, 0]
flux = prt[:, 1]

# Define the radial velocity in km/s (sum of barycentric and systemic velocity)
v_rad = 20.0 # km/s
# Shift the spectrum
wave_shifted = wave * (1.0 - v_rad/2.99792458e5)

# the projected spin of the object is `vsini` (v * sin(i)) where `i`` is the inclination
vsini = 10.0 # km/s

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
plt.savefig('plots/spec.png')
plt.show()
"""