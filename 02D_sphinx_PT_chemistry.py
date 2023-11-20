import numpy as np
import matplotlib.pyplot as plt
import pathlib

def load_sphinx_model(Teff=3100.0, log_g=4.0, logZ=0.0, C_O=0.50):
        
    path = pathlib.Path('data/')
    sign = '+' if logZ >= 0 else '-'
    
    # PT profile
    file = path / f'Teff_{Teff:.1f}_logg_{log_g}_logZ_{sign}{abs(logZ)}_CtoO_{C_O}_atms.txt'
    assert file.exists(), f'File {file} does not exist.'
    t, p = np.loadtxt(file, unpack=True)
    
    # chemistry
    file_chem = path / file.name.replace('atms', 'mixing_ratios')
    
    with open(file_chem, 'r') as f:
        header = f.readline()
        
    header = header.split(',')
    header[0] = 'pressure'
    # remove spaces
    header = [h.strip() for h in header]
    chemistry = np.loadtxt(file_chem, unpack=True)
    VMRs = {k:v for k, v in zip(header, chemistry)}
    
    return t, p, VMRs


t, p, chemistry = load_sphinx_model(Teff=3900.0, log_g=4.0, logZ=0.0, C_O=0.50)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t, p, color='brown', lw=3.)

ax_vmr = ax.twiny()

my_species = ['H2H2', 'H2He', 'H2O', 'CO', 'Na', 'Si', 'Fe']
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
plt.show()