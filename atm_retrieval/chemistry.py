import numpy as np
import matplotlib.pyplot as plt

class Chemistry:
    
    # Dictionary with info per molecular/atomic species
    # (pRT_name, mass, number of (C,O,H) atoms
    species_info = {
        # '12CO':    ('CO_main_iso',             'C1O1',     12.011 + 15.999,            (1,1,0)), 
        '12CO': (12.011 + 15.999,             (1,1,0)),
        'H2O': (1.00794 * 2.0 + 15.999,       (0,1,2)),
        '13CO': (13.0033548378 + 15.999,      (1,1,0)),
        'C18O': (12.011 + 17.999,             (1,1,0)),
        'C17O': (12.011 + 16.999,             (1,1,0)),
        
        'HF': (1.00794 + 18.9984032,           (0,0,1)),
        'CN': (12.011 + 14.0067,               (1,0,0)),
        'H2': (1.00794 * 2.0,                  (0,0,2)),
        # Atomic species
        'H': (1.00794,                        (0,0,1)),
        'He': (4.002602,                       (0,0,0)),
        'Na': (22.98976928,                    (0,0,0)),
        'K': (39.0983,                         (0,0,0)),
        'Li': (6.941,                          (0,0,0)),
        'Rb': (85.4678,                        (0,0,0)),
        'Cs': (132.90545196,                   (0,0,0)),
        'Ca': (40.078,                         (0,0,0)),
        'Mg': (24.305,                         (0,0,0)),
        'Sr': (87.62,                          (0,0,0)),
        'Ba': (137.327,                        (0,0,0)),
        'Fe': (55.845,                         (0,0,0)),
        'Ti': (47.867,                         (0,0,0)),
        'Cr': (51.9961,                        (0,0,0)),
        'Ni': (58.6934,                        (0,0,0)),
        'Y': (88.90584,                        (0,0,0)),
        'Al': (26.9815384,                     (0,0,0)),
        'Si': (28.085,                         (0,0,0)),
        'Sc': (44.955908,                      (0,0,0)),
        'V': (50.9415,                         (0,0,0)),
        'Mn': (54.938044,                      (0,0,0)),
        'Co': (58.933194,                      (0,0,0)),
        'Cu': (63.546,                         (0,0,0)),
        'Zn': (65.38,                          (0,0,0)),
        'Ga': (69.723,                         (0,0,0)),
        }

    
    def __init__(self, pressure):
        # self.line_species = line_species
        self.pressure = pressure
        self.n_atm_layers = len(pressure)
        
    def __call__(self, VMRs):
        
        
        self.VMRs = VMRs

        # Total VMR without H2, starting with He
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He

        # Create a dictionary for all used species
        self.mass_fractions = {}

        C, O, H = 0, 0, 0
        
        for species_i in list(self.VMRs.keys()):
            assert species_i in self.species_info.keys(), f'{species_i} not in species_info.keys()'
            line_species_i = self.VMRs[species_i][-1]
            VMR_i = self.VMRs[species_i][0] * np.ones_like(self.pressure)
            
            # self.VMRs[species_i] = VMR_i
            self.mass_fractions[line_species_i] = VMR_i * self.species_info[species_i][0]
            VMR_wo_H2 += VMR_i
            
            # Add the number of atoms to the total
            C += self.species_info[species_i][1][0] * VMR_i
            O += self.species_info[species_i][1][1] * VMR_i
            H += self.species_info[species_i][1][2] * VMR_i
            
        # Add H2 and He
        self.mass_fractions['He'] = VMR_He * self.species_info['He'][0]
        self.mass_fractions['H2'] = (1.0 - VMR_wo_H2) * self.species_info['H2'][0]
        H += self.species_info['H2'][1][2] * (1.0 - VMR_wo_H2)
        self.mass_fractions['H'] = self.species_info['H'][0] * H
        
        # add mass fractions for H- continuum opacity (using solar values)
        self.mass_fractions['H-'] = 6e-9 # solar
        self.mass_fractions['e-'] = 1e-10 # solar
        
        # Mean Molecular Weight        
        MMW = 0.
        for mass_i in self.mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones_like(self.pressure)
        
        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= MMW
        
        # store in mass_fractions dictionary
        self.mass_fractions['MMW'] = MMW
        
        # Compute the C/O ratio and metallicity
        self.CO = C/O

        #log_CH_solar = 8.43 - 12 # Asplund et al. (2009)
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        self.FeH = np.log10(C/H) - log_CH_solar
        self.CH  = self.FeH

        self.CO = np.mean(self.CO)
        self.FeH = np.mean(self.FeH)
        self.CH = np.mean(self.CH)
            
        return self.mass_fractions
    
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for key in self.mass_fractions:
            if key == 'MMW':
                continue
            ax.plot(self.mass_fractions[key], self.pressure, label=key, **kwargs)
        
        ax.legend()
        ax.set(yscale='log', xscale='log', ylim=(self.pressure.max(), self.pressure.min()), 
               xlabel='Mass Fraction', ylabel='Pressure [bar]')
        
        return ax
    
    
if __name__=='__main__':
    
    
    logP_max = 2.0
    logP_min = -5.0
    n_layers = 30 # plane-parallel layers
    pressure = np.logspace(logP_min, logP_max, n_layers) # from top to bottom
    VMRs = {'H2O_pokazatel_main_iso': 1e-4, 'CO_high': 1e-4, 'Na_allard': 1e-6}
    
    chem = Chemistry(pressure)
    mass_fractions = chem(VMRs)
    
    ax = chem.plot()
    plt.show()