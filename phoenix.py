import numpy as np
import pathlib 
import matplotlib.pyplot as plt
from astropy.io import fits
import atm_retrieval.figures as figs


class Phoenix:
    
    
    # change this path to the folder where the Phoenix files are stored
    path = pathlib.Path('/data1/wolde/GIT/DHTau/phoenix_files/PHOENIX')
    
    
    def __init__(self, Teff, logg, Z=0.0):
        self.Teff = Teff
        self.logg = logg
        self.Z = Z
        assert self.Z == 0.0, f'Z = {self.Z} is not supported. Only Z = 0.0 is supported.'
        self.folder = 'PHOENIX-ACES-AGSS-COND-2011_AtmosFITS_Z-0.0'
        self.load_atmosphere()
        
    def load_atmosphere(self):
        # Find the path to the file
        filename = f'lte{self.Teff:05d}-{self.logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011.ATMOS.fits'
        file = self.path / filename
        assert file.exists(), f'File {file} does not exist.'
        
        # Load the file
        with fits.open(file) as hdul:
            #print(hdul.info())
            self.header = hdul[0].header
            self.atm = hdul[1].data
            # convert to dictionary
            self.atm = {k: self.atm[k] for k in self.atm.columns.names}
            self.abundances = hdul[2].data
            
        #print(f' Available attributes of the atmosphere: {self.atm.keys()}')
        #print(f' Available attributes of the abundances: {self.abundances.columns.names}')
        
        # convert pressure from dyn/cm^2 to bar
        self.atm['pressure'] = self.atm['pgas'] * 1e-6
        return self
    
    def plot_PT(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.atm['temp'], self.atm['pressure'], **kwargs)
        return ax

            
            
if __name__ == '__main__':
    
    # plot a range of PT profiles for different temperatures
    temperatures = np.arange(2800, 4400, 200)
    loggs = [2.0, 3.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(temperatures)))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    
    for i, T in enumerate(temperatures):
        for j, logg in enumerate(loggs):
            ph = Phoenix(T, logg)
            
            label = f'logg = {logg}' if i == 0 else None
            ls = '-' if j == 0 else '--'
            ph.plot_PT(ax=ax, color=colors[i], ls=ls, label=label, alpha=0.7, lw=2.)
            ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]', ylim=(self.atm['pressure'].max(), self.atm['pressure'].min()),
               yscale='log')
        
        
    # create colorbar
    cbar_im = ax.scatter([], [], c=[], cmap='plasma', vmin=temperatures.min(), vmax=temperatures.max())
    cbar = fig.colorbar(cbar_im, ax=ax, label='Temperature [K]')
    ax.legend()
    ax.set_ylim(1e2, 1e-5) # region we are interested in
    plt.savefig('plots/phoenix.png')
    plt.show()

    figs.fig_PT(
        PT=self.pRT_model.PT, 
        # xlim=(x1,x2), 
        bestfit_color=self.bestfit_color,
        envelopes_color=self.bestfit_color,
        int_contr_em_color='red',
        fig_name=self.run_dir / f'plots/retrieval_PT_profile_{fig_label}.pdf',
        )