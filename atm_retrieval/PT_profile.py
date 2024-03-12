import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
class PT:
    
    
    def __init__(self, pressure):
        
        self.pressure = pressure
        self.n_atm_layers = len(pressure)
        
    def spline(self, log_P_knots, T_knots, kind='quadratic'):
        ''' Generate a spline interpolation of the PT-profile
        
        Parameters
        ----------
        log_P_knots : array-like
            log10 of the pressure knots
        T_knots : array-like
            Temperature knots
        kind : str, optional
            Kind of spline interpolation, by default 'quadratic'
            
        Returns
        -------
            self : PT
                Returns an instance of the PT class (with self.temperature)
        '''
        assert len(log_P_knots) == len(T_knots), 'log_P_knots and T_knots must have the same length'
        self.log_P_knots = np.array(log_P_knots)
        self.T_knots = np.array(T_knots)
        # Knots for the spline interpolation
        sort = np.argsort(self.log_P_knots) 

        # Pressure knots must be in increasing order
        self.P_knots = 10.0**self.log_P_knots
        
        # use scipy.interpolated.interp1d
        interp = interp1d(self.log_P_knots[sort], np.log10(self.T_knots[sort]), kind=kind)
        self.temperature = 10**interp(np.log10(self.pressure))
        return self.temperature
    
    # def __call__(self, log_P_knots=None, T_knots=None, kind='quadratic',**kwargs):
    #     self.spline(log_P_knots, T_knots, kind=kind)
    #     return self.temperature
        
    def gradient(self, T_0, log_P_knots, dlnT_dlnP_knots, kind='linear'):
        ''' Generate a temperature profile from the gradients at the knots
        
        Parameters
        ----------
        T_0 : float
            Temperature at the BOTTOM of the atmosphere
        log_P_knots : array-like
            log10 of the pressure knots, sorted in the function from bottom to top
        dlnT_dlnP_knots : array-like
            Gradients of the temperature with respect to the pressure, corresponding to the log_P_knots
        kind : str, optional
            Kind of spline interpolation, by default 'linear'
        
        Returns
        -------
            self : PT
                Returns an instance of the PT class (with bottom-up self.temperature, self.dlnT_dlnP)
        
        '''
        assert len(log_P_knots) == len(dlnT_dlnP_knots), 'log_P_knots and dlnT_dlnP_knots must have the same length'
        self.log_P_knots = np.array(log_P_knots)
        self.dlnT_dlnP_knots = np.array(dlnT_dlnP_knots)
        # Knots for the spline interpolation, sort from high to low pressure
        sort = np.argsort(self.log_P_knots)[::-1]
        # Pressure knots must be in increasing order
        self.P_knots = 10.0**self.log_P_knots[sort]
        self.log_P_knots = self.log_P_knots[sort]
        # use scipy.interpolated.interp1d
        interp = interp1d(self.log_P_knots[sort], self.dlnT_dlnP_knots[sort], kind=kind)
        pressure_decreasing = self.pressure[::-1]
        self.dlnT_dlnP = interp(np.log10(pressure_decreasing))
        
        # temperature defined from bottom to top (high to low pressure)
        self.temperature = [T_0]
        for i in range(1, self.n_atm_layers):
            self.temperature.append(self.temperature[i-1] * np.exp(np.log(self.pressure[i-1]/self.pressure[i]) * self.dlnT_dlnP[i-1]))
        
        self.temperature = np.array(self.temperature[::-1])
        self.dlnT_dlnP = self.dlnT_dlnP[::-1]
        
        return self.temperature
    
    def plot(self, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.temperature, self.pressure, **kwargs)
        if hasattr(self, 'T_knots'):
            ax.plot(self.T_knots, self.P_knots, 'o', markersize=8, color='red')
        # if hasattr(self, 'dlnT_dlnP_knots'):
        #     ax.plot
        else:
            [ax.axhline(P_i, color='k', linestyle='--', alpha=0.2) for P_i in self.P_knots]
        
        ax.set(yscale='log', ylim=(self.pressure.max()*1.1, 0.9*self.pressure.min()), 
               xlabel='Temperature [K]', ylabel='Pressure [bar]')
        return ax
    
    def plot_gradient(self, ax=None, fig_name=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharey=True, gridspec_kw={'wspace':0.05,
                                                                                    'width_ratios': [1.5, 3]})
            
        # ax[0] is for gradient
        lw = kwargs.pop('linewidth', 2)
        ax[0].plot(self.dlnT_dlnP, self.pressure, color='green', lw=lw, **kwargs)
        ax[1].plot(self.temperature, self.pressure,color='brown', lw=lw, **kwargs)
        
        for i, P_i in enumerate(self.P_knots):
            ax[0].axhline(P_i, color='k', linestyle='--', alpha=0.4)
            ax[1].axhline(P_i, color='k', linestyle='--', alpha=0.4)
            ax[0].text(0.02, P_i, f'log P = {np.log10(P_i):.1f}', color='red', verticalalignment='bottom')
            # ax[1].text(0.05, P_i, f'log P = {np.log10(P_i):.1f}', color='red')
            # ax[1].axhspan(P_i, self.temperature[i], color='red', alpha=0.3 * self.dlnT_dlnP_knots[i] / self.dlnT_dlnP_knots.max())
            if i < len(self.P_knots) - 1:
                alpha = np.mean([self.dlnT_dlnP_knots[i], self.dlnT_dlnP_knots[i+1]]) / (self.dlnT_dlnP_knots.max() - self.dlnT_dlnP_knots.min())
                alpha = max(0.0, alpha)
                ax[1].fill_between([0., 20e3], P_i, self.P_knots[i+1], color='g', alpha=0.3*alpha)
            
        ax[0].set(yscale='log', ylim=(self.pressure.max(), self.pressure.min()), 
                  xlim=(0.0, 0.35),
               xlabel='dlnT/dlnP', ylabel='Pressure [bar]')
        
        xlim = (self.temperature.min(), self.temperature.max())
        xpad = 0.05 * (xlim[1] - xlim[0])
        ax[1].set(yscale='log', ylim=(self.pressure.max(), self.pressure.min()), 
               xlabel='Temperature [K]',
               xlim=(xlim[0] - xpad, xlim[1] + xpad))
        if fig_name is not None:
            fig.suptitle(r'PT profile from temperature gradients', fontsize=16, y=0.94)
            plt.savefig(fig_name)
            print(f' PT profile with gradient saved to {fig_name}')
            plt.close()
        return ax
        
        
if __name__=='__main__':
    # Structure of the atmosphere
    logP_max = 2.0
    logP_min = -5.0
    n_layers = 30 # plane-parallel layers
    pressure = np.logspace(logP_min, logP_max, n_layers) # from top to bottom
    log_P_knots = np.linspace(logP_max, logP_min, 8) # bottom to top
    
    # Initialize the PT class
    pt = PT(pressure)
    
    call_spline = False # 
    if call_spline: # this is the classic temperature profile from a set of temperature values
        T_knots = np.array([6000., 4200., 3500., 2500., 1900., 1600., 1400., 800.]) # bottom to top
        assert len(log_P_knots) == len(T_knots), 'log_P_knots and T_knots must have the same length'
        temperature = pt.spline(log_P_knots, T_knots)
    
    
    # New temperature profile from gradients
    # Below we plot the generated temperature profile and gradients for different spline interpolations
    # If the number of knots is high (>5), linear interpolation is recommended
    spline_kind =['linear', 'quadratic', 'cubic']
    for kind in spline_kind:
        temperature_grad = pt.gradient(6000, log_P_knots, [0.18, 0.11, 0.25, 0.12, 0.05, 0.08, 0.06, 0.04], kind=kind)
        pt.plot_gradient(fig_name = f'PT_profile_grad_{kind}.pdf')
    
    