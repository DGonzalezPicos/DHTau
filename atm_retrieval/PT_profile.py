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
        
        
    
    
    def plot(self, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.temperature, self.pressure, **kwargs)
        ax.plot(self.T_knots, self.P_knots, 'o', markersize=8, color='red')
        
        ax.set(yscale='log', ylim=(self.pressure.max(), self.pressure.min()), 
               xlabel='Temperature [K]', ylabel='Pressure [bar]')
        return ax
        
        
if __name__=='__main__':
    # Structure of the atmosphere
    logP_max = 2.0
    logP_min = -5.0
    n_layers = 30 # plane-parallel layers
    pressure = np.logspace(logP_min, logP_max, n_layers) # from top to bottom

    log_P_knots = np.linspace(logP_max, logP_min, 4) # bottom to top
    T_knots = np.array([6000., 2500., 1900., 1400.]) # bottom to top
    pt = PT(pressure)
    
    temperature = pt.spline(log_P_knots, T_knots)
    pt.plot()
    plt.show()
    
    