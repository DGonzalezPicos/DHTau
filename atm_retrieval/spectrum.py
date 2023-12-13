import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter, generic_filter

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.retrieval import rebin_give_width as rgw
import pickle

class Spectrum:
    
    order_wlen_ranges = np.array([
            [[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
            [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
            [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
            [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
            [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
            [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
            [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]],
            ])
    n_pixels = 2048 # CRIRES+ detectors
    
    def __init__(self, wave, flux, err=None):
        
        self.wave = wave
        self.flux = flux
        self.err  = err        
        
        # Make the isfinite mask
        self.update_isfinite_mask()
        self.n_orders, self.n_dets, _ = self.order_wlen_ranges.shape

        
    def __str__(self):
        out = '** Data Spectrum **\n'
        out += '-'*len(out) + '\n' # add line of dashed hyphens
        # show all attributes
        for key, value in self.__dict__.items():
            out += f'- {key} : {value}\n'
        return out
    
    def __repr__(self):
        return self.__str__()
    
    
    def update_isfinite_mask(self, array=None):

        if array is None:
            self.mask_isfinite = np.isfinite(self.flux)
        else:
            self.mask_isfinite = np.isfinite(array)
        self.n_data_points = self.mask_isfinite.sum()
        return None

    def rv_shift(self, rv, wave=None, replace_wave=False):

        # Use the supplied wavelengths
        if wave is None:
            wave = np.copy(self.wave)

        # Apply a Doppler shift to the model spectrum
        wave_shifted = wave * (1 + rv/(nc.c*1e-5))
        if replace_wave:
            self.wave = wave_shifted
        
        return wave_shifted
    
    def rot_broadening(self, vsini, epsilon_limb=0, wave=None, flux=None, replace_wave_flux=False):

        if wave is None:
            wave = self.wave
        if flux is None:
            flux = self.flux

        # Evenly space the wavelength grid
        wave_even = np.linspace(wave.min(), wave.max(), wave.size)
        flux_even = np.interp(wave_even, xp=wave, fp=flux)
        
        # Rotational broadening of the model spectrum
        flux_rot_broad = pyasl.fastRotBroad(wave_even, flux_even, 
                                            epsilon=epsilon_limb, 
                                            vsini=vsini
                                            )
        if replace_wave_flux:
            self.wave = wave_even
            self.flux = flux_rot_broad
        
            return flux_rot_broad
        
        else:
            return wave_even, flux_rot_broad
        
    @classmethod
    def instr_broadening(cls, wave, flux, out_res=1e6, in_res=1e6):

        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / \
                    (2*np.sqrt(2*np.log(2)))

        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, 
                                   mode='nearest'
                                   )
        return flux_LSF
            
    def rebin(self, d_wave, replace_wave_flux=False):

        # Interpolate onto the observed spectrum's wavelength grid
        flux_rebinned = np.interp(d_wave, xp=self.wave, fp=self.flux)

        if replace_wave_flux:
            self.flux = flux_rebinned
            self.wave = d_wave

            # Update the isfinite mask
            self.update_isfinite_mask()
        
        return flux_rebinned

    def shift_broaden_rebin(self, 
                            rv, 
                            vsini, 
                            epsilon_limb=0, 
                            out_res=1e6, 
                            in_res=1e6, 
                            d_wave=None, 
                            rebin=True, 
                            ):

        # Apply Doppler shift, rotational/instrumental broadening, 
        # and rebin onto a new wavelength grid
        self.rv_shift(rv, replace_wave=True)
        self.rot_broadening(vsini, epsilon_limb, replace_wave_flux=True)
        self.flux = self.instr_broadening(self.wave, self.flux, out_res, in_res)
        if rebin:
            self.rebin(d_wave, replace_wave_flux=True)
        return self
    
    def crop(self, wave_min, wave_max):
        assert hasattr(self, 'wave'), 'No wavelength array found'
        assert (wave_max > wave_min), 'Invalid wavelength range'
        
        mask = (self.wave >= wave_min) & (self.wave <= wave_max)
        assert np.sum(mask) > 0, 'No data points found in wavelength range'
        
        attrs = ['wave', 'flux', 'err', 'mask_isfinite']
        for attr in attrs:
            setattr(self, attr, getattr(self, attr)[mask])
        return self
    
    
    @classmethod
    def spectrally_weighted_integration(cls, wave, flux, array):

        # Integrate and weigh the array by the spectrum
        integral1 = np.trapz(wave*flux*array, wave)
        integral2 = np.trapz(wave*flux, wave)

        return integral1/integral2
    
    def plot(self, ax=None, **kwargs):

        assert hasattr(self, 'wave'), 'No wavelength array found'
        assert hasattr(self, 'flux'), 'No flux array found'
        
        if len(self.flux.shape) > 1:
            n_orders, n_dets, n_pixels = self.flux.shape
            if ax is None:
                fig, ax = plt.subplots(n_orders, figsize=(14,8))
                # ax = np.atleast_2d(ax).tolist()
                ax = [ax,] if n_orders == 1 else ax
            for i in range(n_orders):
                for j in range(n_dets):
                    ax[i].plot(self.wave[i,j], self.flux[i,j], **kwargs)
            ax[-1].set(xlabel=r'Wavelength [$\mu$m]', ylabel='Flux')
        else:
            if ax is None:
                fig, ax = plt.subplots(1, figsize=(14,4))
            ax.plot(self.wave, self.flux, **kwargs)
            
            ax.set(xlabel=r'Wavelength [$\mu$m]', ylabel=f'Flux [{self.flux_units}]')
        
        
        return ax
    
    def pickle_save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        return None
    
    def pickle_load(self, file):
        with open(file, 'rb') as f:
            self = pickle.load(f)
        return self
    

class DataSpectrum(Spectrum):

    def __init__(self, 
                 wave=None, 
                 flux=None, 
                 err=None, 
                 ra=None,
                 dec=None,
                 mjd=None,
                 file_target=None, 
                 file_wave=None, 
                 slit='w_0.2', 
                 wave_range=(1900,2500), 
                 flux_units='photons'
                 ):
        
        # Save additional information to calculate barycentric correction
        self.ra = ra
        self.dec = dec
        self.mjd = mjd

        if file_target is not None:
            data = np.loadtxt(file_target).T
            # read only first 3 columns
            wave, flux, err = data[0], data[1], data[2]
            
        # Load in (other) corrected wavelengths
        if file_wave is not None:
            wave, _, _ = np.loadtxt(file_wave).T
            
        self.flux_units = flux_units
        if self.flux_units == 'photons':
            # Convert from [photons] to [erg nm^-1]
            flux /= wave
            err /= wave

        super().__init__(wave, flux, err)

        # Reshape the orders and detectors
        #self.reshape_orders_dets()

        # Set to None initially
        self.transm, self.transm_err = None, None

        # Get the spectral resolution
        self.slit = slit
        if self.slit == 'w_0.2':
            self.resolution = 1e5
        elif self.slit == 'w_0.4':
            self.resolution = 5e4

        self.wave_range = wave_range
        
    
        
    def bary_corr(self, replace_wave=True, return_v_bary=False):

        # Barycentric velocity (using Paranal coordinates)
        self.v_bary, _ = pyasl.helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, 
                                       ra2000=self.ra, dec2000=self.dec, 
                                       jd=self.mjd+2400000.5
                                       )
        print('Barycentric velocity: {:.2f} km/s'.format(self.v_bary))
        if return_v_bary:
            return self.v_bary

        # Apply barycentric correction
        wave_shifted = self.rv_shift(self.v_bary, replace_wave=replace_wave)
        return wave_shifted

    def reshape_orders_dets(self):

        # Ordered arrays of shape (n_orders, n_dets, n_pixels)
        wave_ordered = np.ones((self.n_orders, self.n_dets, self.n_pixels)) * np.nan
        flux_ordered = np.copy(wave_ordered)
        err_ordered  = np.copy(wave_ordered)
        transm_ordered = np.copy(wave_ordered)
        flux_uncorr_ordered = np.copy(wave_ordered)

        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                # Select only pixels within the detector, should be 2048
                mask_wave = np.arange(
                    self.n_pixels * (i*self.n_dets + j), 
                    self.n_pixels * (i*self.n_dets + j + 1), 
                    dtype=int
                    )

                if mask_wave.any():
                    wave_ordered[i,j] = self.wave[mask_wave]
                    flux_ordered[i,j] = self.flux[mask_wave]
                    err_ordered[i,j]  = self.err[mask_wave]

                    if self.transm is not None:
                        transm_ordered[i,j] = self.transm[mask_wave]
                    if hasattr(self, 'flux_uncorr'):
                        flux_uncorr_ordered[i,j] = self.flux_uncorr[mask_wave]

        self.wave = wave_ordered
        self.flux = flux_ordered
        self.err  = err_ordered
        self.transm = transm_ordered
        self.transm = np.where(self.transm<=0.0, 1.0, self.transm)
        self.flux_uncorr = flux_uncorr_ordered

        # Remove empty orders / detectors
        self.clear_empty_orders_dets()

        # Update the isfinite mask
        self.update_isfinite_mask()
        return self
    
    def clear_empty_orders_dets(self):

        # If all pixels are NaNs within an order...
        mask_empty = (~np.isfinite(self.flux)).all(axis=(1,2))
        
        # ... remove that order
        self.wave = self.wave[~mask_empty,:,:]
        self.flux = self.flux[~mask_empty,:,:]
        self.err  = self.err[~mask_empty,:,:]
        self.transm = self.transm[~mask_empty,:,:]
        self.flux_uncorr = self.flux_uncorr[~mask_empty,:,:]

        # Update the wavelength ranges for this instance
        self.order_wlen_ranges = self.order_wlen_ranges[~mask_empty]

        # Update the number of orders, detectors, and pixels 
        # for this instance
        self.n_orders, self.n_dets, self.n_pixels = self.flux.shape
        return None
    
    def clip_det_edges(self, n_edge_pixels=30):
        
        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                idx_low  = self.n_pixels * (i*self.n_dets + j)
                idx_high = self.n_pixels * (i*self.n_dets + j + 1)

                self.flux[idx_low : idx_low + n_edge_pixels]   = np.nan
                self.flux[idx_high - n_edge_pixels : idx_high] = np.nan

        # Update the isfinite mask
        self.update_isfinite_mask()
        return None
    
    def load_molecfit_transm(self, file_transm, tell_threshold=0.0):

        # Load the pre-computed transmission from molecfit
        molecfit = np.loadtxt(file_transm).T
        
        # Confirm that we are using the same wavelength grid
        assert((self.wave == molecfit[0]).all())
    
        
        assert len(molecfit) == 3, f'Expected 3 columns [wave, trans, continuum] in {file_transm}, got {len(molecfit)}'
        self.wave_transm, self.transm, self.cont_transm = np.loadtxt(file_transm, unpack=True)
        # self.transm_err = self.err/np.where(self.transm<=0.0, 1.0, self.transm) 

        mask_high_transm = (self.transm > tell_threshold)
        mask = (self.mask_isfinite & mask_high_transm)
        self.throughput = self.cont_transm.reshape(np.shape(self.wave))
        self.throughput /= np.nanmax(self.throughput)
        return None
    

class ModelSpectrum(Spectrum):

    def __init__(self, 
                 wave, 
                 flux, 
                 lbl_opacity_sampling=1, 
                 multiple_orders=False, 
                 ):

        super().__init__(wave, flux)

        if multiple_orders:
            # New instance is combination of previous (order) instances
            assert(self.wave.ndim == 3)
            assert(self.flux.ndim == 3)

            # Update the shape of the model spectrum
            self.n_orders, self.n_dets, self.n_pixels = self.flux.shape

            # Update the order wavelength ranges
            mask_order_wlen_ranges = \
                (self.order_wlen_ranges.min(axis=(1,2)) > self.wave.min() - 5) & \
                (self.order_wlen_ranges.max(axis=(1,2)) < self.wave.max() + 5)
                
            self.order_wlen_ranges = self.order_wlen_ranges[mask_order_wlen_ranges,:,:]

        # Model resolution depends on the opacity sampling
        self.resolution = int(1e6/lbl_opacity_sampling)

    
    
if __name__ == '__main__':
    
    # file = 'data/crires_example_spectrum.dat'
    file = 'data/prt_fake_spectrum.txt'
    # spec = DataSpectrum(file_target=file, slit='w_0.4').reshape_orders_dets()
    spec = DataSpectrum(file_target=file, slit='w_0.4', flux_units='erg/s/cm2/cm')
    # TODO: flux units...
    ax = spec.plot(color='b')
    
    # FIXME: ValueError: object too deep for desired array --> 3D array
    spec.shift_broaden_rebin(rv=0, vsini=10, epsilon_limb=0, out_res=1e4, d_wave=spec.wave)
    spec.plot(ax=ax, color='r')
    plt.show()