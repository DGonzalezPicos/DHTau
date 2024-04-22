import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter, generic_filter
import warnings

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.retrieval import rebin_give_width as rgw
import pickle

import atm_retrieval.figures as figs

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
    normalized = False
    
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
            return self
        
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
    
    def crop_spectrum(self):
        assert hasattr(self, 'wave_range'), 'No wavelength range found'
        # Crop the spectrum to within a given wavelength range
        mask_wave = (self.wave >= self.wave_range[0]) & \
                    (self.wave <= self.wave_range[1])
        assert np.sum(mask_wave) > 0, 'No data points found in wavelength range'
        
        self.flux[~mask_wave] = np.nan
        if hasattr(self, 'flux_uncorr'):
            self.flux_uncorr[~mask_wave] = np.nan
        return self
    
    def flatten(self, debug=False):
        assert hasattr(self, 'wave'), 'No wavelength array found'
        shape_in = self.flux.shape
        if len(shape_in) == 1:
            print(f'Flux array already flattened, shape = {shape_in}')
            return self

        attrs = ['wave', 'flux', 'err', 'mask_isfinite']
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).flatten())
        shape_out = self.flux.shape
        if debug:
            print(f'Flux array flattened from {shape_in} to {shape_out}')
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
    
    def save(self, file):
        extension = file.split('.')[-1]
        supported_extensions = ['pickle', 'txt', 'dat']
        assert extension in supported_extensions, \
            f'Extension {extension} not supported, use one of {supported_extensions}'
     
        if extension == 'pickle':
            self.pickle_save(file)
            
        elif extension in ['txt', 'dat']:
            np.savetxt(file, np.array([self.wave, self.flux, self.err]).T)
        
        return None
    
    def load(self, file):
        
        extension = file.split('.')[-1]
        assert extension in ['pickle', 'txt', 'dat'], \
            f'Extension {extension} not supported, use one of {supported_extensions}'
        
        if extension == 'pickle':
            self = self.pickle_load(file)
            
        elif extension in ['txt', 'dat']:
            # self.wave, self.flux, self.err = np.loadtxt(file).T
            data = np.loadtxt(file).T
            print(f'Loading data (shape={data.shape}) from {file}')
            if len(data) == 3:
                self.wave, self.flux, self.err = data
            elif len(data) == 6: # New file format with 6 columns
                self.wave, self.flux, self.err = data[:3]
                self.transm, self.cont_transm, self.nans = data[3:]
                
                # apply Nans to flux
                # Nans = True at the edges, outliers from Molecfit and wave exclude in Molecfit preprocessing
                self.flux[self.nans.astype(bool)] = np.nan
        return self

    @staticmethod    
    def planck(wave, Teff):
        '''Calculate Planck function for a given wavelength and temperature
        Parameters
        ----------
            wave : np.array
                Wavelength in nm
            Teff : float
                Effective temperature of standard star in K
                
        Returns
        -------
            flux_bb : np.array
                Planck function in flux units [erg/s/cm2/nm]'''
        
        # wavelength in cm, all constants in cgs units
        wave_cm = wave * 1e-7 
        flux_bb = np.pi * nc.b(Teff, nc.c / wave_cm) # blackbody flux in [erg/s/cm2/Hz]
        # convert [erg/s/cm2/Hz] -> [erg/s/cm2/cm]
        flux_bb *= nc.c / wave_cm**2
        # convert [erg/s/cm2/cm] -> [erg/s/cm2/nm]
        flux_bb *= 1e-7
        return flux_bb
    
    def normalize_flux_per_order(self, fun='median', tell_threshold=0.0):
        assert len(np.shape(self.flux))==3, 'Flux array not reshaped into orders and detectors'
        
        self.normalize_args = {'fun': fun, 'tell_threshold': tell_threshold}
        deep_lines = self.transm < tell_threshold if hasattr(self, 'transm') else np.zeros_like(self.flux, dtype=bool)
        f = np.where(deep_lines, np.nan, self.flux)
        value = getattr(np, f'nan{fun}')(f, axis=-1)[...,None] # median flux per order
        self.flux /= value
        if getattr(self, 'err', None) is not None:
            self.err /= value
        if hasattr(self, 'flux_uncorr'):
            self.flux_uncorr /= value
            
        self.normalized = True
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
                 file_wave=None, # deprecated
                 slit='w_0.2', 
                 wave_range=(1900,2500), 
                 flux_units='photons',
                 Teff_standard=None,
                 night=None,
                 ):
        
        # Save additional information to calculate barycentric correction
        self.ra = ra
        self.dec = dec
        self.mjd = mjd

        # Set to None initially
        # self.transm, self.transm_err = None, None
        
        if file_target is not None:
            data = np.loadtxt(file_target).T
            # first 3 columns
            wave, flux, err = data[0], data[1], data[2]
            if len(data) == 6:
                # 6 columns: [wave, flux, err, transm, cont_transm, nans]
                self.transm, self.cont_transm, nans = data[3:]
                self.nans = nans.astype(bool)
            
        # Load in (other) corrected wavelengths
        if file_wave is not None:
            wave, _, _ = np.loadtxt(file_wave).T
            
        self.flux_units = flux_units
        ref_flux = 1.0 if Teff_standard is None else self.planck(wave, Teff_standard)
        if self.flux_units == 'photons':
            # Convert from [photons] to [erg nm^-1]
            flux /= wave
            err /= wave
            if hasattr(self, 'cont_transm'):
                self.throughput = self.cont_transm / wave
                self.throughput /= ref_flux # remove blackbody of standard-star from throughput

        super().__init__(wave, flux, err)

        # Reshape the orders and detectors
        #self.reshape_orders_dets()



        # Get the spectral resolution
        self.slit = slit
        if self.slit == 'w_0.2':
            self.resolution = 1e5
        elif self.slit == 'w_0.4':
            self.resolution = 5e4

        self.wave_range = wave_range
        self.reshaped = False # default
        
        self.night = night  
        self.night_label=''
        if self.night is not None:
            self.night_label = f'_night{self.night}'
        
    def bary_corr(self, replace_wave=True, return_v_bary=False):

        # Barycentric velocity (using Paranal coordinates)
        self.v_bary, _ = pyasl.helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, 
                                       ra2000=self.ra, dec2000=self.dec, 
                                       jd=self.mjd+2400000.5
                                       )
        print(' Barycentric velocity: {:.2f} km/s'.format(self.v_bary))
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
        self.reshaped = True

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
        assert len(np.shape(self.flux)) == 1, 'Only works for 1D arrays'
        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                idx_low  = self.n_pixels * (i*self.n_dets + j)
                idx_high = self.n_pixels * (i*self.n_dets + j + 1)

                self.flux[idx_low : idx_low + n_edge_pixels]   = np.nan
                self.flux[idx_high - n_edge_pixels : idx_high] = np.nan
                if hasattr(self, 'flux_uncorr'):
                    self.flux_uncorr[idx_low : idx_low + n_edge_pixels]   = np.nan
                    self.flux_uncorr[idx_high - n_edge_pixels : idx_high] = np.nan

        # Update the isfinite mask
        self.update_isfinite_mask()
        return self
    
    def load_molecfit_transm(self, file_transm=None):

        # Load the pre-computed transmission from molecfit
        if not hasattr(self, 'cont_transm'):
            assert file_transm is not None, 'No molecfit transmission file found'
            molecfit = np.loadtxt(file_transm).T
            
            # Confirm that we are using the same wavelength grid
            assert((self.wave == molecfit[0]).all())
        
            
            assert len(molecfit) == 3, f'Expected 3 columns [wave, trans, continuum] in {file_transm}, got {len(molecfit)}'
            self.wave_transm, self.transm, self.cont_transm = np.loadtxt(file_transm, unpack=True)
        
        self.throughput = self.cont_transm.reshape(np.shape(self.wave))
        self.throughput /= np.nanmax(self.throughput)
        return None
    
    def sigma_clip_median_filter(self, sigma=3, filter_width=3, 
                                 replace_flux=True, 
                                 debug=True,
                                 ):

        flux_copy = self.flux.copy()
        sigma_clip_bounds = np.ones((3, self.n_orders, 3*self.n_pixels)) * np.nan

        # Loop over the orders
        for i in range(self.n_orders):

            # Select only pixels within the order, should be 3*2048
            idx_low  = self.n_pixels * (i*self.n_dets)
            idx_high = self.n_pixels * ((i+1)*self.n_dets)
            
            mask_wave = np.zeros_like(self.wave, dtype=bool)
            mask_wave[idx_low:idx_high] = True
            
            mask_order = (mask_wave & self.mask_isfinite)

            if mask_order.any():

                flux_i = flux_copy[mask_order]

                # Apply a median filter to this order
                filtered_flux_i = generic_filter(flux_i, np.nanmedian, size=filter_width)
                
                # Subtract the filtered flux
                residuals = flux_i - filtered_flux_i

                # Sigma-clip the residuals
                mask_clipped = np.isnan(residuals) 
                mask_clipped |= (np.abs(residuals) > sigma*np.nanstd(residuals))
                # mask_clipped = (np.abs(residuals) > sigma*np.std(residuals))

                sigma_clip_bounds[1,i,self.mask_isfinite[mask_wave]] = filtered_flux_i
                sigma_clip_bounds[0,i] = sigma_clip_bounds[1,i] - sigma*np.std(residuals)
                sigma_clip_bounds[2,i] = sigma_clip_bounds[1,i] + sigma*np.std(residuals)

                # Set clipped values to NaNs
                flux_i[mask_clipped]  = np.nan
                flux_copy[mask_order] = flux_i
                
        if debug:
            print(f'[sigma_clip_median_filter] Fraction of clipped data {np.isnan(flux_copy).sum()/np.size(flux_copy):.2f}')

        if replace_flux:
            self.flux = flux_copy

            # Update the isfinite mask
            self.update_isfinite_mask()

        return flux_copy
    
    def sigma_clip(self, sigma=3, filter_width=11, 
                   replace_flux=True,
                   fig_name=None,
                   debug=True,
                    ):
        '''Sigma clip flux values of reshaped DataSpectrum instance'''
        assert self.reshaped, 'DataSpectrum instance not reshaped, use reshape_orders_dets()'
        # np.seterr(invalid='ignore')

        flux_copy = self.flux.copy()
        clip_mask = np.zeros_like(flux_copy, dtype=bool)
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                flux = self.flux[order,det]
                mask = np.isfinite(flux)
                if mask.any():
                    # with np.errstate(invalid='ignore'):
                    with warnings.catch_warnings():
                        # ignore numpy RuntimeWarning: Mean of empty slice
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        filtered_flux_i = generic_filter(flux, np.nanmedian, size=filter_width)
                    residuals = flux - filtered_flux_i
                    nans = np.isnan(residuals)
                    mask_clipped = nans | (np.abs(residuals) > sigma*np.nanstd(residuals))
                    flux_copy[order,det,mask_clipped] = np.nan
                    clip_mask[order,det] = mask_clipped
                    if debug:
                        print(f' [sigma_clip] Order {order}, Detector {det}: {mask_clipped.sum()-nans.sum()} pixels clipped')
                        
        if fig_name is not None:
            figs.fig_sigma_clip(self, clip_mask, fig_name=fig_name)
            
        if replace_flux:
            self.flux = flux_copy
            self.update_isfinite_mask()
            return self
            
        return flux_copy
    
    def fill_nans(self, min_finite_pixels=100, debug=True):
        '''Fill NaNs order-detector pairs with less than `min_finite_pixels` finite pixels'''
        assert self.reshaped, 'The spectrum has not been reshaped yet!'
        
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                mask_ij = self.mask_isfinite[order,det]
                if mask_ij.sum() < min_finite_pixels:
                    if debug:
                        print(f'[fill_nans] Order {order}, detector {det} has only {mask_ij.sum()} finite pixels!')
                    self.flux[order,det,:] = np.nan * np.ones_like(self.flux[order,det,:])
                    # self.err[order,det,~mask_ij] = np.nanmedian(self.err[order,det,mask_ij])
        self.update_isfinite_mask()
        return self
    
    def add_dataset(self, d_spec):
        '''Add another DataSpectrum instance to the current one'''
        assert hasattr(d_spec, 'flux'), 'No flux array found in the input DataSpectrum instance'
        assert hasattr(d_spec, 'wave'), 'No wavelength array found in the input DataSpectrum instance'
        assert hasattr(d_spec, 'err'), 'No error array found in the input DataSpectrum instance'
        
        assert np.shape(self.flux) == np.shape(d_spec.flux), 'Flux arrays have different shapes'
        assert np.shape(self.wave) == np.shape(d_spec.wave), 'Wavelength arrays have different shapes'
        
        print(f'[add_dataset] Adding flux arrays of shape {np.shape(d_spec.flux)}')
        
        attrs = ['flux', 'wave', 'err', 'mask_isfinite']
        for attr in attrs:
            setattr(self, attr, np.concatenate((getattr(self, attr), getattr(d_spec, attr)), axis=1))
            
        self.n_orders, self.n_dets, self.n_pixels = self.flux.shape
        print(f'[add_dataset] New flux array shape: {np.shape(self.flux)}')
        return self
    
    def preprocess(self,
                   file_transm=None,
                   tell_threshold=0.7,
                   tell_grow_mask=0,
                   n_edge_pixels=30,
                   ra=None,
                   dec=None,
                   mjd=None,
                   flux_calibration_factor=0.0,
                   sigma_clip=3.0,
                   sigma_clip_window=11,
                #    fig_name=None,
                fig_dir=None,
                   ):
        '''Wrapper function to apply all preprocessing steps to 
        get the data ready for retrievals'''
        
        print(f'** Preprocessing data **\n----------------------')
        # Load Telluric model (fitted to the data with Molecfit)
        # molecfit_spec = DataSpectrum(file_target='data/DHTauA_molecfit_transm.dat', slit='w_0.4', flux_units='')
        if file_transm is not None:
            self.load_molecfit_transm(file_transm)

        # Divide by the molecfit spectrum 
        # throughput = molecfit_spec.err # read as the third column (fix name)  
        self.flux_uncorr = np.copy(self.flux) / self.throughput
        zeros = self.transm <= 0.01
        self.flux = np.divide(self.flux, self.transm * self.throughput, where=np.logical_not(zeros))
        self.err = np.divide(self.err, self.transm * self.throughput, where=np.logical_not(zeros))
        print(f' Telluric correction applied (threshold = {tell_threshold:.1f})')
        # mask regions with deep telluric lines
        tell_mask = self.transm < tell_threshold
        if tell_grow_mask > 0:
            print(f' Growing telluric mask by {tell_grow_mask} pixels')
            tell_mask = np.convolve(tell_mask, np.ones(tell_grow_mask), mode='same') > 0
            nans = np.isnan(self.flux)
            mask_fraction = (tell_mask.sum()-nans.sum())/tell_mask.size
            print(f' Masking deep tellurics ({100*mask_fraction:.1f} % of pixels)')
        self.flux[tell_mask] = np.nan
        
        # mask regions flagged by molecfit
        if hasattr(self, 'nans'):
            self.flux[self.nans.astype(bool)] = np.nan
            nan_frac = (np.isnan(self.flux).sum()-self.nans.sum())/self.nans.size
            print(f' Masking regions flagged by Molecfit ({100*nan_frac:.1f} % of pixels)')
        
        print(f' Edge pixels clipped: {n_edge_pixels}')
        self.clip_det_edges(n_edge_pixels)
        
        ## shift to barycentric frame
        if (ra is not None) and (dec is not None) and (mjd is not None):
            self.ra, self.dec, self.mjd = ra, dec, mjd
            self.bary_corr()
            
        self.crop_spectrum()
        print(f' Spectrum cropped to {self.wave_range} nm')
        self.update_isfinite_mask()
        
        # if sigma_clip is not None:
            # flux_unclip = np.copy(self.flux)
            # nans_before = np.isnan(self.flux).sum()
            # self.sigma_clip_median_filter(sigma=sigma_clip, filter_width=sigma_clip_window, debug=False)
            # nans_after = np.isnan(self.flux).sum()
            # clipped_fraction = (nans_after-nans_before)/self.flux.size
            # print(f' Sigma clipped at {sigma_clip} sigma ({100*clipped_fraction:.1f} % of pixels)')
            # self.sigma_clip(sigma=sigma_clip, filter_width=sigma_clip_window, replace_flux=True, debug=True)


        
        self.reshape_orders_dets()
        print(f' Data reshaped into orders and detectors')

        self.fill_nans(min_finite_pixels=200)
        # TODO: proper flux calibration...
        # DGP (2024-04-16): just normalize every order-detector pair since we have flux scaling
        # convert to erg/s/cm2/cm (quick manual fix for lack of flux calibration..)
        if flux_calibration_factor > 0.0:
            self.flux *= flux_calibration_factor
            self.err *= flux_calibration_factor
            self.flux_uncorr *= flux_calibration_factor

            self.flux_units = 'erg/s/cm2/nm' # update flux units
            print(f' Flux calibrated to {self.flux_units}')
        else:
            # normalize flux per order ignoring tellurics for calculating median flux
            print(f' Normalizing flux per order')
            self.normalize_flux_per_order(fun='median', tell_threshold=0.0)
        
        if sigma_clip is not None:
            self.sigma_clip(sigma=sigma_clip, filter_width=sigma_clip_window, 
                            replace_flux=True, 
                            fig_name=f'{fig_dir}/sigma_clipped_spec{self.night_label}.pdf',
                            debug=True)
        # plot sigma clipped spectrum
        
        
        # if fig_name is not None:
        #     figs.fig_spec_to_fit(self, fig_name=fig_name)
        if fig_dir is not None:
            # figs.fig_telluric_correction(self, fig_dir=f'{fig_dir}/telluric_correction.pdf')
            figs.fig_spec_to_fit(self, 
                                 overplot_array=self.flux_uncorr,
                                 fig_name=f'{fig_dir}/preprocessed_spec{self.night_label}.pdf') 
        
        print(f' Preprocessing complete!\n')
        return self
    
    def prepare_for_covariance(self):

        # Make a nested array of ndarray objects with different shapes
        self.separation = np.empty((self.n_orders, self.n_dets), dtype=object)
        self.err_eff = np.empty((self.n_orders, self.n_dets), dtype=object)
        
        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                
                # Mask the arrays, on-the-spot is slower
                mask_ij = self.mask_isfinite[i,j]
                # print(f'Order {i}, detector {j}: {mask_ij.sum()} data points')
                wave_ij = self.wave[i,j,mask_ij]

                # Wavelength separation between all pixels within order/detector
                self.separation[i,j] = np.abs(wave_ij[None,:] - wave_ij[:,None])
                self.err_eff[i,j] = np.mean(self.err[i,j,mask_ij])

        return self
        
        
    

class ModelSpectrum(Spectrum):
    flux_units = 'erg/s/cm2/cm'
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

    def make_spline(self, N_knots=5, spline_degree=3):
        '''Create a spline model for the spectrum
        Store new attribute `flux_spline` with the spline model'''
        self.N_knots = N_knots
        self.flux_spline = SplineModel(N_knots, spline_degree)(self.flux) # shape (N_knots, *self.flux.shape)
        return self
    
    
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