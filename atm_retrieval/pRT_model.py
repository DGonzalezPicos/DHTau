import numpy as np

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from parameters import Parameters
from spectrum import DataSpectrum, ModelSpectrum

from chemistry import Chemistry
from PT_profile import PT


class pRT_model:
    
    def __init__(self,
                 line_species_dict, 
                 d_spec, 
                 mode='lbl', 
                 lbl_opacity_sampling=5,  # set to 3 for accuracy, 5 for speed
                 rayleigh_species=['H2', 'He'], 
                 continuum_opacities=['H2-H2', 'H2-He'], 
                 log_P_range=(-5,2), 
                 n_atm_layers=30, 
                 rv_range=(-50,50)):
        
        # Read in attributes of the observed spectrum
        self.d_wave          = d_spec.wave
        self.d_mask_isfinite = d_spec.mask_isfinite
        self.d_resolution    = d_spec.resolution

        self.line_species_dict = line_species_dict
        self.line_species = [line_species_dict[key] for key in line_species_dict.keys()]
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling

        self.rayleigh_species  = rayleigh_species
        self.continuum_species = continuum_opacities
        
        self.rv_max = max(np.abs(list(rv_range)))
        
        # Set up the pressure-temperature profile
        self.pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)


    def get_atmospheres(self, CB_active=False):

        # pRT model is somewhat wider than observed spectrum
        if CB_active:
            self.rv_max = 1000
        wave_pad = 1.1 * self.rv_max/(nc.c*1e-5) * self.d_wave.max()

        self.wave_range_micron = np.concatenate(
            (self.d_wave.min(axis=(1,2))[None,:]-wave_pad, 
                self.d_wave.max(axis=(1,2))[None,:]+wave_pad
            )).T
        self.wave_range_micron *= 1e-3

        self.atm = []
        for wave_range_i in self.wave_range_micron:
            
            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                line_species=self.line_species, 
                rayleigh_species=self.rayleigh_species, 
                continuum_opacities=self.continuum_species, 
                wlen_bords_micron=wave_range_i, 
                mode=self.mode, 
                lbl_opacity_sampling=self.lbl_opacity_sampling, 
                )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)
            self.atm.append(atm_i)
            
    def __call__(self,  
            params, 
            get_contr=False, 
            get_full_spectrum=False, 
            ):
        '''
        Create a new model spectrum with the given arguments.

        Input
        -----
        params : dict
            Parameters of the current model.
        get_contr : bool
            If True, compute the emission contribution function. 

        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class. 
        '''

        # Update certain attributes
        self.mass_fractions = self.get_mass_fractions(params)
        self.temperature    = self.get_temperature(params)
        self.params = params

        # Generate a model spectrum
        m_spec = self.get_model_spectrum(
            get_contr=get_contr, get_full_spectrum=get_full_spectrum
            )
        return m_spec
    def get_model_spectrum(self, get_contr=False, get_full_spectrum=False):
        '''
        Generate a model spectrum with the given parameters.

        Input
        -----
        get_contr : bool
            If True, computes the emission contribution 
            and cloud opacity. Updates the contr_em and 
            opa_cloud attributes.
        
        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class
        '''

        # Loop over all orders
        wave = np.ones_like(self.d_wave) * np.nan
        flux = np.ones_like(self.d_wave) * np.nan

        self.wave_pRT_grid, self.flux_pRT_grid = [], []

        for i, atm_i in enumerate(self.atm):
            
            # Compute emission spectra
            atm_i.calc_flux(
                self.temperature, 
                self.mass_fractions, 
                gravity=10.0**self.params['log_g'], 
                mmw=self.mass_fractions['MMW'],  
                contribution=get_contr, 
                )
            wave_i = nc.c / atm_i.freq
            # flux_i = np.nan_to_num(atm_i.flux, nan=0.0)
            flux_i = np.where(np.isfinite(atm_i.flux), atm_i.flux, 0.0)
            overflow = np.log(atm_i.flux) > 20
            atm_i.flux[overflow] = 0.0
            
            flux_i = atm_i.flux *  nc.c / (wave_i**2)

            # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
            # flux_i /= 1e7
            flux_i = flux_i * 1e-7

            # Convert [cm] -> [nm]
            wave_i *= 1e7

            # Convert to observation by scaling with planetary radius
            flux_i *= (
                (self.params['R_p']*nc.r_jup_mean) / \
                (1e3/self.params['parallax']*nc.pc)
                )**2

            # Create a ModelSpectrum instance
            m_spec_i = ModelSpectrum(
                wave=wave_i, flux=flux_i, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            
            # Apply radial-velocity shift, rotational/instrumental broadening
            m_spec_i.shift_broaden_rebin(
                rv=self.params['rv'], 
                vsini=self.params['vsini'], 
                epsilon_limb=self.params['epsilon_limb'], 
                out_res=self.d_resolution, 
                in_res=m_spec_i.resolution, 
                rebin=False, 
                )
            if get_full_spectrum:
                # Store the spectrum before the rebinning
                self.wave_pRT_grid.append(m_spec_i.wave)
                self.flux_pRT_grid.append(m_spec_i.flux)

            # Rebin onto the data's wavelength grid
            m_spec_i.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)

            wave[i,:,:] = m_spec_i.wave
            flux[i,:,:] = m_spec_i.flux
            
            # TODO: get contribution function
            # if get_contr:
            
            
        # Create a new ModelSpectrum instance with all orders
        m_spec = ModelSpectrum(
            wave=wave, 
            flux=flux, 
            lbl_opacity_sampling=self.lbl_opacity_sampling, 
            multiple_orders=True, 
            )


        # Save memory, same attributes in DataSpectrum
        del m_spec.wave, m_spec.mask_isfinite

        return m_spec
    
    def get_mass_fractions(self, params):
        VMRs = {key: 10.0**params[f'log{key}'] for key in self.line_species_dict.keys()}
        
        self.chem = Chemistry(self.pressure)
        self.mass_fractions = self.chem(VMRs)
        return self.mass_fractions
    
    def get_temperature(self, params):
        
        assert hasattr(params, 'log_P_knots'), 'params must have log_P_knots attribute'
        # Select the temperature knots
        T_knots = [k for k in params.keys() if k.startswith('T') and len(k)==2]
        
        self.PT = PT(self.pressure)
        self.temperature = self.PT.spline(params['log_P_knots'], T_knots)               
        return self.temperature
        
        
    
if __name__=='__main__':
    
    
    file_data = 'data/crires_example_spectrum.dat'
    d_spec = DataSpectrum(file_target=file_data, slit='w_0.4', flux_units='erg/s/cm2/cm')
    
    free_params = {
        # general properties
        # 'Rp'    : ([0.5, 2.0], r'$R_p$ [R$_J$]'),
        # 'log_g' : ([3.0, 5.5], r'$\log(g)$ [cm/s$^2$]'),
        'vsini' : ([1.0, 20.0], r'$v \sin(i)$ [km/s]'),
        'RV'    : ([-30.0, 30.0], r'RV [km/s]'),
        
        # chemistry
        'log12CO' : ([-12, -2], r'$\log$(CO)'),
        'logH2O'  : ([-12, -2], r'$\log$(H$_2$O)'),
        'logNa'   : ([-12, -2], r'$\log$(Na)'),
        
        # temperature profile
        'T1' : ([2000, 20000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
        'T2' : ([1000, 20000], r'$T_2$ [K]'),
        'T3' : ([300,  10000], r'$T_3$ [K]'),
        'T4' : ([300,  5000],  r'$T_4$ [K]'),
    }

    constant_params = {
        'Rp'    : 1.0,
        'log_g' : 4.0,
    }
    
    cube = np.random.rand(len(free_params))
    parameters = Parameters(free_params, constant_params)
    sample = parameters(cube)
    parameters.add_sample(sample)
    
    
    ## Prepare pRT model
    line_species_dict = {
        
        'H2O': 'H2O_pokazatel_main_iso',
    }
    pRT = pRT_model(line_species_dict=line_species_dict,
                    d_spec=d_spec,
                    mode='lbl',
                    lbl_opacity_sampling=5,
                    rayleigh_species=['H2', 'He'],
                    continuum_opacities=['H2-H2', 'H2-He'],
                    log_P_range=(-5,2),
                    n_atm_layers=30,
                    rv_range=(-50,50))
    

    pRT.get_mass_fractions(parameters.params)
    # m_spec = pRT(mass_fractions, temperature, params, get_contr=False, get_full_spectrum=False)
    # m_spec = pRT(params, get_contr=False, get_full_spectrum=False)            