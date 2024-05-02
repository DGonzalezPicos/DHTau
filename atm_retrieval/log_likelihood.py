import numpy as np
from scipy.optimize import nnls

from atm_retrieval.spline_model import SplineModel

class LogLikelihood:
    
    def __init__(self, d_spec, n_params, scale_flux=False):
        self.d_spec = d_spec
        self.n_params = n_params
        self.n_orders, self.n_dets, _ = self.d_spec.flux.shape
        
        self.scale_flux = scale_flux
        
        # make 3d
        if len(self.d_spec.flux.shape) == 1:
            self.d_spec.flux = self.d_spec.flux[None, None, :]
            self.d_spec.err  = self.d_spec.err[None, None, :]
            self.d_spec.mask_isfinite = self.d_spec.mask_isfinite[None, None, :]
    
        
    def __call__(self, m_spec, Cov):
        if len(m_spec.flux.shape) == 1:
            print(f'Warning: m_spec.flux.shape = {m_spec.flux.shape}')
            m_spec.flux = m_spec.flux[None, None, :]
            
        # Calculate the log-likelihood
        self.ln_L = 0.0
        # Array to store the linear flux-scaling terms
        N_knots = m_spec.N_knots  # at least 1
        self.f    = np.ones((N_knots+m_spec.N_veiling, self.n_orders, self.n_dets))
        # Array to store the uncertainty-scaling terms
        self.beta = np.ones((self.n_orders, self.n_dets))
        self.m = np.nan * np.ones_like(self.d_spec.flux) # store the full model

        # Loop over all orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = self.d_spec.mask_isfinite[i,j,:]
                
                # Number of (valid) data points
                N_ij = mask_ij.sum()
                
                if N_ij == 0: # skip if no valid data points
                    # print(f'Warning: no valid data points in order {i}, detector {j}')
                    continue
                
                m_flux_ij = m_spec.flux[i,j,mask_ij]
                M_ij = m_flux_ij[np.newaxis, :] # shape (1, n_pixels) initialize design matrix
                
                d_flux_ij = self.d_spec.flux[i,j,mask_ij]
                # d_err_ij  = self.d_spec.err[i,j,mask_ij]
                d_err_ij = Cov[i,j].err # mask already applied in Cov

                # code below deprecated (now including covariance matrix object)
                # Get the log of the determinant (log prevents over/under-flow)
                # log(det(Cov)) = sum(log(diag(Cov))) for Cov = diag(diag(Cov))
                # cov_logdet = np.sum(np.log(d_err_ij**2)) # can be negative
                # cov_ij = np.diag(d_err_ij**2) # covariance matrix
                # inv_cov_ij = np.diag(1/d_err_ij**2)
                
                # DGP (2024-04-16): new covariance matrix
                if Cov[i,j].is_matrix:
                    # Retrieve a Cholesky decomposition
                    Cov[i,j].get_cholesky()
                Cov[i,j].get_logdet()

                # Set up the log-likelihood for this order/detector
                # Chi-squared and optimal uncertainty scaling terms still need to be added
                # equation from Ruffio et al. 2019 (https://arxiv.org/abs/1909.07571)
                ln_L_ij = -0.5 * (N_ij*np.log(2*np.pi) + Cov[i,j].logdet)
               
                f_ij = [1.0]
        
                if m_spec.N_knots > 1:
                    
                    # m_flux_ij_spline = SplineModel(N_knots=m_spec.N_knots, spline_degree=3)(m_flux_ij)
                    
                    # replace single-component matrix with multi-component matrix
                    M_ij = SplineModel(N_knots=m_spec.N_knots, spline_degree=3)(m_flux_ij)
                    # line below for Gaussian Process
                    # f_ij = nnls(np.dot(m_flux_ij_spline, Cov[i,j].solve(m_flux_ij_spline.T)), 
                    #            np.dot(m_flux_ij_spline, Cov[i,j].solve(d_flux_ij)))[0]
                    # assert np.all(np.isfinite(phi)), f'phi = {phi}'
                    # assert not np.all(phi == 0), f'phi = {phi}'
                    # without GP
                    # phi = nnls(m_flux_ij_spline @ inv_cov_ij @ m_flux_ij_spline.T,
                    #             m_flux_ij_spline @ inv_cov_ij @ d_flux_ij)[0]
                               
                    # m_flux_ij_scaled = f_ij @ m_flux_ij_spline
                    
                if m_spec.N_veiling > 0:
                    # build linear model with veiling components
                    
                    M_ij = np.concatenate([M_ij, m_spec.M_veiling[:,mask_ij]], axis=0) # add veiling components
                    # print(f'M.shape = {M_ij.shape}')
                    
                # solve for the linear scaling factors
                try:
                    f_ij = nnls(np.dot(M_ij, Cov[i,j].solve(M_ij.T)),
                                np.dot(M_ij, Cov[i,j].solve(d_flux_ij)))[0]
                except RuntimeError:
                    # print(f' Warning: nnls did not converge for order {i}, detector {j}... fitting without model spectrum')
                    f_ij = np.zeros((M_ij.shape[0]))
                    f_ij[1:] = nnls(np.dot(M_ij[1:,], Cov[i,j].solve(M_ij[1:].T)),
                                np.dot(M_ij[1:,], Cov[i,j].solve(d_flux_ij)))[0]
                
                # convert from matrix to array again with the fitted linear amplitudes
                m_flux_ij_scaled = f_ij @ M_ij
                    
                    
                # else:
                #     m_flux_ij_scaled, f_ij = self.get_flux_scaling(d_flux_ij, m_flux_ij, inv_cov_ij)
                #     f_ij = np.atleast_1d(f_ij)
                    
                # Calculate the residuals
                res_ij = (d_flux_ij - m_flux_ij_scaled)
                
                # Calculate the chi-squared
                # chi_squared_ij_scaled = res_ij @ inv_cov_ij @ res_ij
                chi_squared_ij_scaled = res_ij @ Cov[i,j].solve(res_ij)
                
                # optimal uncertainty scaling terms
                beta_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled)
                # Chi-squared for optimal linear scaling and uncertainty scaling
                chi_squared_ij = 1/beta_ij**2 * chi_squared_ij_scaled
                # reduced chi-squared
                self.chi_squared_reduced = chi_squared_ij / (N_ij - self.n_params)
                
                # Add chi-squared and optimal uncertainty scaling terms to log-likelihood
                ln_L_ij += -(N_ij/2*np.log(beta_ij**2) + 1/2*chi_squared_ij)
                # print(f'ln_L_ij (after chi2 and scaling) = {ln_L_ij:.2f}')

                # Add to the total log-likelihood and chi-squared
                self.ln_L += ln_L_ij
                
                # Store in the arrays
                self.m[i,j,mask_ij] = m_flux_ij_scaled # scaled model flux (same shape as d_spec.flux)
                self.f[:,i,j]    = f_ij
                self.beta[i,j] = beta_ij
        
        return self.ln_L
    
    
    def get_flux_scaling(self, d_flux_ij, m_flux_ij, inv_cov_ij):
        '''
        Following Ruffio et al. (2019). Find the optimal linear 
        scaling parameter to minimize the chi-squared error. 

        Solve for the linear scaling parameter f in:
        (M^T * cov^-1 * M) * f = M^T * cov^-1 * d

        Input
        -----
        d_flux_ij : np.ndarray
            Flux of the observed spectrum.
        m_flux_ij : np.ndarray
            Flux of the model spectrum.
        inv_cov_ij : Covariance class
            Inverse covariance matrix.

        Returns
        -------
        m_flux_ij*f_ij : np.ndarray
            Scaled model flux.
        f_ij : 
            Optimal linear scaling factor.
        '''
        
        # Left-hand side
        # lhs = np.dot(m_flux_ij, cov_ij.solve(m_flux_ij))
        lhs = m_flux_ij @ inv_cov_ij @ m_flux_ij
        # Right-hand side
        rhs = m_flux_ij @ inv_cov_ij @ d_flux_ij
        
        # Return the scaled model flux
        f_ij = rhs / lhs
        return m_flux_ij * f_ij, f_ij
    
    
if __name__ == '__main__':
    
    from spectrum import DataSpectrum, ModelSpectrum
    file_data = '../data/crires_example_spectrum.dat'
    # spec = DataSpectrum(file_target=file, slit='w_0.4').reshape_orders_dets()
    d_spec = DataSpectrum(file_target=file_data, slit='w_0.4', flux_units='erg/s/cm2/cm')
    
    file_model = '../data/prt_fake_spectrum.txt'
    
    m_wave, m_flux = np.loadtxt(file_model, unpack=True)[:2]
    m_spec = ModelSpectrum(wave=m_wave * 1e3, flux=m_flux)
    
    # crop data to fit the model (usually other way around)
    d_spec.crop(m_spec.wave.min(), m_spec.wave.max())
    
    loglike = LogLikelihood(d_spec, n_params=8)
    logL = loglike(m_spec)
    print(f'logL = {logL:.2f}')
    