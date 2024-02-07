import numpy as np
import matplotlib.pyplot as plt
import json

import corner

class Parameters:
    
    
    # Dictionary of all possible parameters and their default values
    all_params = {
        # Uncertainty scaling
        'a': 0, 'l': 1,  # Gaussian Process
        'a_f': 0, 'l_f': 1, # Gaussian Process with `flux` scaling
        'beta': 1,  # Uncertainty scaling
    }
    def __init__(self, free_params, constant_params):
        
        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)
        # alias
        self.ndim = self.n_params

        # Create dictionary with constant parameter-values
        self.params = self.all_params.copy()
        self.params.update(constant_params)
        
    def __str__(self):
        out = '** Parameters **\n'
        out += '-'*len(out) + '\n' # add line of dashed hyphens

        out += f'- Free Parameters (n={self.n_params}):\n'
        out += '\n'.join([f'  {key_i} : {self.param_priors[key_i]}' for key_i in self.param_keys])
        out += '\n\n- Constant Parameters:\n'
        out += '\n'.join([f'  {key_i} : {self.params[key_i]}' for key_i in self.params if key_i not in self.param_keys])
        return out
    def __repr__(self):
        return self.__str__()
        
        
        
    @staticmethod
    def uniform_prior(bounds):
        return lambda x: x * (bounds[1] - bounds[0]) + bounds[0]
        
    
    def __call__(self, cube, ndim=None, nparams=None):
        '''Return the transformed parameters
        
        WARNING: only works for Uniform Priors'''
        # self.cube_copy = cube.copy()
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])
        
        for i, key_i in enumerate(self.param_keys):
            cube[i] = self.uniform_prior(self.param_priors[key_i])(cube[i]) 
            self.params[key_i] = cube[i]

        return self.cube_copy
    
    def add_sample(self, sample):
        
        for i, key_i in enumerate(self.param_keys):
            self.params[key_i] = sample[i]
            
        return self
    
    @property
    def random_sample(self):
        return self(np.random.rand(self.ndim))
    
    @property
    def lower_edge_sample(self):
        return self(np.zeros(self.ndim))
    @property
    def upper_edge_sample(self):
        return self(np.ones(self.ndim))

    def sample_prior(self, n=2_000):
        p = np.array([self(np.random.rand(self.ndim)) for i in range(n)])
        fig = corner.corner(p, labels=self.param_keys, plot_density=False, show_titles=True)
        # fig.savefig('sample_prior.pdf', bbox_inches='tight')
        plt.show()
        return p
    
    def save(self, filename):
        '''Save free parameters and constant parameters and priors as json file'''
        import json
        with open(filename, 'w') as f:
            # json.dump(self.params, f)
            free_params = {k:[priors, v] 
                           for k, priors, v 
                           in zip(self.param_keys, self.param_priors.values(), self.param_mathtext.values())}
            json.dump({'free_params': free_params, 
                       'constant_params': self.params}, f)
        print(f'Parameters saved to {filename}')
        
    @classmethod
    def load(cls, filename):
        '''Load free parameters and constant parameters and priors from json file'''
            
        with open(filename, 'r') as f:
            params_data = json.load(f)
        
        return cls(params_data['free_params'], params_data['constant_params'])
            
    
       
       
if __name__ == '__main__':
    
    
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
    parameters = Parameters(free_params, constant_params)
    parameters.sample_prior(n=1000)