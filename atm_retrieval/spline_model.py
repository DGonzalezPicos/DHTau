        
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class SplineModel:
    
    def __init__(self, N_knots=10, spline_degree=3):
        ''' Initialize the spline model (Adapted from github.com/jruffio/breads).
        Parameters
        -----------
        N_knots: int
            Number of knots to use for the spline decomposition.
        spline_degree: int
            Degree of the spline interpolation (default: 3).
    ''' 
     
        self.N_knots = N_knots
        self.spline_degree = spline_degree
        
        if self.N_knots <= self.spline_degree:
            self.spline_degree = self.N_knots - 1
            # print('Number of knots must be smaller than spline degree. Setting N_knots = spline_degree - 1.')
  
    def get_spline_model(self):
        ''' Compute a spline based linear model. '''
     
        assert hasattr(self, 'x_knots'), 'Please call the __call__ method first.'
        assert hasattr(self, 'x_samples'), 'Please call the __call__ method first.'
  
        assert np.size(self.x_knots) > 1, 'x_knots must be a list of ndarrays/lists.'
        assert np.size(self.x_knots) > self.spline_degree, 'spline_degree must be smaller than the number of knots.'
  
        self.x_knots = np.atleast_2d(self.x_knots)
        M_list = []
        for nodes in self.x_knots:
            M = np.zeros((np.size(self.x_samples), np.size(nodes)))
            min,max = np.min(nodes),np.max(nodes)
            inbounds = np.where((min<self.x_samples)&(self.x_samples<max))
            _x = self.x_samples[inbounds]
            for chunk in range(np.size(nodes)):
                tmp_y_vec = np.zeros(np.size(nodes))
                tmp_y_vec[chunk] = 1
                spl = InterpolatedUnivariateSpline(nodes, tmp_y_vec, k=self.spline_degree, ext=0)
                M[inbounds[0], chunk] = spl(_x)
            M_list.append(M)
        return np.concatenate(M_list, axis=1)
     
     
  
    def __call__(self, spec):
        ''' Decompose the model in N_knots subcomponents.
        Parameters
        -----------
            spec: np.ndarray
                1D array containing the spectrum to be decomposed.
    
        Returns
        --------
            spline_spec: np.ndarray
                2D array containing the spline model (shape = (N_knots, spec.size)'''
        self.x_samples = np.arange(spec.size)
        self.x_knots = np.linspace(-1, self.x_samples.size+1, self.N_knots)
  
        self.spline_spec = self.get_spline_model().T * spec
        return self.spline_spec




## test the functions above

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 1000)
    # insert gaussian peaks at given positions
    def gaussian(x, x0, sigma):
        return np.exp(-(x-x0)**2 / (2*sigma**2))

    peaks = [2, 4, 6, 8]

    slope =  x * 0.1
    spec = np.ones_like(x) * slope
    for p in peaks:
        spec += gaussian(x, p, np.random.normal(0.05, 0.02))

    # Get the Spline model for the spectrum
    model = SplineModel(N_knots=5, spline_degree=3)(spec)
    print(f' shape model: {model.shape}')
 

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(x, spec, label='model input', color='k', alpha=0.9, lw=2)

    for i, m in enumerate(model):
        ax.plot(x, m, alpha=0.5, label=f'spline {i}')
        
    # add the components of the model
    ax.plot(x, np.sum(model, axis=0), label='model reconstructed', alpha=1.0, ls='--', c='magenta')
    
    
    # ax.plot(x, slope, label='slope', c='r', ls=':')
    ax.legend()
    plt.savefig("test.pdf")
    plt.show()