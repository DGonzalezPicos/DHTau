import numpy as np
import pickle


def pickle_save(file, object_to_pickle):

    with open(file, 'wb') as f:
        pickle.dump(object_to_pickle, f)
        
def pickle_load(file):
    
    with open(file, 'rb') as f:
        pickled_object = pickle.load(f)

    return pickled_object

def quantiles(x, q, weights=None, axis=-1):
    '''
    Compute (weighted) quantiles from an input set of samples.
    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.
    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    '''

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError('Quantiles must be between 0. and 1.')

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q), axis=axis)
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError('Dimension mismatch: len(weights) != len(x).')
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles
    
def weigh_alpha(contr_em, pressure, temperature, ax, alpha_min=0.8, plot=False):
    ''' Overplot white areas on the temperature-pressure diagram to indicate
    where the emission contribution is low. This is done by weighing the
    opacity by the emission contribution and then setting a minimum opacity
    value.
    '''

    from scipy.interpolate import interp1d
    contr_em_weigh = contr_em / contr_em.max()
    contr_em_weigh_interp = interp1d(pressure, contr_em_weigh)

    # extended vector (oversampled)
    p = np.logspace(np.log10(pressure.min()), np.log10(pressure.max()), 200)
    if isinstance(alpha_min, float):
        alpha_min_vec = np.ones_like(p) * alpha_min
    else:
        alpha_min_vec = np.array(alpha_min)
    
    alpha_list = []
    for i_p in range(len(p)-1):
        mean_press = np.mean([p[i_p], p[i_p+1]])
        # print(f'{i_p}: alpha_min = {alpha_min_vec[i_p]}')
        alpha = min(1. - contr_em_weigh_interp(mean_press), alpha_min_vec[i_p])
        # print(f'{i_p}: alpha = {alpha}')
        if plot:
            ax.fill_between(temperature, p[i_p+1], p[i_p], color='white',alpha=alpha,
                            lw=0, rasterized=True, zorder=4)
        alpha_list.append(alpha)
    return alpha_list