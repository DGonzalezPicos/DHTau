import numpy as np
import pathlib
import matplotlib.pyplot as plt

import corner
import pymultinest
import pickle

from atm_retrieval.utils import quantiles

run = 'testing_004'
run_dir = pathlib.Path(f'retrieval_outputs/{run}')

assert run_dir.exists()

posterior = np.load('posteriors.npy')

Q = np.array([quantiles(posterior[:,i], q=[0.16,0.5,0.84]) \
                for i in range(posterior.shape[1])]
                )
        
ranges = np.array(
            [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
                for q_i in Q]
            )

err = np.array(
            [((q_i[0]-q_i[1]), (q_i[2]-q_i[1])) \
                for q_i in Q]
            )

C_ratio = Q[7,1]/Q[8,1]
print(C_ratio)

C_ratio_lower_err = -C_ratio*np.sqrt((err[7,0]/Q[7,1])**2+(err[8,0]/Q[8,1])**2)
print(C_ratio_lower_err)
C_ratio_upper_err = C_ratio*np.sqrt((err[7,1]/Q[7,1])**2+(err[8,1]/Q[8,1])**2)
print(C_ratio_upper_err)



CO_ratio = (Q[7,1]+Q[8,1]+Q[15,1])/(Q[9,1]+Q[10,1]+Q[7,1]+Q[8,1])
print(CO_ratio)

CO_ratio_lower_err = -CO_ratio*np.sqrt(((err[7,0]+err[8,0]+err[15,0])/(Q[7,1]+Q[8,1]+Q[15,1]))**2
                                       +((err[9,0]+err[10,0]+err[7,0]+err[8,0])/(Q[9,1]+Q[10,1]+Q[7,1]+Q[8,1]))**2)
print(C_ratio_lower_err)
CO_ratio_upper_err = CO_ratio*np.sqrt(((err[7,1]+err[8,1]+err[15,1])/(Q[7,1]+Q[8,1]+Q[15,1]))**2
                                       +((err[9,1]+err[10,1]+err[7,1]+err[8,1])/(Q[9,1]+Q[10,1]+Q[7,1]+Q[8,1]))**2)
print(C_ratio_upper_err)
















# with open(run_dir / 'retrieval.pickle', 'rb') as f:
#     ret = pickle.load(f)
    

# # Load the equally-weighted posterior distribution
# # Set-up analyzer object
# analyzer = pymultinest.Analyzer(
#     n_params=ret.parameters.n_params, 
#     outputfiles_basename=f'{str(run_dir)}/pmn_',
#     )
# stats = analyzer.get_stats()

# # Load the equally-weighted posterior distribution
# posterior = analyzer.get_equal_weighted_posterior()
# posterior = posterior[:,:-1]

# # Read the parameters of the best-fitting model
# bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])
# param_labels = np.array(list(ret.parameters.param_mathtext.values()))
# # print(bestfit_params)
# for i in range(len(bestfit_params)):
#     print(f'{param_labels[i]}: {bestfit_params[i]}')

# # plot corner
# fig = corner.corner(posterior, labels=param_labels, truths=bestfit_params)
# fig.savefig(run_dir / 'corner.png')
# print(f'Corner plot saved to:, {run_dir}/corner.png')
# plt.show()