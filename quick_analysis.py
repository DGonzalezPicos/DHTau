import numpy as np
import pathlib
import matplotlib.pyplot as plt

import corner
import pymultinest
import pickle


from atm_retrieval.retrieval import Retrieval




run = 'testing_004'
run_dir = pathlib.Path(f'retrieval_outputs/{run}')

assert run_dir.exists()

with open(run_dir / 'retrieval.pickle', 'rb') as f:
    ret = pickle.load(f)
    

# Load the equally-weighted posterior distribution
# Set-up analyzer object
analyzer = pymultinest.Analyzer(
    n_params=ret.parameters.n_params, 
    outputfiles_basename=f'{str(run_dir)}/pmn_',
    )
stats = analyzer.get_stats()

# Load the equally-weighted posterior distribution
posterior = analyzer.get_equal_weighted_posterior()
posterior = posterior[:,:-1]

# Read the parameters of the best-fitting model
bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])
param_labels = np.array(list(ret.parameters.param_mathtext.values()))
# print(bestfit_params)
for i in range(len(bestfit_params)):
    print(f'{param_labels[i]}: {bestfit_params[i]}')

# plot corner
fig = corner.corner(posterior, labels=param_labels, truths=bestfit_params)
fig.savefig(run_dir / 'corner.png')
print(f'Corner plot saved to:, {run_dir}/corner.png')
plt.show()