from typing import Any
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import os
import time
import json
import copy
import corner

import atm_retrieval.figures as figs



class CallBack:
    
    def __init__(self,
                 d_spec,
                 evaluation=False,
                 n_samples_to_use=2000,
                 posterior_color='C0', 
                 bestfit_color='C1', 
                 PT_color='orangered',
                 ):
        
        self.d_spec = d_spec
        self.evaluation = evaluation
        self.n_samples_to_use = n_samples_to_use
        
        self.posterior_color = posterior_color
        self.bestfit_color = bestfit_color
        self.PT_color = PT_color
        
        
        
    def __call__(self, 
                 Param,
                 LogLike,
                 Cov,
                 PT,
                 Chem,
                 m_spec,
    ):
        
        pass