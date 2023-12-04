import numpy as np
import matplotlib.pyplot as plt
import pathlib

from petitRADTRANS import Radtrans

from PyAstronomy import pyasl # for rotational broadening
from scipy.ndimage import gaussian_filter # for instrumental broadening

def load_sphinx_model(Teff=3100.0, log_g=4.0, logZ=0.0, C_O=0.50):
        
    path = pathlib.Path('data/')
    sign = '+' if logZ >= 0 else '-'
    
    # PT profile
    file = path / f'Teff_{Teff:.1f}_logg_{log_g}_logZ_{sign}{abs(logZ)}_CtoO_{C_O}_atms.txt'
    assert file.exists(), f'File {file} does not exist.'
    t, p = np.loadtxt(file, unpack=True)
    
    # VMRs
    file_chem = path / file.name.replace('atms', 'mixing_ratios')
    
    with open(file_chem, 'r') as f:
        header = f.readline()
        
    header = header.split(',')
    header[0] = 'pressure'
    # remove spaces
    header = [h.strip() for h in header]
    VMRs = np.loadtxt(file_chem, unpack=True)
    VMRs = {k:v for k, v in zip(header, VMRs)}
    
    return t, p, VMRs, file

