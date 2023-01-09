#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:37:58 2022

@author: ccamargo
"""

import numpy as np
# import xarray as xr
# # import pickle
# import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")

from utils_SLB import unc_test, agree_test, zeta_test
#%%
path = '/Volumes/LaCie_NIOZ/data/budget/'
dic = pd.read_pickle(path+'budget.pkl')
#%%
alt = np.array(dic['alt']['trend']).flatten()
alt_unc = np.array(dic['alt']['unc']).flatten()
comp = np.array(dic['sum']['trend']).flatten()
comp_unc = np.array(dic['sum']['unc']).flatten()
#%%
zeta = np.full_like(alt, np.nan)
unc1 = np.full_like(alt, np.nan)
unc2 = np.full_like(alt, np.nan)
unc3 = np.full_like(alt, np.nan)
unc4 = np.full_like(alt, np.nan)

n = len(alt)
zeta = np.array([zeta_test(alt[i], alt_unc[i], comp[i], comp_unc[i])  if np.isfinite(alt[i])  else np.nan  for i in range(n)])
unc1 = np.array([unc_test(alt[i], alt_unc[i], comp[i], comp_unc[i], method = 'square', how = 'subtract'
                 )  if np.isfinite(alt[i])  else np.nan  for i in range(n)])
unc2 = np.array([unc_test(alt[i], alt_unc[i], comp[i], comp_unc[i], method = 'square', how = 'sum'
                 )  if np.isfinite(alt[i])  else np.nan  for i in range(n)])
unc3 = np.array([unc_test(alt[i], alt_unc[i], comp[i], comp_unc[i], method = 'linear'
                 )  if np.isfinite(alt[i])  else np.nan  for i in range(n)])
unc4 = np.array([agree_test(alt[i], alt_unc[i], comp[i], comp_unc[i] )  if np.isfinite(alt[i])  else np.nan  for i in range(n)])

n_cells = len(alt[np.isfinite(alt)])
n_closed_z = len(zeta[zeta==1]) / n_cells * 100
n_closed_u1 = len(unc1[unc1==1]) / n_cells * 100
n_closed_u2 = len(unc1[unc2==1]) / n_cells * 100
n_closed_u3 = len(unc1[unc3==1]) / n_cells * 100
n_closed_u4 = len(unc1[unc4==1]) / n_cells * 100

# for i in range(n):
#     zeta[i] = zeta_test(alt[i], )


#%%
df = dic['som']['df']
df = df[['alt_tr', 'alt_unc', 'sum_tr', 'sum_unc']]
sum_tr = np.array(df['sum_tr'])
alt_tr = np.array(df['alt_tr'])
sum_unc = np.array(df['sum_unc'])
alt_unc = np.array(df['alt_unc'])

res_tr = np.array(df['alt_tr'] - df['sum_tr'])
alt_low = np.array(df['alt_tr'] - df['alt_unc'])
alt_high = np.array(df['alt_tr'] + df['alt_unc'])
sum_low = np.array(df['sum_tr'] - df['sum_unc'])
sum_high = np.array(df['sum_tr'] + df['sum_unc'])

n = len(alt_low)
agree = np.zeros((n))
for i in range(n):
    if  alt_low[i] <= sum_tr[i] <= alt_high[i]:
        agree[i] = 1
    elif sum_low[i] <= alt_tr[i] <= sum_high[i]:
        agree[i] = 1
    else:
        agree[i] = 0
   #%%     
def agree_ci(a,a_sig,b,b_sig):
    '''
    Do measurmenets agree within uncertaities?
    1 = agree
    0 = disagree

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    a_sig : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    b_sig : TYPE
        DESCRIPTION.

    Returns
    -------
    agree : TYPE
        DESCRIPTION.

    '''
    
    if  a + a_sig > b - b_sig and a - a_sig < b + b_sig:
        agree = 1
    else:
        agree = 0  
    
    return agree
agree = [agree_ci(alt_tr[i], alt_unc[i], sum_tr[i], sum_unc[i]) for i in range(n)]
    
    # if alt_low[i] < sum_high[i] and sum_high[i]< alt_high[i]:
    #     agree[i] = 1
    # elif alt_high[i] > sum_low[i] and sum_low[i] <alt_low[i]:
    #     agree[i] = 1
    # else:
    #     agree[i] = 0
    #     print(alt_high[i])
    #     print(alt_low[i])
    #     print(sum_high[i])
    #     print(sum_low[i])
#%%
def zeta_test(a,a_sig,b,b_sig):
    '''
    Test whether values agree acocoridng to zeta test
    1 = agree
    0 = disagree
    0.5 = tension zone

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    a_sig : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    b_sig : TYPE
        DESCRIPTION.

    Returns
    -------
    score : TYPE
        DESCRIPTION.

    '''
    zeta = np.abs( 
        (a-b)/ (np.sqrt(a_sig**2 + b_sig**2))
        )
    
    if zeta < 1:
        score = 1 
    elif 1 < zeta <3:
        score = 0.5
    else:
        score = 0
    return score

zeta = np.zeros((n))
for i in range(n):
    zeta[i] = zeta_test(alt_tr[i], alt_unc[i], sum_tr[i], sum_unc[i])

#%% 
def unc_test(a,a_sig,b,b_sig, method = 'square', how = 'sum'):
    
    res = np.array(a - b)
    if method =='square':
        if how =='sum':
            res_unc = np.sqrt( a_sig**2 + b_sig**2)  
        elif how =='subtract':
            res_unc= np.sqrt(a_sig**2 - b_sig**2)
        else:
            raise('Method not recongized')
    elif method =='linear':
        if how =='sum':
            res_unc = np.array(a_sig + b_sig) 
        elif how =='subtract':
            res_unc= np.array(a_sig - b_sig) 
        else:
            raise('Method not recongized')   
    else:
        raise ('Method not recognized')
        
    if np.abs(res) < res_unc:
        agree = 1
    else:
        agree = 0
        
    return agree
    

