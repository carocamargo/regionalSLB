#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:12:06 2022

@author: ccamargo
"""

import numpy as np
import xarray as xr
# import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")

from utils_SLB import cluster_mean
#%% get budget components
path = '/Volumes/LaCie_NIOZ/data/budget/'
dic = pd.read_pickle(path+'budget_v2.pkl')
#%% sensitivity to budget combination 
da = xr.open_dataset('/Volumes/LaCie_NIOZ/data/budget/combinations.nc')
budget = 'som'
mask_clusters = dic[budget]['mask']
n_clusters = dic[budget]['n']

n_pos = len(da.comb)
cluster_combos_res = np.zeros((n_clusters,n_pos))
cluster_combos_unc = np.zeros((n_clusters,n_pos))
cluster_combos_sig = np.zeros((n_clusters,n_pos))

for ipos in range(n_pos):
    res=np.array(da.res[ipos])
    unc = np.array(da.unc[ipos])
    mat = np.zeros((n_clusters))
    mat2 = np.zeros((n_clusters,))
    # mat3 = np.zeros((n_clusters))
    
    for i in range(n_clusters):
        icluster = i+1
        mask=np.array(mask_clusters)
        mask[np.where(mask!=icluster)]=np.nan
        mask[np.isfinite(mask)]=1
        
        # tmp[i,mask==1] = cluster_mean(np.array(dic[label]['trend']),mask, lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        # tmp2[i,mask==1] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        mat[i] = cluster_mean(res,mask,lat=np.array(da.lat),lon=np.array(da.lon),norm=False )
        mat2[i] = cluster_mean(unc,mask,lat=np.array(da.lat),lon=np.array(da.lon),norm=False )
        # mat3[i] = cluster_mean(np.array( np.abs(dic['alt']['unc'] - sum_comps_unc) ),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        cluster_combos_res[i,ipos] = mat[i]
        cluster_combos_unc[i,ipos] = mat2[i]
        if mat[i]<mat2[i]:
            cluster_combos_sig[i,ipos] = 0 # zero if closed
        else:
            cluster_combos_sig[i,ipos] = 1 # 1 if open

#%% open best dataset
pwd ="/Volumes/LaCie_NIOZ/data/budget/"
dicb = pd.read_pickle(pwd+"best_rmse.pkl")
print(dicb)
variables = np.array(da['dataset'])
n = []
for var in variables:
    if var=='ste': var='steric'
    if var=='bar': var='barystatic'
    if var=='dyn': var = 'dynamic'
    
    if var=='dynamic':
        n.append('ENS') # we only have trends for ENS for now
    elif var=='steric':
        n.append(dicb[var].upper())
    else:
        n.append(dicb[var])
print(n) 

#%% plot
plt.figure(dpi=300)
ax=plt.subplot(111)
alpha=0.1
x=range(n_clusters)
for ipos in range(n_pos):
    if np.any(np.isfinite(cluster_combos_res[:,ipos])):
        col = ['salmon' if sig==1 else 'mediumslateblue' for sig in cluster_combos_sig[:,ipos]]
        sc=plt.scatter(x,cluster_combos_res[:,ipos],
                        c=col,
                        # c=cluster_combos_sig[:,ipos],
                       alpha=alpha/2)

# plot just one to get lengend
ind=np.where(cluster_combos_sig[:,ipos]==1)[0][0]
plt.scatter(x[ind],cluster_combos_res[ind,ipos],
            c=col[ind],
            alpha=alpha,label='open')
ind=np.where(cluster_combos_sig[:,ipos]==0)[0][0]
plt.scatter(x[ind],cluster_combos_res[ind,ipos],
            c=col[ind],
            alpha=alpha,label='closed')

## ENS
names = np.array(da.names)
ipos =[]
ipos = [i for i in range(len(da.comb)) 
        if np.all(names[i,:]==['ENS'])][0]
ipos = [i for i in range(len(da.comb)) 
        if np.all(names[i]==['ENS','ENS','IMB_WGP','ENS'])][0]
col= ['salmon' if sig==1 else 'mediumslateblue' for sig in cluster_combos_sig[:,ipos]]
plt.scatter(x,cluster_combos_res[:,ipos],
            # c='black',
            marker='s',
            c=col,
            # label='ENS'
            )
if np.any(cluster_combos_sig[:,ipos]==1):
    ind=np.where(cluster_combos_sig[:,ipos]==1)[0][0]
    plt.scatter(x[ind],cluster_combos_res[ind,ipos],
                marker='s',
                c=col[ind],
                # alpha=alpha,
                label='ENS - open')
if np.any(cluster_combos_sig[:,ipos]==0):
    ind=np.where(cluster_combos_sig[:,ipos]==0)[0][0]
    plt.scatter(x[ind],cluster_combos_res[ind,ipos],
                c=col[ind],
                marker='s',
                # alpha=alpha,
                label='ENS - closed')
plt.legend(ncol=2)
plt.ylabel('Residual trend mm/yr')
plt.xlabel('SOM cluster number')
plt.show()

#%%
fontsize=20
size=75
y = cluster_combos_res.flatten()
y_sig = cluster_combos_sig.flatten()
x = range(len(y))
col= ['salmon' if sig==1 else 'mediumslateblue' for sig in y_sig]
#% %
plt.figure(dpi=300,figsize=(15,10))
# plot vertical lines to separate clusters
xs = np.linspace(1,len(x),18)
for xc in xs:
    plt.axvline(x=xc, color='gray', linestyle='--',alpha=0.5)
# scatter all options:
alpha=0.1
plt.scatter(x,y,c=col,alpha=alpha)
# plot for legend:
ind=np.where(y_sig==1)[0][0]
plt.scatter(x[ind],y[ind],
            c=col[ind],
            alpha=alpha,label='open')
ind=np.where(y_sig==0)[0][0]
plt.scatter(x[ind],y[ind],
            c=col[ind],
            alpha=alpha,label='closed')

# plot ensemble
col= ['salmon' if sig==1 else 'blue' for sig in y_sig]

ipos = [i for i in range(len(da.comb)) 
        if np.all(names[i,:]==['ENS'])][0]
ipos = [i for i in range(len(da.comb)) 
        if np.all(names[i]==['ENS','ENS','IMB_WGP','ENS'])][0]
y2 = np.full_like(y,np.nan)
y2[ipos:len(y):n_pos] = y[ipos:len(y):n_pos]
y2_sig = np.full_like(y,np.nan)
y2_sig[ipos:len(y):n_pos] = y_sig[ipos:len(y):n_pos]

# col= ['salmon' if sig==1 else 'mediumslateblue' for sig in cluster_combos_sig[:,ipos]]
plt.scatter(x,y2,
            # c='black',
            marker='s',
            c=col,
            s=size,
            # label='ENS'
            )
if np.any(y_sig[ipos:len(y):n_pos]==1):
    ind=np.where(y2_sig==0)[0][0]
    plt.scatter(x[ind],y[ind],
                marker='s',
                c=col[ind],
                # alpha=alpha,
                label='ENS - open')
if np.any(y_sig[ipos:len(y):n_pos]==0):
    ind=np.where(y2_sig==0)[0][0]
    plt.scatter(x[ind],y[ind],
                marker='s',
                
                c=col[ind],
                # alpha=alpha,
                label='ENS - closed')

# plot best rmse commbination
col= ['salmon' if sig==1 else 'darkviolet' for sig in y_sig]

ipos = [i for i in range(len(da.comb)) 
        if np.all(names[i]==n)][0]
y2 = np.full_like(y,np.nan)
y2[ipos:len(y):n_pos] = y[ipos:len(y):n_pos]
y2_sig = np.full_like(y,np.nan)
y2_sig[ipos:len(y):n_pos] = y_sig[ipos:len(y):n_pos]

# col= ['salmon' if sig==1 else 'mediumslateblue' for sig in cluster_combos_sig[:,ipos]]
plt.scatter(x,y2,
            # c='black',
            marker='^',
            c=col,
            s=size*2,
            # label='ENS'
            )
if np.any(y_sig[ipos:len(y):n_pos]==1):
    ind=np.where(y2_sig==1)[0][0]
    plt.scatter(x[ind],y[ind],
                marker='s',
                c=col[ind],
                # alpha=alpha,
                label='RMSE - open')
if np.any(y_sig[ipos:len(y):n_pos]==0):
    ind=np.where(y2_sig==0)[0][0]
    plt.scatter(x[ind],y[ind],
                marker='s',
                
                c=col[ind],
                # alpha=alpha,
                label='RMSE - closed')
plt.ylabel('mm/yr',fontsize=fontsize)
plt.xlabel('combinations',fontsize=fontsize)

plt.legend(fontsize=fontsize)
