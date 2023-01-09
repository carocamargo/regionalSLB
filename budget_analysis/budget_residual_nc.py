#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:33:05 2022

@author: ccamargo
"""

import numpy as np
# import xarray as xr
# # import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")
import xarray as xr
from utils_SLB import unc_test, agree_test, zeta_test
#%%
path = '/Volumes/LaCie_NIOZ/data/budget/'
dic = pd.read_pickle(path+'budget_v2.pkl')
time = np.array(dic['dims']['time']['tdec'])
lat = np.arange(-90,90)
lon=np.arange(0,360)
#%% 1 deg
key = 'res'

data = np.array(dic[key]['ts'])
da=xr.Dataset(data_vars={'data':(('time','lat','lon'),data)},
                 coords={'time':time,
                     'lat':lat,
                         'lon':lon})
da.to_netcdf(path+'ts/res_1deg_v2.nc')
#%% som
key ='som' # cluster
mask = np.array(dic[key]['mask']) # clusters mask
n = dic[key]['n']
df = dic[key]['df']
alt = np.array(dic['alt']['ts'])
comp = np.array(dic['sum']['ts'])
res = np.array(alt-comp)
#% %
cluster_ts = np.zeros((len(time),n))
fig = plt.figure(figsize=(20,10))
for i in range(n): 
    icluster = int(i+1)
    mask_tmp = np.array(mask)
    mask_tmp[np.where(mask_tmp!=icluster)] = np.nan
    mask_tmp[np.isfinite(mask_tmp)]= 1
    
    y_alt = np.nanmean(alt*mask_tmp,axis=(1,2)) - np.nanmean(alt*mask_tmp)
    y_comp = np.nanmean(comp*mask_tmp,axis=(1,2)) - np.nanmean(comp*mask_tmp)
    cluster_ts[:,i] = np.array(y_alt - y_comp)
    ax = plt.subplot(6,3,icluster)
    plt.plot(y_alt,label = 'alt',linewidth=2)
    plt.plot(y_comp, label='sum',linewidth=2,linestyle='--')
    plt.plot(y_alt-y_comp, label='res',alpha = 0.5)
    plt.title('Cluster {}'.format(icluster))
    plt.legend()
plt.tight_layout()
plt.show()
#% %
data = np.array(cluster_ts)
da=xr.Dataset(data_vars={'data':(('time','cluster'),data)},
                 coords={'time':time,
                     'cluster':np.arange(0,n),
                         })
da.to_netcdf(path+'ts/res_{}_v2.nc'.format(key))

#%% dmaps
key ='dmap' # cluster
mask = np.array(dic[key]['mask']) # clusters mask
n = dic[key]['n']
df = dic[key]['df']
alt = np.array(dic['alt']['ts'])
comp = np.array(dic['sum']['ts'])
res = np.array(alt-comp)
#%%
cluster_ts = np.zeros((len(time),n))
for i in range(n): 
    icluster = int(i+1)
    mask_tmp = np.array(mask)
    mask_tmp[np.where(mask_tmp!=icluster)] = np.nan
    mask_tmp[np.isfinite(mask_tmp)]= 1
    
    y_alt = np.nanmean(alt*mask_tmp,axis=(1,2)) - np.nanmean(alt*mask_tmp)
    y_comp = np.nanmean(comp*mask_tmp,axis=(1,2)) - np.nanmean(comp*mask_tmp)
    cluster_ts[:,i] = np.array(y_alt - y_comp)

#%%
data = np.array(cluster_ts)
da=xr.Dataset(data_vars={'data':(('time','cluster'),data)},
                 coords={'time':time,
                     'cluster':np.arange(0,n),
                         })
da.to_netcdf(path+'ts/res_{}_v2.nc'.format(key))


