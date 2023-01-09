#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:24:31 2022

@author: ccamargo
"""
import xarray as xr
import pandas as pd
import numpy as np

#%%
# path = '/Users/ccamargo/Desktop/manuscript_SLB/data/'
# path = '/Volumes/LaCie_NIOZ/data/budget/'
path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
path = path_to_data+'pub/'
# path_save = path_to_data+'pub/'
file = 'budget.pkl'
dic = pd.read_pickle(path_to_data+file)
df = pd.read_pickle(path+'dmaps_trends.pkl')

for key in ['dmap','som']:
    # key = 'dmap'
    #% %
    #% 
    if key=='dmap':
        mask = np.array(dic[key]['mask'])
        
        for ig in [1,4,54]:
            mask[mask==ig]=np.nan
        for ig in [2,12]:
            mask[mask==ig]=np.nan
        mask01 = np.array(mask)
        mask01[np.isfinite(mask)]=1
        mask01[np.isnan(mask)]=np.nan
        
        m = np.array(mask01)
        for i in np.unique(mask[np.isfinite(mask)]):
            m[mask==int(i)] = np.array(df[df['Domain_number']==int(i)].index)[0]
        mask=np.array(m)
        # n_clusters = dic[key]['n']
    else:
        mask = np.array(dic[key]['mask'])
        mask01 = np.array(mask)
        mask01[np.isfinite(mask)]=1
        mask01[np.isnan(mask)]=np.nan
    
    da=xr.Dataset(data_vars={
                            'mask':(('lat','lon'),mask),
                            # 'alt_trend':(('lat','lon'),np.array(dic[key]['alt']['trend'])),
                            # 'alt_unc':(('lat','lon'),np.array(dic[key]['alt']['unc'])),
                            # 'sum_trend':(('lat','lon'),np.array(dic[key]['alt']['trend'])),
                            # 'alt_unc':(('lat','lon'),np.array(dic[key]['alt']['unc'])),
                            
                            },
                     coords={'lat':dic['dims']['lat']['xr'],
                             'lon':dic['dims']['lon']['xr']})
    for var in ['alt','steric','barystatic','dynamic','sum','res']:
        da[var+'_trend'] = (('lat','lon'),np.array(dic[key][var]['trend']))
        da[var+'_unc'] = (('lat','lon'),np.array(dic[key][var]['unc']))
        
    da.to_netcdf(path+key+'.nc')
    
    