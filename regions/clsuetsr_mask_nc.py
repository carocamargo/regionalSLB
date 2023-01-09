#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:07:22 2022

@author: ccamargo
"""
import xarray as xr
import pandas as pd
import numpy as np
#%%
path = '/Volumes/LaCie_NIOZ/data/budget/'
file = 'alt.nc'
ds = xr.open_dataset(path+file)
path = '/Users/ccamargo/Desktop/manuscript_SLB/data/'
dic = pd.read_pickle(path+'budget_v2.pkl')
dic.keys()

#%%
da = xr.Dataset(data_vars={'som':(('lat','lon'),
                                  np.array(dic['som']['mask'])),
                           'dmaps':(('lat','lon'),dic['dmap']['mask'])
                           },
                                  
                coords = {'lat':ds.lat,
                         'lon':ds.lon}
               )

da.to_netcdf(path+'clusters_mask.nc')
