#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:57:39 2022

@author: ccamargo
"""

import xarray as xr



path = "/Volumes/LaCie_NIOZ/data/budget/"

l =  {
       'dynamic':'dynamic_sl',
       'alt':'alt',
      'barystatic': "barystatic_timeseries",
      'steric':'steric_full'
      }
period = ['1993-2017'] # full years
y0,y1=period[0].split('-')
t0='{}-01-01'.format(int(y0))
t1='{}-12-31'.format(int(y1)-1)
for key in l:
    print(key)
    file = l[key]
    
    ds = xr.open_dataset(path + file+ ".nc")
    
    # select time period
    ds = ds.sel(time=slice(t0,t1))
    
    # save time series
    ds.to_netcdf(path+'ts/'+key+'.nc')
