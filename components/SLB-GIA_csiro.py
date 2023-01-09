#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:56:25 2022

@author: ccamargo
"""


import xarray as xr
import numpy as np
import sys
sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import matplotlib.pyplot as plt
#%%
path = '/Volumes/LaCie_NIOZ/data/altimetry/world/csiro/'
f1 = 'jb_iby_srn_gtn_gin.nc' # NO GIA correction
f2 = 'jb_iby_srn_gtn_giy.nc' # WITH GIA correction

d1 = xr.open_dataset(path+f1)
d2 = xr.open_dataset(path+f2)
tdec = np.array(d1.time_years)
lat = np.array(d1.lat)
lon = np.array(d1.lon)
out = sl.get_reg_trend(tdec, np.array(d1['height']), 
                       lat,lon)
t1 = np.array(out[0])
out = sl.get_reg_trend(tdec, np.array(d2['height']), 
                       lat,lon)
t2 = np.array(out[0])

plt.figure()
plt.pcolor(t2-t1);plt.colorbar()

gia = np.array(t2-t1)
#%% 
da = xr.Dataset(
    data_vars={"gia": (( "lat", "lon"), gia),
               "sla_NOGIAcor":(("lat","lon"),t1),
               "SLA_GIAcor":(('lat','lon'),t2)},
    coords={"lat": np.array(d1.lat), 
            "lon": np.array(d1.lon)},
)
sl.add_attrs(da,variables=['lat','lon'])
da.attrs['script']='SLB-GIA_csiro.py'
da.attrs['description']='Inferred GIA correction by taking the difference between the dataset with and without the GIA correction'
da['gia'].attrs['description']='sla_GIAcor - sla_NOGIAcor'
da.attrs['description']='jb_iby_srn_gtn_giy.nc - jb_iby_srn_gtn_gin.nc'
da.attrs['note']='This value should be ADDED to SLA trend'
da.to_netcdf(path+'gia.nc')

#%% regrid
import os
filein='gia.nc'
fileout='gia_180x360.nc'

os.system(
    "cdo -L remapbil,r360x180 "
    + str(path + filein)
    + " "
    + str(path + fileout)
)
    