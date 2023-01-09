#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:27:48 2022

@author: ccamargo
"""
from uncertainties import ufloat
from uncertainties import unumpy
import numpy as np
import pandas as pd
#%% 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# import cmocean as cm
import matplotlib.pyplot as plt

#%%
def from_360_to_180(lon_in):
    # given a longitude array that goes from 0 to 360, 
    # returns an array that goes from -180 to 180
    lon_out=np.copy(lon_in)
    for i,ilon in enumerate(lon_in):
        if ilon > 180: 
            lon_out[i]=ilon-360
    return lon_out
#%%
import xarray as xr
path = '/Volumes/LaCie_NIOZ/data/budget/ts/' 
ds = xr.open_dataset(path+'alt.nc')
# ds = ds.sel(time=slice(t0,t1))
lat=np.array(ds.lat)
lon=np.array(ds.lon)
lon=from_360_to_180(lon)


#%%
path = '/Users/ccamargo/Desktop/manuscript_SLB/data/'
path = '/Volumes/LaCie_NIOZ/data/budget/'
file = 'budget_v2.pkl'
dic = pd.read_pickle(path+file)
dic['dmap'].keys()
#%% 
for key in ['dmap','som']:
    # key = 'dmap'
    #% %
    #% 
    da=xr.Dataset(data_vars={
                            'mask':(('lat','lon'),np.array(dic[key]['mask'])),
                            # 'alt_trend':(('lat','lon'),np.array(dic[key]['alt']['trend'])),
                            # 'alt_unc':(('lat','lon'),np.array(dic[key]['alt']['unc'])),
                            # 'sum_trend':(('lat','lon'),np.array(dic[key]['alt']['trend'])),
                            # 'alt_unc':(('lat','lon'),np.array(dic[key]['alt']['unc'])),
                            
                            },
                     coords={'lat':ds.lat,
                             'lon':ds.lon})
    for var in ['alt','steric','barystatic','dynamic','sum','res']:
        da[var+'_trend'] = (('lat','lon'),np.array(dic[key][var]['trend']))
        da[var+'_unc'] = (('lat','lon'),np.array(dic[key][var]['unc']))
        
    da.to_netcdf(path+key+'.nc')

#%% DMAPS
df = dic['dmap']['df']
df.to_csv(path+'dmaps.csv')
n_sig=2

var = ['alt','steric','barystatic','dynamic','sum','res']

mask = np.array(dic['dmap']['mask'])
#mask.tofile(path+'dmaps_mask.csv',sep=',')
mask[np.isnan(mask)]=0
np.savetxt(path+'dmaps_mask.csv',mask, fmt = '%d', delimiter=",") 

df2 = pd.DataFrame({'cluster_ID':df['cluster_n']})
#% %
lng = []
lt = []
for icluster in np.array(df2['cluster_ID']):
    # icluster = 1 
    mask2 = np.array(mask)
    mask2[mask!=icluster]=np.nan
    # lat = np.arange(-90,90,1)
    # lon = np.arange(0,360,1)
    llon,llat = np.meshgrid(lon,lat)
    llatm = np.array(llat).astype(float)
    llatm[mask!=icluster]=np.nan
    llonm = np.array(llon).astype(float)
    llonm[mask!=icluster]=np.nan
    central_cord = (np.nanmean(llonm),np.nanmean(llatm))
    lng.append(np.nanmean(llonm))
    lt.append(np.nanmean(llatm))
df2['lat']=lt
df2['lon']=lng
for v in var:
    df2[v] = unumpy.uarray(
        np.array(df['{}_tr'.format(v)].round(n_sig)), 
        np.array(df['{}_unc'.format(v)].round(n_sig)), 
        )
df2.to_csv(path+'dmaps_unc.csv')
#%%
plt.pcolor(mask)
da=xr.Dataset(data_vars={'mask':(('lat','lon'),mask)},
                 coords={'lat':lat,
                         'lon':lon})
da.to_netcdf(path+'dmaps_mask.nc')
#%% SOM
df = dic['som']['df']
df.to_csv(path+'som.csv')

var = ['alt','steric','barystatic','dynamic','sum','res']
mask = np.array(dic['som']['mask'])
mask[np.isnan(mask)]=0
np.savetxt(path+'som_mask.csv',mask, fmt = '%d', delimiter=",")  


df2 = pd.DataFrame({'cluster_ID':df['cluster_n']})
#% %
lng = []
lt = []
for icluster in np.array(df2['cluster_ID']):
    # icluster = 1 
    mask2 = np.array(mask)
    mask2[mask!=icluster]=np.nan
    # lat = np.arange(-90,90,1)
    # lon = np.arange(0,360,1)
    llon,llat = np.meshgrid(lon,lat)
    llatm = np.array(llat).astype(float)
    llatm[mask!=icluster]=np.nan
    llonm = np.array(llon).astype(float)
    llonm[mask!=icluster]=np.nan
    central_cord = (np.nanmean(llonm),np.nanmean(llatm))
    lng.append(np.nanmean(llonm))
    lt.append(np.nanmean(llatm))
df2['lat']=lt
df2['lon']=lng

for v in var:
    df2[v] = unumpy.uarray(
        np.array(df['{}_tr'.format(v)].round(n_sig)), 
        np.array(df['{}_unc'.format(v)].round(n_sig)), 
        )
df2.to_csv(path+'som_unc.csv')


#%%
plt.pcolor(mask2)
plt.scatter(central_cord[0],central_cord[1])


fig = plt.figure()

proj=ccrs.PlateCarree()
ax = plt.subplot(111, projection=proj
                     #Mercator()
                     )
ax.set_global()
ax.pcolormesh(lon,\
          lat,\
        mask,
        vmin=np.nanmin(mask),vmax=np.nanmax(mask) ,
        transform=ccrs.PlateCarree(),
        #cmap='Spectral_r'
        cmap='jet'
                          )
ax.scatter(lng,lt,color='white',
           transform=ccrs.PlateCarree(),)

#%%


