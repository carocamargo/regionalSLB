#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:33:08 2022

Use noise model selection from before to run the trends in 
ENS and  CSIRO datasets 
@author: ccamargo
"""


# import pandas as pd
import numpy as np
import xarray as xr
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import utils_hec as hec
import os
# import cmocean as cm
# import matplotlib.pyplot as plt

#%%
def get_NM(name='ENS',
           period = '1993-2017',
           path= "/Volumes/LaCie_NIOZ/data/budget/trends/",
           file = 'alt.nc',
           IC_idx = -1,
           IC='bic_tp'
           ):
    da = xr.open_dataset(path+file)
    ds = da.sel(names=name)
    ds = ds.sel(periods = period)
    ds = ds.sel(ICs = IC)
    var = 'NM_score'
    return ds[var].data, np.array(ds.nm)
def open_ts(name = 'ENS',
            period ='1993-2017',
            path= "/Volumes/LaCie_NIOZ/data/budget/",
            file = 'alt.nc',
            ):
    
    ds = xr.open_dataset(path+file)
    ds = ds.sel(names=name)
    t0,t1 = period.split('-')
    ds=ds.sel(time=slice('{}-01-01'.format(t0),'{}-12-31'.format(t1)))
    ds = ds.where((ds.lat>-66) & (ds.lat<66),np.nan)
    return ds['SLA']
def run_hector(height,time,nm,
               name='alt',
               path_hector = '/Volumes/LaCie_NIOZ/dump/0/',
               ):
    # 1. transform time:
    # time = tdec_to_mjd(tdec)
    
    # set path to hector folder:
    os.chdir(path_hector)
    
    # remove nans:
    # acc_idx = np.isfinite(height)
    # height=height[acc_idx]
    # time=time[acc_idx]
    # interpolate nans:
    if np.any(np.isnan(height)):
        height = np.array(
            pd.Series(height).interpolate().tolist())
    
    
    # estimate sampling period:
    # sp=np.round(np.mean(np.diff(time)))
    sp=30
    
    # create .mom file:
    hec.ts_to_mom(height,time,sp=sp,path=str(path_hector+'raw_files/'),
                  name=name,ext='.mom')
    # # Create a .ctl file:
    hec.create_estimatetrend_ctl_file(name,nm,sp=sp,
                                    GGM_1mphi = 6.9e-07,
                                    seas=True,halfseas=True,
                                    LikelihoodMethod='FullCov'
                                    )
    # Run estimatetrend (hector)
    _=os.system('estimatetrend > estimatetrend.out')

    # get results:
    out=hec.get_results()

    #     trend   unc
    return float(out[0]),float(out[1])

def get_trend(name = 'ENS',
              save=True,
              path_save = '/Volumes/LaCie_NIOZ/data/altimetry/trends/noGIA/'):
    nm_sel, nms = get_NM(name=name) 
    da = open_ts(name=name)
    time = da.time.data
    tdec,_ = sl.get_dec_time(time)
    lat = da.lat.data
    lon = da.lon.data
    height = np.array(da.data)
    trend = np.zeros((len(lat),len(lon)))
    unc = np.zeros((len(lat),len(lon)))
    trend[:,:]=np.nan
    unc[:,:]=np.nan
    mask = np.array(height[0])
    mask[np.isfinite(mask)]=1
    nm_sel = np.array(mask*nm_sel)
    jmax = len(mask[np.isfinite(mask)])
    j=0
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            if np.isfinite(nm_sel[ilat,ilon]):
                nm_idx = int(nm_sel[ilat,ilon])
                nm = nms[nm_idx]
                trend[ilat,ilon],unc[ilat,ilon] = run_hector(height[:,ilat,ilon],tdec,nm)
                print('{} out of {}'.format(j,jmax))
                j=j+1
    if save:
        print('Saving for {}'.format(name))
        ds = xr.Dataset(
             data_vars={'best_trend':(('lat','lon'),trend),
                        'best_unc':(('lat','lon'),unc),
                        },
             coords={'names':name,
                     'lat':da.lat,
                     'lon':da.lon,
                     'periods':da.periods,
                     'ICs':da.ICs,
                     }
             )
        ds['metadata'] = da.metadata
        ds['correction']='{} trends corrected to not include GIA uncertainties'.format(name)
        ds['script']='SLB/components/alt_trend_no_GIA.py'
        ds.to_netcdf(path_save+name+'.nc')
    return trend, unc


def replace_trends(names=['ENS'],
           period = '1993-2017',
           path= "/Volumes/LaCie_NIOZ/data/budget/trends/",
           file = 'alt.nc',
           file_save = 'alt_noGIA',
           IC_idx = -1,
           IC='bic_tp'
           ):
    da = xr.open_dataset(path+file)
    data_t = np.array(da.best_trend)
    data_u = np.array(da.best_unc)
    
    for name in names:
        print('Starting {}'.format(name))
        idx = np.where(da.names==name)[0][0]
        trend,unc = get_trend(name = name, save=False) # re-run trend with select NM
        data_t[idx,0,IC_idx,:,:] = trend
        data_u[idx,0,IC_idx,:,:] = unc
        print('finished {}'.format(name))

    ds = xr.Dataset(
         data_vars={'NM_score':(('names','periods','ICs','lat','lon'),da.NM_score),
                    'best_trend':(('names','periods','ICs','lat','lon'),data_t),
                    'best_unc':(('names','periods','ICs','lat','lon'),data_u),
                    },
         coords={'names':da.names,
                 'lat':da.lat,
                 'lon':da.lon,
                 'periods':da.periods,
                 'ICs':da.ICs,
                 }
         )
    ds['metadata'] = da.metadata
    ds['correction']='{} trends corrected to not include GIA uncertainties'.format(names)
    ds['idx_corrected'] = 'only {} corrected'.format(IC)
    ds['script']='SLB/components/alt_trend_no_GIA.py'
    ds.to_netcdf(path+file_save+'.nc')
    return 

def get_quicktrend(name = 'ENS',
              save=True,
              nm='WN',
              path_save = '/Volumes/LaCie_NIOZ/data/altimetry/trends/noGIA/'):
    # nm_sel, nms = get_NM(name=name) 
    
    da = open_ts(name=name)
    time = da.time.data
    tdec,_ = sl.get_dec_time(time)
    lat = da.lat.data
    lon = da.lon.data
    height = np.array(da.data)
    trend = np.zeros((len(lat),len(lon)))
    unc = np.zeros((len(lat),len(lon)))
    trend[:,:]=np.nan
    unc[:,:]=np.nan
    mask = np.array(height[0])
    mask[np.isfinite(mask)]=1
    # nm_sel = np.array(mask*nm_sel)
    jmax = len(mask[np.isfinite(mask)])
    j=0
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            if np.isfinite(mask[ilat,ilon]):
                # nm_idx = int(nm_sel[ilat,ilon])
                # nm = nms[nm_idx]
                trend[ilat,ilon],unc[ilat,ilon] = run_hector(height[:,ilat,ilon],tdec,nm)
                print('{} out of {}'.format(j,jmax))
                j=j+1
    if save:
        print('Saving for {}'.format(name))
        ds = xr.Dataset(
             data_vars={'best_trend':(('lat','lon'),trend),
                        'best_unc':(('lat','lon'),unc),
                        },
             coords={'names':name,
                     'lat':da.lat,
                     'lon':da.lon,
                     'periods':da.periods,
                     'ICs':da.ICs,
                     }
             )
        ds['metadata'] = da.metadata
        ds['correction']='{} trends corrected to not include GIA uncertainties'.format(name)
        ds['script']='SLB/components/alt_trend_no_GIA.py'
        ds.to_netcdf(path_save+name+'.nc')
    return trend, unc

def quick_replace_trends(names=['ENS'],
           period = '1993-2017',
           path= "/Volumes/LaCie_NIOZ/data/budget/trends/",
           file = 'alt.nc',
           file_save = 'alt_noGIA_WN',
           IC_idx = -1,
           IC='bic_tp'
           ):
    da = xr.open_dataset(path+file)
    data_t = np.array(da.best_trend)
    data_u = np.array(da.best_unc)
    
    for name in names:
        print('Starting {}'.format(name))
        idx = np.where(da.names==name)[0][0]
        trend,unc = get_quicktrend(name = name,save=False) # re-run trend with select NM
        data_t[idx,0,IC_idx,:,:] = trend
        data_u[idx,0,IC_idx,:,:] = unc
        print('finished {}'.format(name))

    ds = xr.Dataset(
         data_vars={'NM_score':(('names','periods','ICs','lat','lon'),da.NM_score),
                    'best_trend':(('names','periods','ICs','lat','lon'),data_t),
                    'best_unc':(('names','periods','ICs','lat','lon'),data_u),
                    },
         coords={'names':da.names,
                 'lat':da.lat,
                 'lon':da.lon,
                 'periods':da.periods,
                 'ICs':da.ICs,
                 }
         )
    ds['metadata'] = da.metadata
    ds['correction']='{} trends corrected to not include GIA uncertainties'.format(names)
    ds['idx_corrected'] = 'only {} corrected'.format(IC)
    ds['script']='SLB/components/alt_trend_no_GIA.py'
    ds.to_netcdf(path+file_save+'.nc')
    return 
#%% run
names=['ENS','csiro']
replace_trends(names=names)
# quick_replace_trends(names=names)
# 

