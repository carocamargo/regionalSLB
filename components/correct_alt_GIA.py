#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:31:23 2022

Correct Altimetry trends for GIA

1c) Correct altimetry trend with GIA:
	components/correct_alt_GIA.py
		input data: trends at /Volumes/LaCie_NIOZ/data/budget/trends/'
		output:
            
@author: ccamargo
"""
import xarray as xr
import numpy  as np
#%%
def open_GIA_peltier(mode = 'RSL',
                     model=6):
    
    path = '/Volumes/LaCie_NIOZ/data/GIA/peltier/'
    
    if mode =='RSL':
        var = 'Dsea_250'
        if model ==6:
            file = 'dsea.1grid_O512.nc'
        elif model==5:
            file = 'dsea250.1grid.ICE5Gv1.3_VM2_L90_2012'
        else:
            raise( NameError('Model not known for this variable') )
        
    elif mode =='RAD':
        var = 'Drad_250'
        if model ==6:
            file = 'drad.1grid_O512.nc'
        elif model==5:
            file = 'drad250.1grid.ICE5Gv1.3_VM2_L90_2012'
        else:
            raise( NameError('Model not known for this variable') )

    elif mode =='GEO':
        var1='Dsea_250'
        var2='Drad_250'
        if model ==6:
            file1 = 'dsea.1grid_O512.nc'
            file2 = 'drad.1grid_O512.nc'            

        elif model==5:
            file = 'dGeoid250.1grid.ICE5Gv1.3_VM2_L90_2012'
        else:
            raise( NameError('Model not known for this variable') )
    
    elif mode=='VLM':
        var1='Dsea'
        var2='Drad'
        if model ==6:
            file1 = 'dsea.1grid_O512.nc'
            file2 = 'drad.1grid_O512.nc'
        elif model==5:
            file1 = 'dsea250.1grid.ICE5Gv1.3_VM2_L90_2012'
            file2 = 'drad250.1grid.ICE5Gv1.3_VM2_L90_2012'
        else:     
            raise( NameError('Model not known for this variable') )

    else:
        raise( NameError('Model and variable not known ') )
    
    # open dataset:
    if mode=='VLM':
        ds1 = xr.open_dataset(path+file1)
        data1 = np.array(ds1[var1])
        ds2 = xr.open_dataset(path+file2)
        data2 = np.array(ds2[var2])
        data = np.array(data1-data2)
        lat = np.array(ds1['Lat'])
        lon = np.array(ds1['Lon'])
        
    if mode =='GEO' and model==6:
        ds1 = xr.open_dataset(path+file1)
        data1 = np.array(ds1[var1])
        ds2 = xr.open_dataset(path+file2)
        data2 = np.array(ds2[var2])
        data = np.array(data1+data2)
        lat = np.array(ds1['Lat'])
        lon = np.array(ds1['Lon'])        
        

    else:
        ds = xr.open_dataset(path+file)
        data = np.array(ds[var])
        lat = np.array(ds['Lat'])
        lon = np.array(ds['Lon'])
     # make dataset
    da = xr.Dataset(
        data_vars={"gia": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon},
    )
    
    da = da.sortby('lat',ascending=True)

    
    return da
    

def open_GIA_csiro():
    file = 'gia_180x360.nc'
    path = '/Volumes/LaCie_NIOZ/data/altimetry/world/csiro/'
    da = xr.open_dataset(path+file)
    
    return da

def open_alt(file = 'alt_noGIA.nc'):
    path = "/Volumes/LaCie_NIOZ/data/budget/trends/"
    
    ds = xr.open_dataset(path+file)
    return ds

def cor_alt(gia_cor='peltier',mode='RAD',model=6,how='sum'):
    ds = open_alt()
    if  gia_cor=='csiro':
        da = open_GIA_csiro()
    elif gia_cor=='peltier':
        da = open_GIA_peltier(mode = mode,
                             model=model,
                             )
    else:
        raise(NameError('GIA Correction Unknown'))
        
    gia = np.array(da.gia)
    idx = np.where(ds.ICs =='bic_tp')[0][0]
    # get trends for all datasets, for period of 1993-2017
    trends = np.array(ds['best_trend'][:,0,idx,:,:])
    uncs = np.array(ds['best_unc'][:,0,idx,:,:])
    
    trends_cor =np.zeros((trends.shape))
    for i in range(len(ds.names)):
        if how=='sum':
            trends_cor[i] = np.array(trends[i] + gia)
        else:
            trends_cor[i] = np.array(trends[i] - gia)
    
    da = xr.Dataset(data_vars={
        "trends": (("names","lat", "lon"), trends_cor),
        "uncs": (("names","lat", "lon"), uncs)
        },
        coords={"lat": ds.lat, "lon": ds.lon,
                "names":ds.names},
    )
    da.to_netcdf( "/Volumes/LaCie_NIOZ/data/budget/trends/alt_GIAcor.nc")
#%%
cor_alt(
        #gia_cor='csiro'
        gia_cor='peltier',
        # mode='RAD',
        mode='GEO', # RAD+SEA
        how ='subtract',
        model=6
        )
    