#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:43:27 2022

@author: ccamargo
"""
path_to_data= '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
fname = 'budget_components'
#%% open budget dic 
def open_dic():
    # open dic of budget components from budget.py script
    import pandas as pd
    path = '/Volumes/LaCie_NIOZ/data/budget/'
    path = path_to_data
    # fname = 'budget_v2.pkl'
    # fname = fname
    dic = pd.read_pickle(path+fname+'.pkl')
    return dic

def get_dimensions():
    import xarray as xr
    import numpy as np
    # dimensions
    path = '/Volumes/LaCie_NIOZ/data/budget/ts/' 
    ds = xr.open_dataset(path+'alt.nc')
    period = ['1993-2017'] # full years
    y0,y1=period[0].split('-')
    t0='{}-01-01'.format(int(y0))
    t1='{}-12-31'.format(int(y1)-1)
    ds = ds.sel(time=slice(t0,t1))
    da = ds['sla_ens'][0,:,:]
    
    da = da.where((ds.lat>-66) & (ds.lat<66),np.nan)
    # da.plot()
    landmask = np.array(da.data)
    landmask[np.isfinite(landmask)]=1
    # plt.pcolor(landmask)
    
    da = xr.Dataset(data_vars = {'mask_1deg':(('lat','lon'),landmask)},
                    coords = {'lat':ds.lat,
                              'lon':ds.lon,
                              'time':ds.time}
                    )
    return da


def make_nc_components():
    # import xarray as xr
    dic = open_dic()
    da = get_dimensions()
    keys = [key for key in dic.keys() if key not in ['steric_up','time']]
    labels = {'steric':'steric',
              'alt':'altimetry',
              'barystatic':'GRD',
              'dynamic':'dynamic',
              'res':'residual',
              'sum':'sum'}
    
    for key in keys:
        da[labels[key]+'_trend'] = (('lat','lon'),dic[key]['trend'])
        da[labels[key]+'_unc'] = (('lat','lon'),dic[key]['unc'])
        da[labels[key]+'_ts'] = (('time','lat','lon'),dic[key]['ts'])
        
    path = path_to_data
    # fname = fname   
    da.to_netcdf(path+fname+'.nc')
    return

#%% call main function
make_nc_components()
