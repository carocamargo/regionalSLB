#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:00:21 2022

Part of budget.py script (budget_analysis folder)
@author: ccamargo
"""

import numpy as np
import xarray as xr
import os
# import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")

from utils_SLB import sum_linear, sum_square, get_dectime
#%% get budget components

dic = {}

period = ['1993-2017'] # full years
y0,y1=period[0].split('-')
t0='{}-01-01'.format(int(y0))
t1='{}-12-31'.format(int(y1)-1)
path = '/Volumes/LaCie_NIOZ/data/budget/trends/' 
path2 = '/Volumes/LaCie_NIOZ/data/budget/ts/' 
flist = [file for file in os.listdir(path) if not file.startswith('.')]
flist=['steric.nc', 'alt.nc', 'barystatic.nc', 'dynamic.nc']
gia_cor = True

for file in flist:
    comp = file.split('.')[0]
    # print(comp)
    
    ds=xr.open_dataset(path+file)
    da = xr.open_dataset(path2+file)
    # print(da)
    ds = ds.sel(periods=period)
    da = da.sel(time=slice(t0,t1))
    
    if comp=='alt':
        if gia_cor:
            ds=xr.open_dataset(path+'alt_GIAcor.nc')
            ds=ds.sel(names=['ENS'])
            trend = np.array(ds.trends[0,:,:])
            unc = np.array(ds.uncs[0,:,:])
        else:
            ds=ds.sel(names=['ENS'])
            idx = np.where(ds.ICs =='bic_tp')[0][0]
            trend = np.array(ds.best_trend[0,0,idx,:,:])
            unc = np.array(ds.best_unc[0,0,idx,:,:])
        
        ts = np.array(da.sel(names='ENS')['SLA']) # mm
        tdec = get_dectime(da.time)
        dic['time']=tdec
        #% % land mask
        da = da['sla_ens'][0,:,:]
        da = da.where((ds.lat>-66) & (ds.lat<66),np.nan)
        # da.plot()
        landmask = np.array(da.data)
        landmask[np.isfinite(landmask)]=1

        
    elif comp =='dynamic':
        idx = np.where(ds.ICs =='bic_tp')[0][0]
        trend = np.array(ds.best_trend[0,idx,:,:])
        unc = np.array(ds.best_unc[0,idx,:,:])
        # ts = np.array(da['ens_dyn_v1'] * 1000) # mm
        da=da.sel(names=['ENS'])
        ts = np.array(da['DSL'][0,:,:,:] * 1000) # mm

    elif comp=='steric':
        ds=ds.sel(names=['ENS'])
        trend = np.array(ds.trend_full[0,0,:,:])
        unc = np.array(ds.unc[0,0,:,:])
        ts = np.array(da.sel(names='ENS')['steric_full'] * 1000) # mm

        trend_up = np.array(ds.trend_up[0,0,:,:])
        unc_up = np.array(ds.unc[0,0,:,:])
        ts_up = np.array(da.sel(names='ENS')['steric_up']*1000)
        dic['steric_up'] = {'trend':trend_up,
                            'unc':unc_up,
                            'ts':ts_up}
        
    elif comp=='barystatic':
        ds=ds.sel(names=['IMB_WGP'])
        trend = np.array(ds.SLA[0,0,:,:])
        unc = np.array(ds.SLA_UNC[0,0,:,:])
        da=da['SLF_ASL'].sel(reconstruction='ENS')
        ts = np.array(da.data)
    
    dic[comp] = {'trend':trend,
                'unc':unc,
                'ts':ts
                }

lat=np.array(ds.lat)
lon=np.array(ds.lon)

#%% sum of comps and residual
datasets = ['steric','barystatic','dynamic']
das_unc = []
das_trend = []
das_ts = []
for key in datasets:
    das_unc.append(dic[key]['unc'])
    das_trend.append(dic[key]['trend'])
    das_ts.append(dic[key]['ts'])
    
sum_comps_trend = sum_linear(das_trend)
sum_comps_unc = sum_square(das_unc)
sum_comps_ts = sum_linear(das_ts)

res_trend = sum_linear([dic['alt']['trend'], sum_comps_trend], how='subtract')
# res_unc = sum_square([dic['alt']['unc'], sum_comps_unc], how='subtract')
res_unc = sum_square([dic['alt']['unc'], sum_comps_unc])
res_ts = sum_linear([dic['alt']['ts'], sum_comps_ts], how='subtract')

dic['res']={'trend':res_trend,'unc':res_unc,'ts':res_ts}
dic['sum']={'trend':sum_comps_trend,'unc':sum_comps_unc,'ts':sum_comps_ts}
#%%
# load pickle module
import pickle
path = '/Volumes/LaCie_NIOZ/data/budget/'
path= '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
# create a binary pickle file 
f = open(path+"budget_components.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(dic,f)

# close file
f.close()
