#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:52:46 2022

find dataset closets to ENS

@author: ccamargo
"""


import xarray as xr
import numpy as np
from scipy import stats
import pickle

import matplotlib.pyplot as plt

def rmse(y_actual,y_predicted):
    '''compute RMSE between two datasets'''

    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
    RMSE = np.sqrt(MSE)
    return RMSE

def find_lowest_rmse(y_actual, y_predicted_list):
    ''' Given an actual dataset and a list of datasets, 
    find the one with smallest RMSE in relation to the actual dataset
    Returns the index of the lowest RMSE dataset, and the dataset
    '''
    y_predicted_list = [y_obs if np.any(np.isfinite((y_obs))) else np.zeros((len(y_actual)))
                                for y_obs in y_predicted_list]
    RMSEs = np.array([rmse(y_actual[np.isfinite(y_obs)],y_obs[np.isfinite(y_obs)]) 
                      for y_obs in y_predicted_list])

    # return more then one
    idx = np.where(RMSEs==np.nanmin(RMSEs))[0]
    y_best = [y_predicted_list[ind] for ind in idx]
    
    # return only the first one withthe lowest RMSE
    idx = np.where(RMSEs==np.nanmin(RMSEs))[0][0]
    y_best = y_predicted_list[idx]
    
    return idx, y_best

#%% get budget components

period = ['1993-2017'] # full years
y0,y1=period[0].split('-')
t0='{}-01-01'.format(int(y0))
t1='{}-12-31'.format(int(y1)-1)
path = '/Volumes/LaCie_NIOZ/data/budget/ts/' 


dic={}

comps = ['alt','dynamic','barystatic','steric']
for comp in comps:
    # print(comp)
    if comp=='alt':
        file = comp+'.nc'
        ds=xr.open_dataset(path+file)
        ds = ds.where((ds.lat>-66) & (ds.lat<66),np.nan)
        names=np.array(ds.names)
        ind= np.where(ds.names=='ENS')[0][0] # index of the ensemble
        idx = np.arange(0,len(ds.names)) # all the index
        idx = np.delete(idx,ind)
        da = ds.SLA
        
    if comp =='dynamic':
        ds=xr.open_dataset('/Volumes/LaCie_NIOZ/data/budget/dynamic_sl.nc')
        ds = ds.where((ds.lat>-66) & (ds.lat<66),np.nan)
        names=np.array(ds.names)
        ind= np.where(ds.names=='ENS')[0][0] # index of the ensemble
        idx = np.arange(0,len(ds.names)) # all the index
        idx = np.delete(idx,ind)
        da = ds.DSL
    if comp=='barystatic':
        file = comp+'.nc'
        ds=xr.open_dataset(path+file)
        ds=ds.SLF_ASL
        ds = ds.where((ds.lat>-66) & (ds.lat<66),np.nan)
        recs = [r for r in np.array(ds.reconstruction)]
        if int(y0)<2002:
            recs.remove('JPL')
            recs.remove('CSR')
        ds = ds.sel(reconstruction=recs)
        names=np.array(ds.reconstruction)
        ind = np.where(ds.reconstruction=='ENS')[0][0] # index of the ensemble
        idx = np.arange(0,len(ds.reconstruction)) # all the index
        idx = np.delete(idx,ind)
        da = ds
    if comp=='steric':
        file = comp+'.nc'
        ds=xr.open_dataset(path+file)
        ds = ds.where((ds.lat>-66) & (ds.lat<66),np.nan)
        names=np.array(ds.names)
        ind= np.where(ds.names=='ENS')[0][0] # index of the ensemble
        idx = np.arange(0,len(ds.names)) # all the index
        idx = np.delete(idx,ind)
        da = ds.steric_full    

    dic[comp] = {'ds':da,
                 'names':names,
                 'ind':ind,
                 'idx':idx}

#%% loop over compoenents:
best_datasets = {}
plot=False
mode = 'gmsl'
for key in dic.keys():
    da = dic[key]['ds']
    names = dic[key]['names']
    ind = dic[key]['ind']
    idx = dic[key]['idx']
    comp = key
    print(comp)
    #% % 3D:
    if mode =='3D':
        y_ens = np.array(da.data[ind,:,:,:]) # get ensemble
        ys= np.array(da.data[idx,:,:,:]) # get all the other ones
        
        best_idx = np.zeros((180,360))
        best_idx.fill(np.nan)
        for i in range(180):
            for j in range(360):
                if np.any(np.isfinite(y_ens[:,i,j])):
                    y_actual = np.array(y_ens[:,i,j] - np.nanmean(y_ens[:,i,j]) )
                    
                    acc_idx = np.isfinite(y_actual)
                    y_predicted_list = [ys[k,acc_idx,i,j] - np.nanmean(ys[k,acc_idx,i,j])  for k in idx]
                    best_idx[i,j], _ = find_lowest_rmse(y_actual[acc_idx], y_predicted_list)
                    # plt.plot(y_actual)
                    # for k in idx:
                    #     plt.plot(y_preds[k])
        if plot:
            plt.figure(dpi=300)
            plt.pcolor(best_idx,vmin=0,vmax=9,cmap='Pastel1')
            plt.colorbar()
            i = stats.mode(best_idx[np.isfinite(best_idx)].flatten())[0][0]
            plt.title('Most chosen: {}'.format(names[int(i)]) )
            plt.show()
        best_datasets[comp] = names[int(i)]
    
    elif mode=='gmsl':
        #% % global mean
        da = da.mean(dim=('lat','lon'))
        y_ens = np.array(da.data[ind,:]) # get ensemble
        ys= np.array(da.data[idx,:]) # get all the other ones
        
        #% %
        y_actual = np.array(y_ens - np.nanmean(y_ens))
        acc_idx = np.isfinite(y_actual)
        y_predicted_list = [ys[k,acc_idx] - np.nanmean(ys[k,acc_idx]) for k in idx]
        best_idx, best_ts = find_lowest_rmse(y_actual[acc_idx], y_predicted_list)
        if plot:
            plt.figure(dpi=300)
            plt.plot(y_actual,label='ENS')
            for k in idx:
                if k ==best_idx:
                    alpha=1
                else: 
                    alpha=0.5
                plt.plot(y_predicted_list[k],label=names[k],alpha=alpha)
            plt.legend()
            plt.title('Most chosen: {}'.format(names[int(best_idx)]) )
            plt.show()
    
        best_datasets[comp] = names[int(best_idx)]
#%% save dic
# pwd ="/Volumes/LaCie_NIOZ/data/budget/"
pwd = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
a_file = open(pwd+"best_rmse.pkl", "wb")
pickle.dump(best_datasets, a_file)
a_file.close()
