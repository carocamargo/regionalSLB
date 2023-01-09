#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:38:14 2022

@author: ccamargo
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))
#%%
vmax=0.1
cmap='Greys_r' # 
cmap = "YlGnBu_r"
#%%
path = "//Volumes/LaCie_NIOZ/budget/regions/som/"
path = '/Users/ccamargo/Desktop/SOM/output/'

for x in [3,4,5,6,7]:
    y=x
    flist = [
        'som_{}x{}_sig_init2_norm_range_train_n10_ngb_function_ep_mask_atlantic_.nc'.format(x,y),
        'som_{}x{}_sig_init2_norm_range_train_n10_ngb_function_ep_mask_indopacific_.nc'.format(x,y)
    ]
    ds = xr.open_dataset(path+flist[0])
    ds.modos[0,:,0].plot()
    # ds.modos[0].plot(col_wrap='neurons')
    modos1 = np.array(ds.modos[0])
    mask1 = np.array(ds.bmu_map)
    n_neurons1 = len(ds.neurons)
    neurons1 = np.array(ds.neurons)
    
    # #%%
    # plt.figure()
    # for neuron in ds.neurons:
    #     neuron = int(neuron)
    #     # neuron = 1
    #     plt.subplot(3,3,neuron)
    #     ds.modos[0,:,neuron-1].plot()
    #     plt.title('neuron = {}'.format(neuron))
    # plt.tight_layout()
    # plt.show()
    # % %
    ds = xr.open_dataset(path+flist[1])
    # ds.modos[0].plot(col_wrap='neurons')
    modos2 = np.array(ds.modos[0])
    mask2 = np.array(ds.bmu_map)
    n_neurons2 = len(ds.neurons)
    neurons2 = np.array(ds.neurons)
    # #%%
    # plt.figure()
    # for neuron in ds.neurons:
    #     neuron = int(neuron)
    #     # neuron = 1
    #     plt.subplot(3,3,neuron)
    #     ds.modos[0,:,neuron-1].plot()
    #     plt.title('neuron = {}'.format(neuron))
    # plt.tight_layout()
    # plt.show()
    #% %
    mask = np.array(mask2+n_neurons1)
    mask[np.isnan(mask)] = 0
    mask1[np.isnan(mask1)] = 0
    mask = np.array(mask+mask1)
    
    modos = np.zeros((len(modos1),n_neurons1+n_neurons2))
    modos[:,0:n_neurons1] = modos1
    modos[:,n_neurons1:n_neurons2+n_neurons1] = modos2
    
    
    #% 
    n_time,n_neurons = modos.shape
    table = np.zeros((n_neurons,n_neurons))
    absolute = False
    for i in range(n_neurons):
        #% %
        # i=0
        ref = np.array(modos[:,i])
        # plt.figure(figsize=(20,10))
        for j in range(n_neurons):
            target = np.array(modos[:,j])
            dif = np.array(ref-target)
            # plt.subplot(3,6,j+1)
            # plt.plot(ref,label='n = {}'.format(i))        
            # plt.plot(target,label='n = {}'.format(j))
            
            # plt.plot(dif,label='t{}-t{}'.format(i,j))
            # # plt.legend()
            # plt.title('n{} - n{}'.format(i,j))
            if absolute == True:
                table[i,j] = np.std(np.abs(dif))
            else:
                table[i,j] = mad(dif)
        # plt.show()
    
    #% %
    # corr = np.corrcoef(np.random.randn(10, 200))
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(np.abs(table), mask=mask, 
                         # vmax=.3, 
                         vmin=0,vmax=vmax,
                          square=True, 
                          # cmap="YlGnBu_r"
                          cmap = cmap
                         )
        plt.title('SOM 2x{}x{}'.format(x,y))
        plt.show()
        
#% %
for x in [3,4,5,9]:
    y=x
    file = 'som_{}x{}_sig_init2_norm_range_train_n10_ngb_function_ep_mask_ocean_.nc'.format(x,y)
    ds = xr.open_dataset(path+file)
    modos = np.array(ds.modos[0])
    mask = np.array(ds.bmu_map)
    n_neurons = len(ds.neurons)
    neurons = np.array(ds.neurons)
    
    n_time,n_neurons = modos.shape
    table = np.zeros((n_neurons,n_neurons))
    absolute = False
    for i in range(n_neurons):
        #% %
        # i=0
        ref = np.array(modos[:,i])
        # plt.figure(figsize=(20,10))
        for j in range(n_neurons):
            target = np.array(modos[:,j])
            dif = np.array(ref-target)
            # plt.subplot(3,6,j+1)
            # plt.plot(ref,label='n = {}'.format(i))        
            # plt.plot(target,label='n = {}'.format(j))
            
            # plt.plot(dif,label='t{}-t{}'.format(i,j))
            # # plt.legend()
            # plt.title('n{} - n{}'.format(i,j))
            if absolute == True:
                table[i,j] = np.std(np.abs(dif))
            else:
                table[i,j] = mad(dif)
        # plt.show()
    # corr = np.corrcoef(np.random.randn(10, 200))
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(np.abs(table), mask=mask, 
                         vmin=0,vmax=vmax, 
                         square=True, 
                          cmap=cmap
                         # cmap = 'Greys_r'
                         )
        plt.title('SOM 1x{}x{}'.format(x,y))
        plt.show()
        