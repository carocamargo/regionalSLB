#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:42:56 2022

@author: ccamargo
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
path = "//Volumes/LaCie_NIOZ/budget/regions/som/"
flist = [
    "som_3x3_alt_1993_2019_n10_sig2_ep_atlantic.nc",
    "som_3x3_alt_1993_2019_n10_sig2_ep_indopacific.nc",
]
ds = xr.open_dataset(path+flist[0])
ds.modos[0,:,0].plot()
ds.modos[0].plot(col_wrap='neurons')
modos1 = np.array(ds.modos[0])
mask1 = np.array(ds.bmu_map)
n_neurons1 = len(ds.neurons)
neurons1 = np.array(ds.neurons)

#%%
plt.figure()
for neuron in ds.neurons:
    neuron = int(neuron)
    # neuron = 1
    plt.subplot(3,3,neuron)
    ds.modos[0,:,neuron-1].plot()
    plt.title('neuron = {}'.format(neuron))
plt.tight_layout()
plt.show()
#%%
ds = xr.open_dataset(path+flist[1])
ds.modos[0].plot(col_wrap='neurons')
modos2 = np.array(ds.modos[0])
mask2 = np.array(ds.bmu_map)
n_neurons2 = len(ds.neurons)
neurons2 = np.array(ds.neurons)
#%%
plt.figure()
for neuron in ds.neurons:
    neuron = int(neuron)
    # neuron = 1
    plt.subplot(3,3,neuron)
    ds.modos[0,:,neuron-1].plot()
    plt.title('neuron = {}'.format(neuron))
plt.tight_layout()
plt.show()
#%%
mask = np.array(mask2+n_neurons1)
mask[np.isnan(mask)] = 0
mask1[np.isnan(mask1)] = 0
mask = np.array(mask+mask1)

modos = np.zeros((len(modos1),n_neurons1+n_neurons2))
modos[:,0:n_neurons1] = modos1
modos[:,n_neurons1:n_neurons2+n_neurons1] = modos2


#%%
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
            table[i,j] = np.std(dif)
    # plt.show()
#%%

#%%
# corr = np.corrcoef(np.random.randn(10, 200))
mask = np.zeros_like(table)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(np.abs(table), mask=mask, 
                     vmax=.3, square=True, 
                      cmap="YlGnBu_r"
                     # cmap = 'Greys_r'
                     )
    plt.title('SOM 2x3x3')
    plt.show()
    
#%%
file = 'som_4x4_alt_1993_2019_n10_sig2_gaussian_world.nc'
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
            table[i,j] = np.std(dif)
    # plt.show()
# corr = np.corrcoef(np.random.randn(10, 200))
mask = np.zeros_like(table)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(np.abs(table), mask=mask, 
                     vmax=.3, square=True, 
                      cmap="YlGnBu_r"
                     # cmap = 'Greys_r'
                     )
    plt.title('SOM 1x4x4')
    plt.show()