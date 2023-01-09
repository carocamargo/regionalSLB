#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:37:45 2022

@author: ccamargo
"""


import numpy as np
import xarray as xr
# import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils_SLB import cluster_mean
#%%
def get_area(lat,lon,grid):
    
    ''' 
    given a certain grid, compute the area of each grid cell and the total surface area
    
    Inputs:
        lat: latitude, in decimal degrees
        lon: longitude, in decimal degrees,
        grid: grid with dimensions len(lat)xlen(lon)
        
    Outputs:
        surface_area (float): total surface area, in km**2:
        grid_area (matrix): area at each grid cell, in km**2
    '''

    R=6371 # Earth's radius in km
    
    #Check the grid resolution:
    deltalat=180/len(lat);
    deltalon=360/len(lon) 
    
    #Transform from degrees to km:

    deltay=(2*np.pi*R*deltalat)/360 #lat to km
    deltax=(2*np.pi*R*np.cos(np.radians(lat))*deltalon)/360 #lon to km
    
    area=np.array([deltax*deltay]*len(lon)).transpose()
    ocean_surf=np.nansum(area*grid)
    
    return ocean_surf, area
#%% get budget components

path = '/Volumes/LaCie_NIOZ/data/budget/'
dic = pd.read_pickle(path+'budget_v2.pkl')
# ds=xr.open_dataset(path+'trends/alt.nc')
lat = np.array(dic['dims']['lat']['lat'])
lon = np.array(dic['dims']['lon']['lon'])
llon,llat = np.meshgrid(lon,lat)
lonv = llon.flatten()
latv = llon.flatten()
dimlat = len(lat)
dimlon = len(lon)
#%% SOM & dmaps
for j,key in enumerate(['som','dmap']):
    # key = 'som'
    res = dic[key]['res']['trend'] # cluster residual trend
    unc = dic[key]['res']['unc'] # cluster uncertainty
    mask = dic[key]['mask'] # clusters mask
    df = dic[key]['df']
    # get area grid
    surf, area = get_area(lat,lon,np.ones((dimlat,dimlon)) )
    
    cluster_area = np.zeros((len(df)))
    central_lat = np.full_like(cluster_area,0)
    central_lon = np.full_like(cluster_area,0)
    n_grid = np.full_like(cluster_area,0)
    
    cluster_sig = np.full_like(cluster_area,0)
    for i,icluster in enumerate(np.unique(mask[np.isfinite(mask)])):
        
        # cluster mask, 1 for the cluster, nan everywhere else
        mask_tmp = np.array(mask)
        mask_tmp [mask_tmp!=icluster] = np.nan
        mask_tmp [np.isfinite(mask_tmp)] = 1
        
        # area of the cluster
        cluster_area[i] = np.nansum(area*mask_tmp)
        
        # number of grid in cluster
        n_grid[i] = len(mask_tmp[np.isfinite(mask_tmp)])
        
        # get central lat and lon of the cluster
        # central_lat[i], central_lon[i] =  find_central_coord(mask_tmp, llat, llon)
        
        # see if cluster is closed or not
        # 0 for closed, 1 for open
        if np.abs(df['res_tr'][i])< df['res_unc'][i]:
            cluster_sig[i] = 0
        else:
            cluster_sig[i] = 1
    #% %
    df['size']=cluster_area
    df['grid_n'] = n_grid
    df['sig']=cluster_sig
    dic[key]['df'] = df
    
    #%%
x_name = 'size'
xmin = min(dic['som']['df'][x_name].min(),dic['dmap']['df'][x_name].min(),)
xmax = max(dic['som']['df'][x_name].max(),dic['dmap']['df'][x_name].max(),)

y_name = 'res_tr'
ymin = min(dic['som']['df'][y_name].min(),dic['dmap']['df'][y_name].min(),)
ymax = max(dic['som']['df'][y_name].max(),dic['dmap']['df'][y_name].max(),)

x_offset = 10**6
y_offset = 0.5
plt.figure()
for j,key in enumerate(['som','dmap']):
    df = dic[key]['df']
    plt.subplot(1,2,int(j+1))
    plt.scatter(df[x_name],df[y_name],
                alpha=0.5,
                marker='o',
                # facecolors='none', edgecolors='b'
               )
    plt.title(key)
    plt.xlim(xmin-x_offset,xmax+x_offset)
    plt.ylim(ymin-y_offset,ymax+y_offset)
plt.show()
#%%

plt.figure()
colors=['red','blue']
labels={'dmap': '$\delta$-MAPS',
        'som':'SOM'}
for j,key in enumerate(['dmap','som',]):
    df = dic[key]['df']
    # plt.subplot(1,2,int(j+1))
    plt.scatter(df[x_name]/10**6,df[y_name],
                alpha=0.5,
                marker='o',
                color=colors[j],
                label=labels[key]
                # facecolors='none', edgecolors='b'
               )
    # plt.title(key)
    
    # plt.xlim(xmin-x_offset,xmax+x_offset)
    plt.ylim(ymin-y_offset,ymax+y_offset)
plt.legend()
plt.xlabel('Area (million km2)')
plt.ylabel('Residual trend \n (mm/yr)')
plt.show()
#%% 
plt.figure()
for j,key in enumerate(['som','dmap']):
    df = dic[key]['df']
    plt.subplot(1,2,int(j+1))
    plt.scatter(np.log(df[x_name]),df[y_name],
                alpha=0.5,
                marker='o',
                # facecolors='none', edgecolors='b'
               )
    plt.title(key)
    # plt.xlim(12,18)
    plt.ylim(ymin-y_offset,ymax+y_offset)
plt.show()




