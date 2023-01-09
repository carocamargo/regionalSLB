#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 19:30:15 2022

@author: ccamargo
"""

# Import libraries
import xarray as xr
import numpy as np
# import os
import pandas as pd
# import sys
# sys.path.append("/Users/ccamargo/Documents/github/SLB/")

# from utils_SLB import cluster_mean#, plot_map_subplots, sum_linear, sum_square, get_dectime
# from utils_SLB import plot_map2 as plot_map

# sys.path.append("/Users/ccamargo/Documents/py_scripts/")
# import utils_SL as sl

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.cm import ScalarMappable
# cmap_trend = cm.cm.balance
# cmap_unc = cm.tools.crop(cmap_trend,0,3,0)
from matplotlib import cm as cmat
import matplotlib.colors as col
# import seaborn as sns
# import scipy.stats as st
# from scipy import stats
# import sklearn.metrics as metrics
# import random
import warnings
warnings.filterwarnings("ignore","Mean of empty slice", RuntimeWarning)

import string
#%%

path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'
# make_figure(save=False)

def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'som_extra',
                figfmt='png'):
    settings = set_settings()
    dic = load_data()
    lat = dic['dims']['lat']['lat']
    lon = dic['dims']['lon']['lon']
    
    wi, hi = settings['fsize']
    dpi = settings['dpi']
    fig = plt.figure(figsize=(wi,hi), facecolor='w',dpi=dpi)
    cmin=1
    # dic = load_data()
    # interval = 1
    
    for i,n in enumerate([4,9]):
        fname = 'som_{}x{}_sig_init2_norm_range_train_n10_ngb_function_ep_mask_ocean_.nc'.format(n,n)
    
        # SOM 4x4 
        mask,cmap,cmax = load_SOMs(
                        path = '/Volumes/LaCie_NIOZ/reg/SOM_MATLAB/output/',
                        file = fname
                        )
        
    
        ax1 = plt.subplot(2,1,int(i+1), 
                         # projection=ccrs.PlateCarree()
                          projection=ccrs.Robinson(central_longitude=settings['lon0'])
                         )
        ax1.set_global()
        ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                    edgecolor='gray', facecolor=settings['landcolor']))
        
        cs1=ax1.pcolormesh(lon,lat,
                              mask,cmap=cmap,
                       vmin=cmin,vmax=cmax, 
                          zorder=0,
                        transform=ccrs.PlateCarree()
                       )
            
        plt.title('SOM {}x{}'.format(n,n),fontsize=settings['fontsize'])
    
        cbar = plt.colorbar(cs1,orientation='vertical')
        cbar.set_label(label='Domains',size=settings['fontsize']-2, family='serif')
   
    
    plt.show()
    
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return

def load_SOMs(
        path,
        file
        ):
    ds = xr.open_dataset(path+file)
    mask = np.array(ds.bmu_map)
    ds
    clusters = np.array(ds.neurons)
    cmax=len(clusters)
    colors = []
    if cmax >20:
        for n in range(4):
            for i in range(20):
                colors.append(cmat.tab20(i))
        colors.append(cmat.tab20(i+1))
    else:
        for i in range(cmax):
            colors.append(cmat.tab20(i))
    cmap = col.ListedColormap(colors)

    
    return mask,cmap,cmax


def set_settings():
    global settings
    settings = {}
    
    # Global plotting settings
    settings['fontsize']=25
    settings['lon0']=201;
    settings['fsize']=(15,10)
    settings['proj']='robin'
    settings['land']=True
    settings['grid']=False
    settings['landcolor']='papayawhip'
    settings['extent'] = False
    settings['plot_type'] = 'contour'
    settings['dpi']= 300
    
    settings['textsize'] = 13
    settings['labelsize']=18  
    settings['ticksize']=15
    settings['colors_dic'] = {
        'Altimetry':'mediumseagreen',
        'Sum':'mediumpurple',
        'Steric':'goldenrod',
        'Dynamic':'indianred',
        'Barystatic':'steelblue',
        'Sterodynamic':'palevioletred',
        'Residual':'gray'
        }
    settings['acronym_dic'] = {
        'alt':'Altimetry',
        'sum':'Sum',
        'steric':'Steric',
        'res':'Residual',
        'dynamic':'Dynamic',
        'barystatic':'Barystatic'
        }
    settings['titles_dic'] = {
            'alt':r"$\eta_{obs}$",
             'steric': r"$\eta_{SSL}$", 
             'sum': r"$\sum(\eta_{SSL}+\eta_{GRD}+\eta_{DSL})$", 
              'barystatic':r"$\eta_{GRD}$",
             'res': r"$\eta_{obs} - \eta_{\sum(drivers)}$", 
              'dynamic':r"$\eta_{DSL}$",
                 }
    settings['letters'] = list(string.ascii_lowercase)
    
    return settings

def load_data(path = '/Volumes/LaCie_NIOZ/data/budget/',
              file = 'budget_v2',
              fmt = 'pkl'
              ):
    if fmt =='pkl':
        return pd.read_pickle(path+file+'.'+fmt)
    elif fmt =='nc':
        
        return xr.open_dataset(path+file+'.'+fmt)
    else:
        raise 'format not recognized'
        return 

