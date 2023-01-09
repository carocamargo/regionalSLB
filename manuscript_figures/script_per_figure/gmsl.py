#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:09:27 2022

@author: ccamargo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:10:42 2022

@author: ccamargo
"""

# Import libraries
import xarray as xr
import numpy as np
# import os
import pandas as pd
import sys
# sys.path.append("/Users/ccamargo/Documents/github/SLB/")

# from utils_SLB import cluster_mean, plot_map_subplots, sum_linear, sum_square, get_dectime
# from utils_SLB import plot_map2 as plot_map

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.cm import ScalarMappable
cmap_trend = cm.cm.balance
cmap_unc = cm.tools.crop(cmap_trend,0,3,0)
# from matplotlib import cm as cmat
# import matplotlib.colors as col

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
def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'gmsl',
                figfmt='png'
                ):
    fontsize=25
    datasets = ['alt','sum', 'steric','barystatic', 'dynamic']
     
    global settings
    settings = set_settings()
    #% %  make list with datasets
    titles = [settings['titles_dic'][dataset] for dataset in datasets]
    das_unc,das_trend,das_ts = das(datasets)
    dic = load_data(fmt = 'pkl')
    landmask = dic['landmask']
    tdec = dic['dims']['time']['tdec']
    fig = plt.figure(figsize=(15,5),dpi=300)
    ax2 = plt.subplot(111)
    for idata,data in enumerate(das_ts):
        data = data*landmask
        mu = np.nanmean(data,axis=(1,2))
        out = sl.get_ts_trend(tdec,mu,plot=False)
        tr = np.round(out[0],2)
        if tr==0:
            tr=0.00
        ax2.plot(tdec, mu - np.nanmean(mu[144:276]),
                 color=settings['colors_dic'][settings['acronym_dic'][datasets[idata]]],
                label='{}: {:.2f} mm/yr'.format(titles[idata],tr),
                linewidth=3)
    plt.title('Global Mean Sea Level',fontsize=fontsize)
    plt.ylabel('mm',fontsize=fontsize-5)
    plt.xlabel('time',fontsize=fontsize-5)
    
    #. plt.legend(fontsize=fontsize-5)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5,-0.4),
              ncol=3, 
               fancybox=True, 
               shadow=True,
               fontsize=fontsize-8)
    
    plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return

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

def das(datasets):
    dic = load_data(fmt = 'pkl')
    das_unc = []
    das_trend = []
    das_ts = []
    for key in datasets:
        if key =='sterodynamic':
            das_unc.append(dic['steric']['unc'] + dic['dynamic']['unc'])
            das_trend.append(dic['steric']['trend'] + dic['dynamic']['trend'])
            das_ts.append(dic['steric']['ts'] + dic['dynamic']['ts'])
        else:
            das_unc.append(dic[key]['unc']*dic['landmask'])
            das_trend.append(dic[key]['trend']*dic['landmask'])
            das_ts.append(dic[key]['ts'])
    return das_unc,das_trend,das_ts
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

