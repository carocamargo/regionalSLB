#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 19:22:55 2022

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
# make_figure(save=False)

def set_settings():
    global settings
    settings = {}
    

    settings['lon0']=201;
    settings['fsize']=(15,10)
    settings['proj']='robin'
    settings['land']=True
    settings['grid']=False
    settings['landcolor']='papayawhip'
    settings['extent'] = False
    settings['plot_type'] = 'contour'
    settings['dpi']= 300
    settings['font']='serif'
    settings['titlesize']=25
    settings['textsize'] = 13
    # settings['labelsize']=18  
    settings['ticksize']=15
    settings['ticklabelsize']=18
    settings['labelsize']=20
    settings['legendsize'] = 13
    settings['contourfontsize']=12
    
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
            'alt':r"$\eta_{total}$",
             'steric': r"$\eta_{SSL}$", 
             'sum': r"$\sum(\eta_{SSL}+\eta_{GRD}+\eta_{DSL})$", 
              'barystatic':r"$\eta_{GRD}$",
             'res': r"$\eta_{total} - \eta_{\sum(drivers)}$", 
              'dynamic':r"$\eta_{DSL}$",
                 }
    settings['letters'] = list(string.ascii_lowercase)
    
    return settings



def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'budget_drivers',
                figfmt='png'):
    settings = set_settings()
    letters = settings['letters']
    
    path = '/Volumes/LaCie_NIOZ/data/dynamicREA/'
    # load pickle module
    dyn_sl = pd.read_pickle(path+'dyn_sl.pkl')
    keys = [key for key in dyn_sl.keys() if not key.startswith('_')]
    # http://localhost:8888/notebooks/Documents/py_scripts/Dynamic_SL.ipynb
    t0=2005
    t1=2015
    ds=xr.open_dataset('/Volumes/LaCie_NIOZ/data/sterodynamic/dyn_sl_GRACE_{}-{}.nc'.format(t0,t1))
    ds= ds.sel(lat=slice(-66,66))
    keys = ['gra_dyn','dyn_rea']
    labels= ['GRACE AVG','REANALYSIS']

    # plot
    dpi=300
    wi=15;hi=8
    # dimlat=180;dimlon=360
    fontsize=25
    ticksize=20
    X=np.array(dyn_sl['_extra_']['lon'])
    X[-1]=360
    Y=np.array(dyn_sl['_extra_']['lat'])
    clim=7
    cmap='RdBu_r'
    cmap = cm.cm.balance
    cmax=clim
    cmin=-clim
    interval = 1
    t0=2005
    t1=2015

    labels= ['GRACE AVG','REANALYSIS', 'GRACE JPL','GRACE CSR']
    keys = ['gra_dyn','dyn_rea','jpl','csr']
    
    ## PLOT
    fig = plt.figure(figsize=(wi,hi), facecolor='w',dpi=dpi)

    for i, name in enumerate(keys):
        
        ax1 = plt.subplot(2,2,i+1, 
                         # projection=ccrs.PlateCarree()
                          projection=ccrs.Robinson()
                         )
        ax1.set_global()
        if i<2:
            data = np.array(dyn_sl[name]['DYNSL_trend'])
        else:
            data = np.array(ds.dynamic_sl[i-2])
        name = labels[i]
        ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                edgecolor='gray', 
                                                facecolor=settings['landcolor']))
    
        cs=ax1.pcolormesh(X,Y,data,cmap=cmap,
                   vmin=cmin,vmax=cmax, 
                      zorder=0,
                   transform=ccrs.PlateCarree())
        
        lv=np.arange(cmin,cmax+interval,interval)
        csf=plt.contourf(X,Y,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
        
        plt.title('({}).{}'.format(letters[i],name),size=fontsize,)
        
    
        
    plt.tight_layout()
    cbar_ax2 = fig.add_axes([0.13, -0.15, 0.75, 0.05])
    cbar2 = plt.colorbar(csf, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label(label='Trends \n {}-{} (mm/yr)'.format(t0,t1),size=fontsize, family='serif')
    cbar2.ax.tick_params(labelsize=ticksize) 
    
    plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')


    return

def make_figure_dyn_dif(save=True,
                path_to_figures = path_figures,
                figname = 'dynSL_dif',
                figfmt='png'):
    settings = set_settings()
    letters = settings['letters']
    
    path = '/Volumes/LaCie_NIOZ/data/dynamicREA/'
    # load pickle module
    dyn_sl = pd.read_pickle(path+'dyn_sl.pkl')
    keys = [key for key in dyn_sl.keys() if not key.startswith('_')]
    # http://localhost:8888/notebooks/Documents/py_scripts/Dynamic_SL.ipynb
    t0=2005
    t1=2015
    # ds=xr.open_dataset('/Volumes/LaCie_NIOZ/data/sterodynamic/dyn_sl_GRACE_{}-{}.nc'.format(t0,t1))
    # ds= ds.sel(lat=slice(-66,66))
    # keys = ['gra_dyn','dyn_rea']
    # labels= ['GRACE AVG','REANALYSIS']

    # plot
    dpi=settings['dpi']
    wi=15;hi=8
    # dimlat=180;dimlon=360
    fontsize=settings['titlesize']
    ticksize=settings['ticksize']
    X=np.array(dyn_sl['_extra_']['lon'])
    X[-1]=360
    Y=np.array(dyn_sl['_extra_']['lat'])
    clim=7
    # cmap='RdBu_r'
    cmap = cm.cm.balance
    cmax=clim
    cmin=-clim
    interval = 1
    t0=2005
    t1=2015
    dyn_sl['diff'] ={'DYNSL_trend': np.array(
        (dyn_sl['gra_dyn']['DYNSL_trend'] - dyn_sl['dyn_rea']['DYNSL_trend']) 
        )}
    labels= ['GRACE','Reanalysis', 'GRACE - Reanalysis']
    keys = ['gra_dyn','dyn_rea','diff']
    ##### HERE
    ## PLOT
    fig = plt.figure(figsize=(wi,hi), facecolor='w',dpi=dpi)

    for i, name in enumerate(keys):
        
        ax1 = plt.subplot(3,1,i+1, 
                         # projection=ccrs.PlateCarree()
                          projection=ccrs.Robinson()
                         )
        ax1.set_global()
        # if i<2:
        # if name=='diff':
        #     cmap = cm.cm.curl
        # else:
        #     cmap = cm.cm.balance
                
        data = np.array(dyn_sl[name]['DYNSL_trend'])

        name = labels[i]
        ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                edgecolor='gray', 
                                                facecolor=settings['landcolor']))
    
        cs=ax1.pcolormesh(X,Y,data,cmap=cmap,
                   vmin=cmin,vmax=cmax, 
                      zorder=0,
                   transform=ccrs.PlateCarree())
        
        lv=np.arange(cmin,cmax+interval,interval)
        csf=plt.contourf(X,Y,data,levels=lv,
                  # transform = ccrs.PlateCarree(),
                  cmap=cmap)
        
        plt.title('({}).{}'.format(letters[i],name),size=fontsize,)
        
    
        
    plt.tight_layout()
    cbar_ax2 = fig.add_axes([0.35, -0.05, 0.3, 0.05])
    cbar2 = plt.colorbar(csf, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label(label='Trends \n {}-{} (mm/yr)'.format(t0,t1),size=fontsize, family='serif')
    cbar2.ax.tick_params(labelsize=ticksize) 
    
    plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')


    return

