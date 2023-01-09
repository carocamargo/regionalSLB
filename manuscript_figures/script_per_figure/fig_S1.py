#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:14:38 2022

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
                figname = 'components_unc-contour',
                figfmt='png'
                ):
    # plot uncertainty for each component
    interval = 0.1
    datasets = ['alt','sum', 'steric','barystatic', 'dynamic']
     
    global settings
    settings = set_settings()
    #% %  make list with datasets
    titles = [settings['titles_dic'][dataset] for dataset in datasets]
    das_unc,das_trend,das_ts = das(datasets)
    dic = load_data(fmt = 'pkl')
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    #% % plot trends for each component
    
    clim=3
    dataset = das_unc
    cmap=cmap_unc
    cmin=0;cmax=clim
    clabel='Uncertainty \nmm/yr'
    
    lon[-1]=360
    

    nrow=8;ncol=2
    proj = 'robin'
    plot_type = 'contour'
    fig = plt.figure(figsize=(15,12),dpi=100)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    letters = settings['letters']
    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=settings['lon0'])
    else:
        proj=ccrs.PlateCarree()
    
    idata = 0
    ax = plt.subplot2grid((nrow,ncol), (0, 0), 
                          rowspan=3, projection=proj)
    ax.set_global()
    data = dataset[idata]
    if plot_type=='pcolor':
        mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    if plot_type =='contour':
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
    
        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    # plot global mean contour:
    _,glb,_=sl.reg_to_glb(data,lat,lon)
    glb=np.round(glb,3)
    cs=ax.contour(lon,lat,data,levels=[glb],
                    # vmin=-0.6,
                    # vmax=0.6,
                transform = ccrs.PlateCarree(),
                #cmap='coolwarm',#extend='both'
                colors=('black',),linestyles=('-',),linewidths=(1,)
                )
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=12)
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=20)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
    ax = plt.subplot2grid((nrow,ncol), (3, 0), rowspan=3, projection=proj)
    ax.set_global()
    idata = idata+1
    data = dataset[idata]
    if plot_type=='pcolor':
        mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    if plot_type =='contour':
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
    
        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    # plot global mean contour:
    _,glb,_=sl.reg_to_glb(data,lat,lon)
    glb=np.round(glb,3)
    cs=ax.contour(lon,lat,data,levels=[glb],
                    # vmin=-0.6,
                    # vmax=0.6,
                transform = ccrs.PlateCarree(),
                #cmap='coolwarm',#extend='both'
                colors=('black',),linestyles=('-',),linewidths=(1,)
                )
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=12)
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=20)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
    ax = plt.subplot2grid((nrow,ncol), (0, 1),rowspan=2, projection=proj)
    ax.set_global()
    idata = idata+1
    data = dataset[idata]
    if plot_type=='pcolor':
        mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    if plot_type =='contour':
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
    
        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    # plot global mean contour:
    _,glb,_=sl.reg_to_glb(data,lat,lon)
    glb=np.round(glb,3)
    cs=ax.contour(lon,lat,data,levels=[glb],
                    # vmin=-0.6,
                    # vmax=0.6,
                transform = ccrs.PlateCarree(),
                #cmap='coolwarm',#extend='both'
                colors=('black',),linestyles=('-',),linewidths=(1,)
                )
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=12)
    
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=20)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
    ax = plt.subplot2grid((nrow,ncol), (2, 1),rowspan=2, projection=proj)
    ax.set_global()
    idata = idata+1
    data = dataset[idata]
    if plot_type=='pcolor':
        mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    if plot_type =='contour':
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
    
        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    # plot global mean contour:
    _,glb,_=sl.reg_to_glb(data,lat,lon)
    glb=np.round(glb,3)
    cs=ax.contour(lon,lat,data,levels=[glb],
                    # vmin=-0.6,
                    # vmax=0.6,
                transform = ccrs.PlateCarree(),
                #cmap='coolwarm',#extend='both'
                colors=('black',),linestyles=('-',),linewidths=(1,)
                )
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=12)
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=20)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
    ax = plt.subplot2grid((nrow,ncol), (4, 1),rowspan=2, projection=proj)
    ax.set_global()
    idata = idata+1
    data = dataset[idata]
    if plot_type=='pcolor':
        mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    if plot_type =='contour':
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
    
        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    # plot global mean contour:
    _,glb,_=sl.reg_to_glb(data,lat,lon)
    glb=np.round(glb,3)
    cs=ax.contour(lon,lat,data,levels=[glb],
                    # vmin=-0.6,
                    # vmax=0.6,
                transform = ccrs.PlateCarree(),
                #cmap='coolwarm',#extend='both'
                colors=('black',),linestyles=('-',),linewidths=(1,)
                )
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=12)
        
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=20)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
    plt.tight_layout()
    # # fig.subplots_adjust(right=0.8)
    cbar_ax2 = fig.add_axes([0.25, 0.2, 0.5, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=settings['fontsize']-5, family='serif')    
    cbar2.ax.tick_params(labelsize=settings['fontsize']-5) 
    
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

