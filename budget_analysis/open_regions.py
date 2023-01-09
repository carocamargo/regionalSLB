#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:09:14 2022

@author: ccamargo
"""
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm
import xarray as xr
import string
import numpy as np
#%% 
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
    
def compare_values(x,x_unc,y,y_unc):
    # is y within the interval of x:
    if x+x_unc>= y and x-x_unc<=y:
        #print('y within x')
        agree = 1
    elif y+y_unc>= x and y-y_unc<=x:
        #print('x within y')
        agree = 1
    else: 
        agree=0
    return agree

#%% load data

path = '/Volumes/LaCie_NIOZ/data/budget/'
dic = load_data(path=path, file='budget_v2')
lat = dic['dims']['lat']['lat']
lon = dic['dims']['lon']['lon']

key = 'dmap'
df = dic[key]['df']
mask = dic[key]['mask']
#%%
mask2=np.full_like(mask,np.nan)
t=1
for i,row in df.iterrows():
    A = np.array(row['alt_tr'])
    A_error = np.array(row['alt_unc'])
    B = np.array(row['sum_tr'])
    B_error = np.array(row['sum_unc'])

    C = compare_values(A,A_error,B,B_error)
    
    if C==0:
        # mask2=np.array(mask)
        mask2[mask==row['cluster_n']]=row['cluster_n']
        mask2[mask==row['cluster_n']]=t
        
        # plt.pcolor(lon,lat,mask2)
        t=t+1
#%%
def plot_open_regions(save=True):
    dic = load_data(path=path, file='budget_v2')
    lat = dic['dims']['lat']['lat']
    lon = dic['dims']['lon']['lon']
    
    key = 'dmap'
    df = dic[key]['df']
    mask = dic[key]['mask']
    mask2=np.full_like(mask,np.nan)
    t=1
    _,_,df3 = prep_dmaps(info=False)
    for i,row in df3.iterrows():
        A = np.array(row['Altimetry'])
        A_error = np.array(row['Altimetry_err'])
        B = np.array(row['Sum'])
        B_error = np.array(row['Sum_err'])
    
        C = compare_values(A,A_error,B,B_error)
        
        if C==0:
            # mask2=np.array(mask)
            mask2[mask==row['cluster_n']]=row['cluster_n']
            mask2[mask==row['cluster_n']]=t
            
            # plt.pcolor(lon,lat,mask2)
            t=t+1
    #% %
    clim=1
    cmap='tab20'
    clabel='Budget Residual (mm/yr)'
    plot_type = 'pcolor'
    proj='robin'
    fig = plt.figure(# figsize=fsize,
                     dpi=100)
    data = mask2
    cmin=np.nanmin(data)
    cmax=np.nanmax(data)
    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=settings['lon0'])
    else:
        proj=ccrs.PlateCarree()
    ax = plt.subplot(111, projection=proj
                         #Mercator()
                         )
    ax.set_global()
    ##             min_lon,,max_lon,minlat,maxlat
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
        interval=1
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)
    
        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    
        
        
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    plt.title(' $\delta$-MAPS regions with \n non-closed budgets')
    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    # d01 box
    
    # title = '({}). '.format(letters[idata+idata])+str(titles[idata])
    # plt.title(title,fontsize=fontsize)
