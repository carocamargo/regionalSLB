#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:58:28 2022

@author: ccamargo
"""

# Import libraries
import xarray as xr
import numpy as np
# import os
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")

from utils_SLB import cluster_mean#, plot_map_subplots, sum_linear, sum_square, get_dectime
# from utils_SLB import plot_map2 as plot_map

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cmocean as cm
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
def prep_drivers():
    dic = load_data()
    df_dmap, mask_dmap, df3 = prep_dmaps(info=False)
    df = dic['som']['df']
    df = pd.DataFrame({'Altimetry':df['alt_tr'],
                      'Altimetry_err':df['alt_unc'],
                       'Sum_err':df['sum_unc'],
                      'Dynamic':df['dynamic_tr'],
                       'GRD':df['barystatic_tr'],
                      'Steric':df['steric_tr']})
    
    df['cluster_n'] = [int(i) for i in dic['som']['df']['cluster_n']]
    df['Sum'] = df['Steric'] + df['Dynamic'] + df['GRD']
    
    df = df.set_index('cluster_n')
    
    df2 = df3[df3['color']=='m']
    
    sel= [87,28,74,43,38,80,68,71,37,93,95,49,97,72,67,94,82,90]
    df2['n'] = df2.index
    df2 = df2.set_index('cluster_n')
    df2 = df2.reindex(sel)
    df2
    df2 = df2.set_index('n')
    
    return df,df2


path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'
# make_figure(save=False)

def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'budget_drivers',
                figfmt='png'
                ):
    
    settings = set_settings()
    df,df2 = prep_drivers()
    dic = load_data()
    fontsize=20
    labelsize=15
    legendsize = 13
    offset=0.15
    fig = plt.figure(figsize=(15,15),dpi=300)
    nrow=6;ncol=2
    colors_dic = settings['colors_dic']
    #### trends
    ## SOM
    ax = plt.subplot2grid((nrow,ncol),(0,0),rowspan=2,colspan=2)
    
    plt.axhline(0,color='k',linestyle='--')
    comp = ['Dynamic','Barystatic','Steric']
    colors = [colors_dic[c] for c in comp]
    comp = ['Dynamic','GRD','Steric']
    df[comp].plot(
            kind='bar',
            stacked=True, 
            ax = ax,
            zorder=0,
            alpha=0.8,
            color=colors,
            #edgecolor="k"
    )
    x = np.arange(len(df))
    name = "Altimetry"
    y = np.array(df[name])
    yerr=df['{}_err'.format(name)]/2
    plt.scatter(x-offset,y,marker='*',
                zorder=2,
                s=100,
                color=colors_dic[name],
                label=name)
    plt.errorbar(x-offset,y,
                 yerr=yerr,
                 alpha=0.5,
                 capsize=3,capthick=2,
                 ecolor=colors_dic[name],
                 #lw=2,
                 zorder=1,
                 fmt='none')
    
    name = "Sum"
    y = np.array(df[name])
    yerr=df['{}_err'.format(name)]/2
    plt.scatter(x+offset,y,marker='^',
                s=100,zorder=3,
                color=colors_dic[name],label=name)
    plt.errorbar(x+offset,y,
                 yerr=yerr,
                 alpha=0.5,
                 capsize=3,capthick=2,
                 ecolor=colors_dic[name],
                 #lw=2,
                 zorder=1,
                 fmt='none')
    plt.legend(fontsize=legendsize,loc = 'upper left')
    
    plt.xlabel('SOM Region number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    plt.ylabel('mm/yr',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.text(17,6,'({})'.format(settings['letters'][0]),fontsize=settings['fontsize'])
    
    ## dmaps
    ax = plt.subplot2grid((nrow,ncol),(2,0),rowspan=2,colspan=2)
    
    
    plt.axhline(0,color='k',linestyle='--')
    comp = ['Dynamic','Barystatic','Steric']
    colors = [colors_dic[c] for c in comp]
    comp = ['Dynamic','GRD','Steric']
    df2[comp].plot(
            kind='bar',
            stacked=True, 
            ax = ax,
            zorder=0,
            alpha=0.8,
            color=colors,
            #edgecolor="k"
    )
    x = np.arange(len(df2))
    name = "Altimetry"
    y = np.array(df2[name])
    yerr=df2['{}_err'.format(name)]/2
    plt.scatter(x-offset,y,marker='*',
                zorder=2,
                s=100,
                color=colors_dic[name],
                label=name)
    plt.errorbar(x-offset,y,
                 yerr=yerr,
                 alpha=0.5,
                 capsize=3,capthick=2,
                 ecolor=colors_dic[name],
                 #lw=2,
                 zorder=1,
                 fmt='none')
    
    name = "Sum"
    y = np.array(df2[name])
    yerr=df2['{}_err'.format(name)]/2
    plt.scatter(x+offset,y,marker='^',
                s=100,zorder=3,
                color=colors_dic[name],label=name)
    plt.errorbar(x+offset,y,
                 yerr=yerr,
                 alpha=0.5,
                 capsize=3,capthick=2,
                 ecolor=colors_dic[name],
                 #lw=2,
                 zorder=1,
                 fmt='none')
    plt.legend(fontsize=legendsize,loc = 'upper left')
    
    plt.xlabel('$\delta$-MAPS Region number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    plt.ylabel('mm/yr',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.text(17,6,'({})'.format(settings['letters'][1]),
             fontsize=settings['fontsize'])
    
    
    ###### Time Series
    comp = ['Altimetry','Steric','Barystatic','Dynamic']
    colors = [colors_dic[c] for c in comp]
    var = ['alt','steric','barystatic','dynamic']
    titles = ['Altimetry','Steric','GRD','Dynamic']
    tdec = dic['dims']['time']['tdec']
    # da = xr.open_dataset('/Volumes/LaCie_NIOZ/data/budget/ts/alt.nc')
    # tdec, _ = sl.get_dec_time(da.time)
    tdec = tdec[0:288]
    lat = dic['dims']['lat']['lat']
    lon = dic['dims']['lon']['lon']
    
    #######
    
    
    n_cluster = 2
    y=1
    x=2
    
    ts=np.zeros((n_cluster,len(var),len(tdec)))
    tr=np.zeros((n_cluster,len(var)))
    i=0
    n_dmap = [82,45]
    for icluster_som, icluster_dmap in zip([1,12],[87,49]):
        ax = plt.subplot2grid((nrow,ncol),(4,i),rowspan=2)
        # SOM
        mask=np.array(dic['som']['mask'])
        mask[np.where(mask!=icluster_som)]=np.nan
        mask[np.isfinite(mask)]=1
        for ivar,v in enumerate(var):
            ts[i,ivar,:] = cluster_mean(np.array(dic[v]['ts']),
                                        mask,time=tdec,lat=lat,lon=lon,norm=True)
            out = sl.get_OLS_trend(tdec,ts[i,ivar])
            tr[i,ivar]=out[0]
            plt.plot(tdec,ts[i,ivar,:],
                     c=colors[ivar],
                     linewidth=2,
                     label='{}'.format(titles[ivar]))
            
        # dmap
        mask=np.array(dic['dmap']['mask'])
        mask[np.where(mask!=icluster_dmap)]=np.nan
        mask[np.isfinite(mask)]=1
        for ivar,v in enumerate(var):
            ts[i,ivar,:] = cluster_mean(np.array(dic[v]['ts']), 
                                        mask,time=tdec,lat=lat,lon=lon,norm=True)
            out = sl.get_OLS_trend(tdec,ts[i,ivar])
            tr[i,ivar]=out[0]
            plt.plot(tdec,ts[i,ivar,:],
                     c=colors[ivar],
                     linewidth=2,
                     linestyle='--',
                    #  label='{}'.format(titles[ivar])
                    )
        
        plt.title('({}). Regions {} and {}'.format(settings['letters'][i+2],
                                                   icluster_som, n_dmap[i]),
                  fontsize=settings['fontsize'])
        plt.ylim(-200,200)
        if icluster_som ==1:
            # plt.legend()
            plt.ylabel('mm',fontsize=settings['fontsize'])
    
        i=i+1
        plt.xlabel('Time',fontsize=settings['fontsize'])
        plt.legend(fontsize=legendsize,loc = 'lower left')
    plt.tight_layout()
    
    
    ######
    plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return

def make_figure_SI_ts(save=True,
                path_to_figures = path_figures,
                figname = 'budget_drivers',
                figfmt='png'):
    
    settings = set_settings()
    letters = settings['letters']
    df,df2 = prep_drivers()
    _, _, df3 = prep_dmaps(info=False)
    sel= [87,28,74,43,38,80,68,71,37,93,95,49,97,72,67,94,82,90]

    dic = load_data()
    colors_dic = settings['colors_dic']
    colors = [colors_dic[c] for c in ['Altimetry', 'Steric', 'Barystatic', 'Dynamic']]
    var = ['alt','steric','barystatic','dynamic']
    titles = ['Altimetry','Steric','GRD','Dynamic']
    n_cluster = len(df)
    y=6
    x=3
    tdec = dic['dims']['time']['tdec']
    lat = dic['dims']['lat']['lat']
    lon = dic['dims']['lon']['lon']
    
    fig = plt.figure(figsize=(20,20),dpi=300)
    for i in range(n_cluster):
        icluster = i+1
        plt.subplot(y,x,icluster)
    
        mask=np.array(dic['som']['mask'])
        mask[np.where(mask!=icluster)]=np.nan
        mask[np.isfinite(mask)]=1
        for ivar,v in enumerate(var):
            ts = cluster_mean(np.array(dic[v]['ts']), mask,time=tdec,lat=lat,lon=lon,norm=True)
            #out = sl.get_OLS_trend(tdec,ts[i,ivar])
            #tr[i,ivar]=out[0]
            plt.plot(tdec,ts,
                     c=colors[ivar],
                     linewidth=2,
                     label='{}'.format(titles[ivar]))
            
        # DMAPS
        icluster_dmap = sel[i]
        mask=np.array(dic['dmap']['mask'])
        mask[np.where(mask!=icluster_dmap)]=np.nan
        mask[np.isfinite(mask)]=1
        for ivar,v in enumerate(var):
            ts = cluster_mean(np.array(dic[v]['ts']), mask,time=tdec,lat=lat,lon=lon,norm=True)
            #out = sl.get_OLS_trend(tdec,ts[i,ivar])
            #tr[i,ivar]=out[0]
            plt.plot(tdec,ts,
                     c=colors[ivar],
                     linewidth=2,
                     linestyle = '--',
                    #  label='{}'.format(titles[ivar])
                    )
            
        
        plt.title('({}). Region {};{}'.format(letters[i],icluster,df3[df3['cluster_n']==sel[i]].index[0]))
        plt.ylim(-200,200)
        if icluster ==18:
            plt.legend()
        if icluster >15:
            plt.xlabel('Time')
        if i%3==0:
            plt.ylabel('mm')
            
    plt.tight_layout()
    plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return

def make_figure_SI_dmaps(save = True, 
                   path_to_figures = path_figures,
                   figname = 'budget_drivers_dmaps',
                   figfmt='png'):
    fontsize=20
    labelsize=15
    legendsize = 13
    df_dmap, mask_dmap, df3 = prep_dmaps(info=False)
    settings = set_settings()
    colors_dic = settings['colors_dic']
    fig = plt.figure(figsize=(15,20),dpi=300)
    
    #### trends
    i=0
    for j in range(4):
        ax = plt.subplot(4,1,j+1)
        df2 = df3.iloc[i:(i+23)]
        i=i+23
    
        plt.axhline(0,color='k',linestyle='--')
        comp = ['Dynamic','Barystatic','Steric']
        colors = [colors_dic[c] for c in comp]
        comp = ['Dynamic','GRD','Steric']
        df2[comp].plot(
                kind='bar',
                stacked=True, 
                ax = ax,
                zorder=0,
                alpha=0.8,
                color=colors,
            legend=False
                #edgecolor="k"
        )
        x = np.arange(len(df2))
        name = "Altimetry"
        y = np.array(df2[name])
        yerr=df2['{}_err'.format(name)]/2
        plt.scatter(x,y,marker='*',
                    zorder=2,
                    s=100,
                    color=colors_dic[name],
                    label=name)
        plt.errorbar(x,y,
                     yerr=yerr,
                     alpha=0.5,
                     capsize=3,capthick=2,
                     ecolor=colors_dic[name],
                     #lw=2,
                     zorder=1,
                     fmt='none')
    
        name = "Sum"
        y = np.array(df2[name])
        yerr=df2['{}_err'.format(name)]/2
        plt.scatter(x,y,marker='^',
                    s=100,zorder=3,
                    color=colors_dic[name],label=name)
        plt.errorbar(x,y,
                     yerr=yerr,
                     alpha=0.5,
                     capsize=3,capthick=2,
                     ecolor=colors_dic[name],
                     #lw=2,
                     zorder=1,
                     fmt='none')
        plt.xlabel('')
        plt.ylabel('mm/yr',fontsize=fontsize)
    # plt.legend(fontsize=legendsize,loc = 'upper left')
    
    plt.xlabel('Dmaps Region number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    # ax.legend(title='SUBJECT',title_fontsize=30,loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(bbox_to_anchor =(0.1,-0.35), loc='lower center', ncol=2)
    
    plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return

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
    
    
    settings['textsize'] = 13
    settings['labelsize']=18    
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

def prep_dmaps(info):
    lats = []
    lons = []
    size = []
    c = []
    n = []
    a = []
    key='dmap'
    dic = load_data(fmt = 'pkl')
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    million = 1e+6

    grid_area = get_grid_area(dic['landmask'])/million

    data = np.array(np.array(dic[key]['mask']))
    for ig in [1,4,54]:
        data[data==ig]=np.nan
    
    
    for icluster in np.unique(data[np.isfinite(data)]):
        # icluster = 1
        n.append(icluster)
        Z = np.array(data)
        Z[Z!=icluster] = np.nan
        Z[np.isfinite(Z)] = 1
        a.append(np.nansum(Z*grid_area))
        llon,llat = np.meshgrid(lon,lat)
    
        llon[np.isnan(Z)] = np.nan
        if np.max(np.diff(llon[np.isfinite(llon)]))>300:
            llon[llon<180]=np.nan
    
        llat[np.isnan(Z)] = np.nan
        lats.append(round_off_rating(np.nanmean(llat)))
        lons.append(int(np.nanmean(llon)))
        size.append(len(Z[np.isfinite(Z)]))
        if icluster in [87,28,74,43,38,80,68,71,37,93,95,49,97,72,67,94,82,90]:
            c.append('m')
        else:
            c.append('black')
    
    df_dmap = pd.DataFrame({
        'cluster_n':n,
        'lat':lats,
        'lon':lons,
        'size':size,
        'color':c,
        'area':a
    })
    # df.loc[df.cluster==97]
    
    # df_dmap['cluster_n'] = [int(i) for i in df_dmap['cluster']]
    
    # get dmaps datafreme
    df2 = dic['dmap']['df']
    df2 = pd.DataFrame({'Altimetry':df2['alt_tr'],
                      'Altimetry_err':df2['alt_unc'],
                       'Sum_err':df2['sum_unc'],
                      'Dynamic':df2['dynamic_tr'],
                       'GRD':df2['barystatic_tr'],
                      'Steric':df2['steric_tr'],
                        'res_tr':df2['res_tr'],
                        'res_unc':df2['res_unc']
                       })
    
    
    df2['cluster_n'] = [int(i) for i in dic['dmap']['df']['cluster_n']]
    df2['Sum'] = df2['Steric'] + df2['Dynamic'] + df2['GRD']
    len(df2)
    
    # remove interior/small seas
    for ig in [1,4,54]:
        df2 = df2.drop(df2[df2.cluster_n ==ig].index)
    
    # remove areas with NaNs    
    df2.dropna(inplace=True)
    
    
    df3 = pd.merge(df_dmap, df2, on="cluster_n")
    
    df3['index'] = [i+1 for i in df3.index]
    df3.set_index('index', inplace=True)
    
    df_dmap = df3.drop(df3[df3['size'] < 50].index)
    
    
    A = np.array(df3['Altimetry'])
    A_error = np.array(df3['Altimetry_err'])
    B = np.array(df3['Sum'])
    B_error = np.array(df3['Sum_err'])
    
    C = np.array([compare_values(A[i],A_error[i],B[i],B_error[i]) for i in range(len(A))])
    # res_closed = len(C[C==1])
    df3['closure'] = C
    ((df3['area']*df3['closure']).sum() * 100)/df3['area'].sum()
    if info:
        print('delta-maps has a total area of {}'.format(np.nansum(Z*grid_area)))
        print('and {} is closed within uncertainties'.format((df3['area']*df3['closure']).sum()) )
        print('That is, {} % of the ocean area is closed'.format(((df3['area']*df3['closure']).sum() * 100)/df3['area'].sum()))
    
    
    mask_clusters = np.array(dic[key]['mask'])
    mask_dmap = np.array(mask_clusters)
    for ig in [1,4,54]:
        mask_dmap[mask_dmap==ig]=np.nan
    mask_dmap[np.isfinite(mask_dmap)]=1

    df = dic['dmap']['df']
    df['cluster_n'] = [int(i) for i in df['cluster_n']]
    for ig in [1,4,54]:
        df.drop(df[df['cluster_n'] ==ig].index, inplace=True)
    # df
    dic['dmap']['df'] = df
    dic['dmap']['mask'] = dic['dmap']['mask'] * mask_dmap

    return df_dmap, mask_dmap, df3

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

def get_grid_area(grid):
    # given a grid that has dimensions: len(lat) x len(lon), compute the 
    #area in each grid cell, in km2
    # input: grid: numpy array
    
    earth_radius = 6371 # km
    # earth_radius = 6378137/1000# a more precise earth radius
    earth_diam = 2* earth_radius # diameter in km
    earth_circ = np.pi* earth_diam # earth's circunference in meters
    
    #% %
    # grid=np.zeros((360,720)) # 0.5 degree grid
    dimlat=grid.shape[0]
    dimlon=grid.shape[1]
    deltalat=180/dimlat
    deltalon=360/dimlon
    if deltalat==0.5:
        lat=np.arange(-89.875,90,deltalat)
        lon=np.arange(0.125,360,deltalon)
    else:
        lat=np.arange(-89.5,90,deltalat)
        lon=np.arange(0.5,360,deltalon)       

    # deltalat=lat[1]-lat[0]
    # deltalon=lon[1]-lon[0]  
    #Transform from degrees to km:
    deltay=(earth_circ*deltalat)/360 #lat to km
    deltax=(earth_circ*np.cos(np.radians(lat))*deltalon)/360 #lon to km
    
    grid_area=np.array([deltax*deltay]*len(lon)).transpose()
    # print(np.sum(grid_area))
    
    # earth_area = np.sum(grid_area)
    # earth_ref = 510072000 # km
    # print('My Earths surface is '+str(earth_area/earth_ref)+' of the google reference' )

    return grid_area

def round_off_rating(number):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""

    return round(number * 2) / 2
