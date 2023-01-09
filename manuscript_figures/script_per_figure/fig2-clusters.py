#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:17:51 2022

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
# import sys
# sys.path.append("/Users/ccamargo/Documents/github/SLB/")

# from utils_SLB import cluster_mean, plot_map_subplots, sum_linear, sum_square, get_dectime
# from utils_SLB import plot_map2 as plot_map

# sys.path.append("/Users/ccamargo/Documents/py_scripts/")
# import utils_SL as sl

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from matplotlib.cm import ScalarMappable
cmap_trend = cm.cm.balance
cmap_unc = cm.tools.crop(cmap_trend,0,3,0)
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
# make_figure(save=False)
path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'
def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'clusters',
                figfmt='png'
                ):
    # SOM & dmaps Clusters
    df_dmap, _, _ = prep_dmaps(info = True)
    coords, coordsx1 = central_coords()
    clim=5
    
    # cmap = cmap_trend
    # datasets = ['alt','sum', 'steric','barystatic', 'dynamic']
     # 
    global settings
    settings = set_settings()
    fsize = settings['fsize']
    #% %  make list with datasets
    # titles = [settings['titles_dic'][dataset] for dataset in datasets]
    # das_unc,das_trend,das_ts = das(datasets)
    dic = load_data(fmt = 'pkl')
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    #% % plot trends for each component
    
    # dataset = das_trend
    
    cmin=-clim;cmax=clim
    clabel='Trend \nmm/yr'
    lon[-1]=360
    
    ####
    grid = False
    nrow = 2
    ncol= 1
    gs = GridSpec(nrow, ncol)
    
    # interval=1
    fig = plt.figure(figsize=fsize,dpi=100)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    proj=ccrs.Robinson(central_longitude=settings['lon0'])
    
    # dmaps
    key='dmap'
    mask_clusters = np.array(dic[key]['mask'])
    mask_dmap = np.array(mask_clusters)
    for ig in [1,4,54]:
        mask_dmap[mask_dmap==ig]=np.nan
    mask_dmap[np.isfinite(mask_dmap)]=1
    n_clusters = dic[key]['n']
    colors = []
    for n in range(3):
        for i in range(20):
            colors.append(cmat.tab20(i))
    for i in range(14):
        colors.append(cmat.tab20(i))
    cmap = col.ListedColormap(colors)
    
    data = np.array(mask_clusters*mask_dmap)
    title = ' $\delta$-MAPS Clusters'    
    clabel='cluster number'
    cmin = 0 
    cmax=n_clusters
    idata = 0
    ax = plt.subplot(gs[idata], projection=proj
                         #Mercator()
                         )
    ax.set_global()
    Z = np.array(data)
    Z[np.isnan(Z)]=0
    plt.contour(lon,lat,Z,
                levels=np.arange(0,cmax),
                linewidths=0.01,
                colors='gray',
               #  levels=lv,linewidths=np.repeat(0.1,len(lv)),
               # corner_mask=True
                transform=ccrs.PlateCarree(),
               )
    
    # cmap = plt.get_cmap(cmap, len(np.unique(data[np.isfinite(data)])))
    
    
    plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                   alpha=1,
                transform = ccrs.PlateCarree(),cmap=cmap)
    for icluster in np.array(df_dmap.cluster_n):
        central_lat = df_dmap.loc[df_dmap.cluster_n==icluster]['lat']
        central_lon = df_dmap.loc[df_dmap.cluster_n==icluster]['lon']
        if icluster==97:
            central_lon=105
        ax.text(central_lon,central_lat,
                np.array(df_dmap.loc[df_dmap.cluster_n==icluster].index)[0],
                # int(icluster),
                color=np.array(df_dmap.loc[df_dmap.cluster_n==icluster]['color'])[0],
                transform=ccrs.PlateCarree(),)
        
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                edgecolor=None, facecolor=settings['landcolor']))
    
    if grid:
        gl=ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER
        
    title = '\n({}). '.format(settings['letters'][idata])+str(title)
    plt.title(title,fontsize=settings['fontsize'])
    
    # SOM
    key='som'
    mask_clusters = np.array(dic[key]['mask'])
    n_clusters = dic[key]['n']
    cmap = 'tab20'
    data = np.array(mask_clusters)
    title = 'SOM Regions'    
    clabel='region number'
    cmin = 0 
    cmax = 19
    idata = 1
    cmap = plt.get_cmap(cmap, n_clusters)
    ax = plt.subplot(gs[idata], projection=proj
                         #Mercator()
                         )
    ax.set_global()
    ##             min_lon,,max_lon,minlat,maxlat
    
    mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    
    
    
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                edgecolor=None,# 'gray',
                                                facecolor=settings['landcolor']))
    
    for icluster in range(1,19):
        for x,y in zip(coords[icluster]['x'],coords[icluster]['y']):
            ax.text(x,y,icluster,
                          transform=ccrs.PlateCarree(),
                   )
        
    if grid:
        gl=ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER
    
        
    title = '({}). '.format(settings['letters'][idata])+str(title)
    plt.title(title,fontsize=settings['fontsize'])
    
    # color bar
    labels = np.arange(1,cmax,1)
    loc = np.linspace(0.5,18,len(labels))
    cbar_ax = fig.add_axes([0.285, 0.07, 0.455, 0.04])
    cbar = plt.colorbar(mm,cax=cbar_ax,orientation='horizontal')
    
    cbar.set_label(label=clabel,size=settings['fontsize']-10, family='serif')    
    cbar.ax.tick_params(labelsize=settings['fontsize']-10) 
    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    
    
    # plt.tight_layout()
    
    
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


# def prep_som():
#     key='som'
#     dic = load_data(fmt = 'pkl')
#     mask_clusters = np.array(dic[key]['mask'])
#     n_clusters = dic[key]['n']
#     cmap = 'tab20'
#     oceanmask = np.array(mask_clusters)
#     oceanmask[np.isfinite(oceanmask)]=1
#     return


def central_coords():
    coords = {}
    ##
    icluster = 1
    coords[icluster] = {
        'x' : [304,266],
        'y' : [33.5,25.5]
                       }
    ##
    icluster = 2
    coords[icluster] = {
        'x' : [358, 330, 323],
        'y' : [-47.5, -12.5, 45.5]
                       }
    
    
    icluster = 3
    coords[icluster] = {
        'x' : [320 ],
        'y' : [ -47.5]
                       }
    ##
    
    icluster = 4
    coords[icluster] = {
        'x' : [336,344,15 ],
        'y' : [ 33.5,-21.5,-39.5]
                       }
    ##
    
    icluster = 5
    coords[icluster] = {
        'x' : [ 3],
        'y' : [ -10.5]
                       }
    ##
    
    icluster = 6
    coords[icluster] = {
        'x' : [314,325,307],
        'y' : [ 49.5,10.5,-58.5]
                       }
    ##
    
    
    icluster = 7
    coords[icluster] = {
        'x' : [292,330 ],
        'y' : [15.5,-33.5]
                       }
    ##
    
    icluster = 8
    coords[icluster] = {
        'x' : [357,343 ],
        'y' : [63.5,27.5]
                       }
    ##
    
    icluster = 9
    coords[icluster] = {
        'x' : [324 ],
        'y' : [ 56.5]
                       }
    ##
    
    
    icluster = 10
    coords[icluster] = {
        'x' : [ 177,94,183,252],
        'y' : [ 32.5,-24.5,-34.5,-44.5]
                       }
    ##
    
    icluster = 11
    coords[icluster] = {
        'x' : [ 58,43,156,212,154],
        'y' : [ 2.5,-55.5,-23.5,-45.5,49.5]
                       }
    ##
    
    icluster = 12
    coords[icluster] = {
        'x' : [ 252,80],
        'y' : [ 0.5,-12.5]
                       }
    ##
    
    icluster = 13
    coords[icluster] = {
        'x' : [91,110,132,211,179 ],
        'y' : [ -1.5,7.5,-40.5,-24.5,43.5]
                       }
    ##
    
    icluster = 14
    coords[icluster] = {
        'x' : [80,238,278,165,188,204 ],
        'y' : [ -47.5,-24.5,-56.5,-30.5,-52.5,44.5]
                       }
    ##
    
    icluster = 15
    coords[icluster] = {
        'x' : [ 56,194,227,273,184],
        'y' : [ -14.5,54.5,38.5,-14.5,-60.5]
                       }
    ##
    
    icluster = 16
    coords[icluster] = {
        'x' : [ 140],
        'y' : [ 4.5]
                       }
    ##
    
    icluster = 17
    coords[icluster] = {
        'x' : [ 33,177,150],
        'y' : [ -38.5,19.5,-60.5]
                       }
    ##
    
    icluster = 18
    coords[icluster] = {
        'x' : [ 215,219,261,237],
        'y' : [ 49.5,21.5,-21.5,-57.5]
                       }
    ##
    
    coordsx1 = {}
    ##
    icluster = 1
    coordsx1[icluster] = {
        'x' : [304],
        'y' : [33.5]
                       }
    ##
    icluster = 2
    coordsx1[icluster] = {
        'x' : [358],
        'y' : [-47.5]
                       }
    
    
    icluster = 3
    coordsx1[icluster] = {
        'x' : [320 ],
        'y' : [ -47.5]
                       }
    ##
    
    icluster = 4
    coordsx1[icluster] = {
        'x' : [336],
        'y' : [33.5]
                       }
    ##
    
    icluster = 5
    coordsx1[icluster] = {
        'x' : [ 3],
        'y' : [ -10.5]
                       }
    ##
    
    icluster = 6
    coordsx1[icluster] = {
        'x' : [325],
        'y' : [10.5]
                       }
    ##
    
    
    icluster = 7
    coordsx1[icluster] = {
        'x' : [330 ],
        'y' : [-33.5]
                       }
    ##
    
    icluster = 8
    coordsx1[icluster] = {
        'x' : [357],
        'y' : [62.5]
                       }
    ##
    
    icluster = 9
    coordsx1[icluster] = {
        'x' : [324 ],
        'y' : [ 56.5]
                       }
    ##
    
    
    icluster = 10
    coordsx1[icluster] = {
        'x' : [ 177],
        'y' : [ 32.5]
                       }
    ##
    
    icluster = 11
    coordsx1[icluster] = {
        'x' : [ 58],
        'y' : [ 2.5]
                       }
    ##
    
    icluster = 12
    coordsx1[icluster] = {
        'x' : [ 252],
        'y' : [ 0.5]
                       }
    ##
    
    icluster = 13
    coordsx1[icluster] = {
        'x' : [130 ],
        'y' : [-43.5]
                       }
    ##
    
    icluster = 14
    coordsx1[icluster] = {
        'x' : [80],
        'y' : [ -47.5]
                       }
    ##
    
    icluster = 15
    coordsx1[icluster] = {
        'x' : [ 56],
        'y' : [ -14.5]
                       }
    ##
    
    icluster = 16
    coordsx1[icluster] = {
        'x' : [ 140],
        'y' : [ 4.5]
                       }
    ##
    
    icluster = 17
    coordsx1[icluster] = {
        'x' : [ 175],
        'y' : [ 17.5]
                       }
    ##
    
    icluster = 18
    coordsx1[icluster] = {
        'x' : [219],
        'y' : [18.5]
                       }

    return coords, coordsx1
    return coords, coordsx1

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


