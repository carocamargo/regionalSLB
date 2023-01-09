#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:20:10 2022

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
# from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import math

import seaborn as sns
import scipy.stats as st
# from scipy import stats
import sklearn.metrics as metrics
import random
import warnings
warnings.filterwarnings("ignore","Mean of empty slice", RuntimeWarning)

import string
#%% path to save
path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/figures/RADSEA/'
# path_to_data = '/Volumes/LaCie_NIOZ/data/budget/'
# path_to_data = '/Users/ccamargo/Desktop/data/'
path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'

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

def load_data(path = path_to_data,
              file = 'budget',
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
    dic = load_data()
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

# def generate_colormap(number_of_distinct_colors: int = 80):
#     if number_of_distinct_colors == 0:
#         number_of_distinct_colors = 80

#     number_of_shades = 7
#     number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

#     # Create an array with uniformly drawn floats taken from <0, 1) partition
#     linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

#     # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
#     #     but each saw tooth is slightly higher than the one before
#     # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
#     arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

#     # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
#     arr_by_shade_columns = arr_by_shade_rows.T

#     # Keep number of saw teeth for later
#     number_of_partitions = arr_by_shade_columns.shape[0]

#     # Flatten the above matrix - join each row into single array
#     nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

#     # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
#     initial_cm = hsv(nums_distributed_like_rising_saw)

#     lower_partitions_half = number_of_partitions // 2
#     upper_partitions_half = number_of_partitions - lower_partitions_half

#     # Modify lower half in such way that colours towards beginning of partition are darker
#     # First colours are affected more, colours closer to the middle are affected less
#     lower_half = lower_partitions_half * number_of_shades
#     for i in range(3):
#         initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

#     # Modify second half in such way that colours towards end of partition are less intense and brighter
#     # Colours closer to the middle are affected less, colours closer to the end are affected more
#     for i in range(3):
#         for j in range(upper_partitions_half):
#             modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
#             modifier = j * modifier / upper_partitions_half
#             initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

#     return initial_cm
def generate_colormap():
    from matplotlib import cm as cmat
    import matplotlib.colors as col
    colors = []
    for n in range(3):
        for i in range(20):
            colors.append(cmat.tab20(i))
    for i in range(14):
        colors.append(cmat.tab20(i))
    cmap = col.ListedColormap(colors)
    return cmap    

# def generate_colormap(number_of_distinct_colors: int = 80):
#     # import numpy as np
#     import math
#     from matplotlib.cm import hsv
#     import matplotlib.colors as col
    
#     if number_of_distinct_colors == 0:
#         number_of_distinct_colors = 80

#     number_of_shades = 7
#     number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

#     # Create an array with uniformly drawn floats taken from <0, 1) partition
#     linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

#     # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
#     #     but each saw tooth is slightly higher than the one before
#     # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
#     arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

#     # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
#     arr_by_shade_columns = arr_by_shade_rows.T

#     # Keep number of saw teeth for later
#     number_of_partitions = arr_by_shade_columns.shape[0]

#     # Flatten the above matrix - join each row into single array
#     nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

#     # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
#     initial_cm = hsv(nums_distributed_like_rising_saw)

#     lower_partitions_half = number_of_partitions // 2
#     upper_partitions_half = number_of_partitions - lower_partitions_half

#     # Modify lower half in such way that colours towards beginning of partition are darker
#     # First colours are affected more, colours closer to the middle are affected less
#     lower_half = lower_partitions_half * number_of_shades
#     for i in range(3):
#         initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

#     # Modify second half in such way that colours towards end of partition are less intense and brighter
#     # Colours closer to the middle are affected less, colours closer to the end are affected more
#     for i in range(3):
#         for j in range(upper_partitions_half):
#             modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
#             modifier = j * modifier / upper_partitions_half
#             initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

#     return col.ListedColormap(initial_cm) 

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

#%% FIG 1 - trends
def make_figure_trends(save=True,
                path_to_figures = path_figures,
                figname = 'components_trends-contour',
                figfmt='png'
                ):
    # plot trends for each component
    clim=5
    interval = 0.1
    cmap = cmap_trend
    datasets = ['alt','sum', 'steric','barystatic', 'dynamic']
     
    global settings
    settings = set_settings()
    #% %  make list with datasets
    titles = [settings['titles_dic'][dataset] for dataset in datasets]
    das_unc,das_trend,das_ts = das(datasets)
    dic = load_data()
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    #% % plot trends for each component
    
    dataset = das_trend
    
    cmin=-clim;cmax=clim
    clabel='Trend \nmm/yr'
    lon[-1]=360
    

    nrow=8;ncol=2
    proj = 'robin'
    plot_type = 'contour'
    fig = plt.figure(figsize=(15,12),dpi=100)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family=settings['font'])
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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
    	edgecolor='gray', facecolor=settings['landcolor']))
    
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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
    	edgecolor='gray', facecolor=settings['landcolor']))
    
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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])

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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])

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
 
        
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))

    plt.tight_layout()
    # # fig.subplots_adjust(right=0.8)
    cbar_ax2 = fig.add_axes([0.25, 0.2, 0.5, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=settings['labelsize'], family=settings['font'])    
    cbar2.ax.tick_params(labelsize=settings['labelsize']) 
    
    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()
    return


#%% FIG S1 - uncs
def make_figure_uncs(save=True,
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
    dic = load_data()
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
    plt.rc('font', family=settings['font'])
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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])
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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])
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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])

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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])

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
    ax.clabel(cs,cs.levels,fmt='%5.2f',colors='k',fontsize=settings['contourfontsize'])
    title = '({}). '.format(letters[idata])+str(titles[idata])
    plt.title(title,fontsize=settings['titlesize'])

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
    plt.tight_layout()
    # # fig.subplots_adjust(right=0.8)
    cbar_ax2 = fig.add_axes([0.25, 0.2, 0.5, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=settings['titlesize']-5, family=settings['font'])    
    cbar2.ax.tick_params(labelsize=settings['titlesize']-5) 
    
    #plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return


#%% FIG SI - gmsl
def make_figure_gmsl(save=True,
                path_to_figures = path_figures,
                figname = 'gmsl',
                figfmt='png'
                ):
    datasets = ['alt','sum', 'steric','barystatic', 'dynamic']
     
    global settings
    settings = set_settings()
    fontsize=settings['titlesize']
    #% %  make list with datasets
    titles = [settings['titles_dic'][dataset] for dataset in datasets]
    das_unc,das_trend,das_ts = das(datasets)
    dic = load_data()
    landmask = dic['landmask']
    tdec = dic['dims']['time']['tdec']
    fig = plt.figure(figsize=(15,5),dpi=settings['dpi'])
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
    plt.ylabel('mm',fontsize=settings['labelsize'])
    plt.xlabel('time',fontsize=settings['labelsize'])
    
    #. plt.legend(fontsize=fontsize-5)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5,-0.4),
              ncol=3, 
               fancybox=True, 
               shadow=True,
               fontsize=settings['legendsize'])
    
    #plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG 2 - clusters
def make_figure_clusters(save=True,
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
    dic = load_data()
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
    plt.rc('font', family=settings['font'])
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
    title = ' $\delta$-MAPS Domains'
    clabel='domain number'
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
        c = np.array(df_dmap.loc[df_dmap.cluster_n==icluster].index)[0]
        if c in [82,69,66,33,88,45,92,62,89]:
            text = '{}*'.format(c)
            # text = str(np.array(df_dmap.loc[df_dmap.cluster_n==icluster].index)[0], '*')
        else: 
            text = c
        text = c
        if icluster==97:
            central_lon=105
        ax.text(central_lon,central_lat,
                text,
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
    plt.title(title,fontsize=settings['titlesize'])
    
    # SOM
    key='som'
    mask_clusters = np.array(dic[key]['mask'])
    n_clusters = dic[key]['n']
    cmap = 'tab20'
    data = np.array(mask_clusters)
    title = 'SOM Domains'
    clabel='SOM domain number'
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
            if icluster in [1,63,8,9,10,12,13,15,16]:
                text = '{}*'.format(icluster)
                # text = str(np.array(df_dmap.loc[df_dmap.cluster_n==icluster].index)[0], '*')
            else: 
                text = icluster
            text = icluster
            ax.text(x,y,text,
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
    plt.title(title,fontsize=settings['titlesize'])
    
    # color bar
    labels = np.arange(1,cmax,1)
    loc = np.linspace(0.5,18.5,len(labels))
    cbar_ax = fig.add_axes([0.285, 0.04, 0.455, 0.04])
    cbar = plt.colorbar(mm,cax=cbar_ax,orientation='horizontal')
    
    cbar.set_label(label=clabel,size=settings['labelsize'], 
                   family=settings['font'])    
    cbar.ax.tick_params(labelsize=settings['ticksize']) 
    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    
    
    # plt.tight_layout()
    
    
    #plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG 3 residuals map + scatter & histograms SI
def make_figure_hist(save=True,
                path_to_figures = path_figures,
                figname = 'budget_res_histogram',
                figfmt='png'
                ):
    # plot trends for each component
    clim=5
    dic = load_data()
    das_res = [dic['res']['trend'],
           dic['dmap']['df']['res_tr'],
           dic['som']['df']['res_tr'],
                ]
    das_alt = [dic['alt']['trend'],
               dic['dmap']['df']['alt_tr'],
               dic['som']['df']['alt_tr'],
                    ]
    das_sum = [dic['sum']['trend'],
               dic['dmap']['df']['sum_tr'],
               dic['som']['df']['sum_tr'],
                    ]
    
    das_res_u = [dic['res']['unc'],
               dic['dmap']['df']['res_unc'],
               dic['som']['df']['res_unc'],
                    ]
    
    # das_sum_u = [dic['sum']['unc'],
    #            dic['dmap']['df']['sum_unc'],
    #            dic['som']['df']['sum_unc'],
    #                 ]
    
    # das_alt_u = [dic['alt']['unc'],
    #            dic['dmap']['df']['alt_unc'],
    #            dic['som']['df']['alt_unc'],
    #                 ]
    global settings
    settings = set_settings()
    colors_dic = settings['colors_dic']
    letters = settings['letters']
    
    i=0
    ws = []
    clim=7.5
    titles=['1 degree', '$\delta$-MAPS', 'SOM']
    
    pos=[1,3,5]
    lims = [(0,11000),(0,45),(0,18)]
    fig = plt.figure(figsize=(10,15))
    for i in range(3):
        df = pd.DataFrame({'Altimetry':np.hstack(das_alt[i]),
                           'Sum':np.hstack(das_sum[i]),
                          # 'res':np.hstack(adas_res[i])
                          })
    
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df = df.drop('index',axis=1)
    
        ax = plt.subplot(3,2,pos[i])
        ax.tick_params(axis='both', which='major', 
                       labelsize=settings['ticklabelsize'])
        for name in ['Altimetry','Sum']:
            sns.histplot(df[name],kde=False, #label='Trend',
                         alpha=0.5,
                         label=name,
                         # stat="percent", 
                         color=colors_dic[name],
                        bins=np.arange(-clim, clim+0.1,0.5))
        
        if i==2:
            plt.xlabel('mm/yr', fontsize=settings['labelsize'])
        else:
            plt.xlabel('')
        plt.ylabel('Number regions',fontsize=settings['labelsize'])
        # plt.title(title[i])
        title = '({}). '.format(letters[i+i])+str(titles[i])
        plt.title(title,fontsize=settings['titlesize'])
        
        plt.ylim(lims[i])
        plt.legend(loc='upper left')
    i=0
    pos=[2,4,6]
    for trend,unc in zip(das_res,das_res_u):
        df= pd.DataFrame( {
                        'Trend':np.hstack(trend),
        'Unc':np.hstack(unc),
        })
        
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df = df.drop('index',axis=1)
        
        ax = plt.subplot(3,2,pos[i])
        ax.tick_params(axis='both', which='major', 
                       labelsize=settings['ticklabelsize'])
        c = {'Trend':'darkgray',
            'Unc':'lightpink'}
        for var in ['Trend','Unc']:
            sns.histplot(df[var],kde=False, #label='Trend',
                         alpha=0.5,
                         color=c[var],
                         label=var,
                         # stat="percent", 
                        bins=np.arange(-clim, clim+0.1,0.5))
    
        # plt.legend(prop={'size': 12})
        if i==2:
            plt.xlabel('mm/yr', fontsize=settings['labelsize'])
        else:
            plt.xlabel('')
        plt.ylabel('')
        plt.xlim(-clim,clim)
        
        plt.ylim(lims[i])
        var = 'Unc'
        x = np.array(df[var])
        ci_level=0.95
        ci = st.norm.interval(alpha=ci_level, loc=np.mean(x), scale=x.std())
        plt.axvline(x.mean(),color='k',linestyle='--')
    
        plt.axvline(ci[0],c=c[var],linestyle='--',alpha=1,label='{}% CI'.format(ci_level*100))
        plt.axvline(ci[1],c=c[var],linestyle='--',alpha=0.5)
        
        ci_width = np.abs(ci[0]-ci[1])/2
        ws.append(ci_width)
        # plt.title(title[i]+': {:.3f}'.format( ci_width))
        title = '({}). '.format(letters[i+i+1])+str(titles[i]+': {:.3f}'.format( ci_width))
        plt.title(title,fontsize = settings['titlesize'])
        
        i=i+1
        plt.legend(loc='upper left')
    #plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return ws
def make_figure_scatter(save=True,
                path_to_figures = path_figures,
                figname = 'residuals_map_scatter',
                figfmt='png',
                info=True
                ):
    
    ws = make_figure_hist(save=False)
    dic = load_data()
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    interval = 0.1
    global settings
    settings = set_settings()
    # colors_dic = settings['colors_dic']
    letters = settings['letters']
    
    wi=10
    hi=15
    fig = plt.figure(figsize=(wi,hi),dpi=settings['dpi'])
    nrow=3;ncol=2
    gs = GridSpec(nrow, ncol)
    textsize = settings['textsize']
    labelsize=settings['labelsize']
    fontsize=settings['titlesize']
    ticksize=settings['ticksize']
    df_dmap, mask_dmap, _ = prep_dmaps(info=False)
    
    #% %  Plot residual clusters
    #% % make list with datasets
    das_res_map = [dic['res']['trend'],
               dic['dmap']['res']['trend'] * mask_dmap,
               dic['som']['res']['trend'],
                    ]
    
    titles = ["1 degree",
                r"$\delta$-MAPS", 
              r"SOM",
              # r"unc", r"unc",                     
              ]
    for da,title in zip(das_res_map,titles):
        print(title,'min:',np.round(np.nanmin(da),3),'max:',np.round(np.nanmax(da),3),
              'mean:',np.round(np.nanmean(da),3),)
    #% % Plot residual clusters
    #% % make list with datasets
    das_res_u_map = [dic['res']['unc'],
               dic['dmap']['res']['unc'] * mask_dmap,
               dic['som']['res']['unc'],
                    ]
    
    for da,title in zip(das_res_u_map,titles):
        print(title,'min:',np.round(np.nanmin(da),3),
              'max:',np.round(np.nanmax(da),3),
              'mean:',np.round(np.nanmean(da),3),)
        
    #####################
    #####################
    ## Scatter plots
    #####################
    #####################
    labels = titles
    inds = [1,3,5]
    cmin=-5
    cmax=10
    clim = [cmin,cmax]
    j=0
    for i,label in zip(inds,labels):
        ax = plt.subplot(gs[i])
        key ='som' # cluster
        mask = np.array(dic[key]['mask']) # clusters mask
        mask_tmp = np.array(mask)
        mask_tmp[np.isfinite(mask_tmp)]=1
        if label =='1 degree':
            y = np.array(dic['alt']['trend'] * mask_tmp).flatten() 
            x = np.array(dic['sum']['trend'] * mask_tmp).flatten()
            yerr = np.array(dic['alt']['unc'] * mask_tmp).flatten() 
            xerr = np.array(dic['sum']['unc'] * mask_tmp).flatten()
        elif label=='SOM':
            key = 'som'
            y = np.array(dic[key]['df']['alt_tr']).flatten()
            x = np.array(dic[key]['df']['sum_tr']).flatten()
            yerr = np.array(dic[key]['df']['alt_unc']).flatten() 
            xerr = np.array(dic[key]['df']['sum_unc']).flatten()
        else:
            key = 'dmap'
            y = np.array(dic[key]['df']['alt_tr']).flatten()
            x = np.array(dic[key]['df']['sum_tr']).flatten()
            yerr = np.array(dic[key]['df']['alt_unc']).flatten() 
            xerr = np.array(dic[key]['df']['sum_unc']).flatten()  
        
    
        plt.title('({}). {}'.format(letters[i],label),fontsize=fontsize)
    
        x = x[np.isfinite(y)]
        xerr = xerr[np.isfinite(y)]
        yerr = yerr[np.isfinite(y)]
        y = y[np.isfinite(y)]
    
        y = y[np.isfinite(x)]
        xerr = xerr[np.isfinite(x)]
        yerr = yerr[np.isfinite(x)]
        x = x[np.isfinite(x)]
    
    
        plt.plot(clim, clim, ls="--", c=".1")
        w=ws[j]/2
        j=j+1
        plt.plot([cmin-w,cmax-w], [cmin+w,cmax+w], ls="--", c="pink")
        plt.plot([cmin+w,cmax+w], [cmin-w,cmax-w], ls="--", c="pink")
    
        plt.errorbar(x,y,
                     yerr=yerr,
                     xerr = xerr,
                     #c=c,
                     # s=1,
                     alpha=0.1,
                     zorder=0,
                     capsize=3,capthick=2,ecolor='gray',lw=2,fmt='none')
        plt.scatter(x,y,
                    alpha=0.5,
                    marker='o',
                    # facecolors='none', edgecolors='b'
                   )
    
    
        plt.xlim(clim)
        plt.ylim(clim)
        # if len(x)>500:
        #     idx = random.sample(list(np.arange(0,len(x))),500)
        #     xx = x[idx]
        #     yy = y[idx]
        # else:
        #     xx=x
        #     yy=y
        xx=x
        yy=y
    
        #ax.annotate("$R^2$ = {:.2f}".format(metrics.r2_score(y-np.nanmean(y),x-np.nanmean(x))), (clim[0]+1,clim[1]-3))
        r,p = st.pearsonr(xx, yy)
        if p<0.05:
            if p<0.0001:
                ax.annotate("Pearsons r = {:.2f}**".format(r), (clim[0]+0.5,clim[1]-2),fontsize=textsize)
            else:
                 ax.annotate("Pearsons r = {:.2f}*".format(r), (clim[0],clim[1]-2),fontsize=textsize)  
        else:
             ax.annotate("Pearsons r = {:.2f}".format(r), (clim[0],clim[1]-2),fontsize=textsize)
        ax.annotate("RMSE = {:.2f}".format(np.sqrt(metrics.mean_squared_error(y,x))),(clim[0]+0.5,clim[1]-3.5),fontsize=textsize)
    
        plt.xlabel('', fontsize=labelsize)
        plt.ylabel('Altimetry \n(mm/yr)', fontsize=labelsize)
        if i==5:
            plt.xlabel('Sum of Components \n(mm/yr)', fontsize=labelsize)
    
    #####################
    ######################
    ### Rsidual maps
    #####################
    #####################
    clim=1
    dataset = das_res_map
    cmap=cmap_trend
    cmin=-clim;cmax=clim
    titles=titles
    clabel='Budget Residual (mm/yr)'
    # offset_y = -0.2
    
    
    plot_type = 'pcolor'
    proj='robin'
    #fig = plt.figure(figsize=fsize,dpi=100)
    inds = [0,2,4]
    plt.rc('text', usetex=True)
    plt.rc('font', family=settings['font'])
    
    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=settings['lon0'])
    else:
        proj=ccrs.PlateCarree()
    for idata,data in enumerate(dataset):
        ax = plt.subplot(gs[inds[idata]], projection=proj
                         #Mercator()
                         )
        #ax.background_img(name='pop', resolution='high')
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
            lv=np.arange(cmin,cmax+interval,interval)
            mm=plt.contourf(lon,lat,data,levels=lv,
                      transform = ccrs.PlateCarree(),cmap=cmap)
    
            plt.pcolormesh(lon,lat,data,
                    vmin=cmin,vmax=cmax,
                    zorder=0,
                    transform = ccrs.PlateCarree(),cmap=cmap)
    
        
        
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
        # d01 box
        
        title = '({}). '.format(letters[idata+idata])+str(titles[idata])
        plt.title(title,fontsize=fontsize)
        if titles[idata] =='1 degree':
            A = np.array(dic['alt']['trend']).flatten()
            A_error = np.array(dic['alt']['unc']).flatten()
            B = np.array(dic['sum']['trend']).flatten()
            B_error = np.array(dic['sum']['unc']).flatten()
            A[np.isnan(B)]=np.nan
            B[np.isnan(A)]=np.nan
            A_error[np.isnan(B)]=np.nan
            B_error[np.isnan(A)]=np.nan
            # A = A[np.isfinite(A)]
            # A_error = A_error[np.isfinite(A_error)]
            # B = B[np.isfinite(B)]
            # B_error = B_error[np.isfinite(B_error)]
            
            n_total = len(A[np.isfinite(A)])
            
            C = np.array([compare_values(A[i],A_error[i],B[i],B_error[i]) 
                          for i in range(len(A))])
            n_closed = len(C[C==1])
            _,area =  get_area(lat,lon,A.reshape(180,360))
            area_closed = np.nansum(area*C.reshape(180,360))
            mask = np.array(A.reshape(180,360))
            mask[np.isfinite(mask)]=1
            total_area = np.nansum(area*mask)
            perc_area_closed = (area_closed*100)/total_area
            perc_n_closed = (n_closed*100)/n_total
            
            if info:
                print('1degree has a total area of {}'.format(total_area))
                print('and {} is closed within uncertainties'.format(area_closed))
                print('That is, {} % of the ocean area is closed'.format(perc_area_closed))
                print('1degree has {} grid cells'.format(n_total))
                print('and {} is closed within uncertainties'.format(n_closed))
                print('That is, {} % of the ocean area is closed'.format(perc_n_closed))
                
            
    
    
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()
    y2x_ratio = (ymax-ymin)/(xmax-xmin) * nrow/ncol
    # x2y_ratio = (xmax-xmin)/(ymax-ymin) * nrow/ncol
    
    # Apply new h/w aspect ratio by changing h
    # Also possible to change w using set_figwidth()
    fig.set_figheight(wi * y2x_ratio)
    # fig.set_figwidth(15)
    
    plt.tight_layout()
    # # fig.subplots_adjust(aright=0.8)
    cbar_ax2 = fig.add_axes([0.1, 0.06, 0.25, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=ticksize, family=settings['font'])    
    cbar2.ax.tick_params(labelsize=labelsize) 

    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

def make_figure_hist_desc(save=True,
                path_to_figures = path_figures,
                figname = 'budget_res_histogram_desc',
                figfmt='png'
                ):
    # plot trends for each component
    clim=5
    dic = load_data()
    das_res = [
           dic['som']['df']['res_tr'],
           dic['dmap']['df']['res_tr'],
           dic['res']['trend'],
                ]
    das_alt = [
               dic['som']['df']['alt_tr'],
               dic['dmap']['df']['alt_tr'],
               dic['alt']['trend'],
                    ]
    das_sum = [dic['som']['df']['sum_tr'],
               dic['dmap']['df']['sum_tr'],
               dic['sum']['trend'],
                    ]
    
    das_res_u = [dic['som']['df']['res_unc'],
               dic['dmap']['df']['res_unc'],
               dic['res']['unc'],
                    ]
    
    # das_sum_u = [dic['sum']['unc'],
    #            dic['dmap']['df']['sum_unc'],
    #            dic['som']['df']['sum_unc'],
    #                 ]
    
    # das_alt_u = [dic['alt']['unc'],
    #            dic['dmap']['df']['alt_unc'],
    #            dic['som']['df']['alt_unc'],
    #                 ]
    global settings
    settings = set_settings()
    colors_dic = settings['colors_dic']
    letters = settings['letters']
    
    i=0
    ws = []
    clim=7.5
    titles=[ 'SOM', '$\delta$-MAPS','1 degree']
    
    pos=[1,3,5]
    lims = [(0,11000),(0,45),(0,18)]
    fig = plt.figure(figsize=(10,15))
    for i in range(3):
        df = pd.DataFrame({'Altimetry':np.hstack(das_alt[i]),
                           'Sum':np.hstack(das_sum[i]),
                          # 'res':np.hstack(adas_res[i])
                          })
    
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df = df.drop('index',axis=1)
    
        ax = plt.subplot(3,2,pos[i])
        for name in ['Altimetry','Sum']:
            sns.histplot(df[name],kde=False, #label='Trend',
                         alpha=0.5,
                         label=name,
                         # stat="percent", 
                         color=colors_dic[name],
                        bins=np.arange(-clim, clim+0.1,0.5))
        
        if i==2:
            plt.xlabel('mm/yr')
        else:
            plt.xlabel('')
        plt.ylabel('Number regions',fontsize=settings['labelsize'])
        # plt.title(title[i])
        title = '({}). '.format(letters[i+i])+str(titles[i])
        plt.title(title,fontsize=settings['titlesize'])
        
        plt.ylim(lims[i])
        ax.tick_params(axis='both', which='major', 
                       labelsize=settings['ticklabelsize'])
        
        plt.legend(loc='upper left',fontsize=settings['legendsize'])
    i=0
    pos=[2,4,6]
    for trend,unc in zip(das_res,das_res_u):
        df= pd.DataFrame( {
                        'Trend':np.hstack(trend),
        'Unc':np.hstack(unc),
        })
        
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df = df.drop('index',axis=1)
        
        ax = plt.subplot(3,2,pos[i])
        c = {'Trend':'darkgray',
            'Unc':'lightpink'}
        for var in ['Trend','Unc']:
            sns.histplot(df[var],kde=False, #label='Trend',
                         alpha=0.5,
                         color=c[var],
                         label=var,
                         # stat="percent", 
                        bins=np.arange(-clim, clim+0.1,0.5))
    
        # plt.legend(prop={'size': 12})
        if i==2:
            plt.xlabel('mm/yr')
        else:
            plt.xlabel('')
        plt.ylabel('')
        plt.xlim(-clim,clim)
        
        plt.ylim(lims[i])
        var = 'Unc'
        x = np.array(df[var])
        ci_level=0.95
        ci = st.norm.interval(alpha=ci_level, loc=np.mean(x), scale=x.std())
        plt.axvline(x.mean(),color='k',linestyle='--')
    
        plt.axvline(ci[0],c=c[var],linestyle='--',alpha=1,label='{}% CI'.format(ci_level*100))
        plt.axvline(ci[1],c=c[var],linestyle='--',alpha=0.5)
        
        ci_width = np.abs(ci[0]-ci[1])/2
        ws.append(ci_width)
        # plt.title(title[i]+': {:.3f}'.format( ci_width))
        title = '({}). '.format(letters[i+i+1])+str(titles[i]+': {:.3f}'.format( ci_width))
        plt.title(title,fontsize=settings['titlesize'])
        
        i=i+1
        ax.tick_params(axis='both', which='major', labelsize=settings['ticklabelsize'])

        plt.legend(loc='upper left',fontsize=settings['legendsize'])
     
    #plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()


    return ws
def make_figure_scatter_desc(save=True,
                path_to_figures = path_figures,
                figname = 'residuals_map_scatter-desc',
                figfmt='png'
                ):
    
    ws = make_figure_hist(save=False)
    dic = load_data()
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    interval = 0.1
    global settings
    settings = set_settings()
    # colors_dic = settings['colors_dic']
    letters = settings['letters']
    
    wi=10
    hi=15
    fig = plt.figure(figsize=(wi,hi),dpi=settings['dpi'])
    nrow=3;ncol=2
    gs = GridSpec(nrow, ncol)
    textsize = settings['textsize']
    labelsize=settings['labelsize']
    fontsize=settings['titlesize']
    ticksize=settings['ticksize']
    df_dmap, mask_dmap, _ = prep_dmaps(info=False)
    
    #% %  Plot residual clusters
    #% % make list with datasets
    das_res_map = [
               dic['som']['res']['trend'],
               dic['dmap']['res']['trend'] * mask_dmap,
               dic['res']['trend'],
                    ]
    
    titles = [ r"SOM residuals ",
     r"$\delta$-MAPS residuals ", 
              "1 degree residuals",
                        # r"unc", r"unc",                     
              ]
    for da,title in zip(das_res_map,titles):
        print(title,'min:',np.round(np.nanmin(da),3),'max:',np.round(np.nanmax(da),3),
              'mean:',np.round(np.nanmean(da),3),)
    #% % Plot residual clusters
    #% % make list with datasets
    das_res_u_map = [
               dic['som']['res']['unc'],
               dic['dmap']['res']['unc'] * mask_dmap,
               dic['res']['unc'],
                    ]
    
    for da,title in zip(das_res_u_map,titles):
        print(title,'min:',np.round(np.nanmin(da),3),
              'max:',np.round(np.nanmax(da),3),
              'mean:',np.round(np.nanmean(da),3),)
        
    #####################
    #####################
    ## Scatter plots
    #####################
    #####################
    labels = [ 'SOM', 'dmap','1 degree']
    inds = [1,3,5]
    cmin=-5
    cmax=10
    clim = [cmin,cmax]
    j=0
    for i,label in zip(inds,labels):
        ax = plt.subplot(gs[i])
        key ='som' # cluster
        mask = np.array(dic[key]['mask']) # clusters mask
        mask_tmp = np.array(mask)
        mask_tmp[np.isfinite(mask_tmp)]=1
        if label =='1 degree':
            y = np.array(dic['alt']['trend'] * mask_tmp).flatten() 
            x = np.array(dic['sum']['trend'] * mask_tmp).flatten()
            yerr = np.array(dic['alt']['unc'] * mask_tmp).flatten() 
            xerr = np.array(dic['sum']['unc'] * mask_tmp).flatten()
        elif label=='SOM':
            key = 'som'
            y = np.array(dic[key]['df']['alt_tr']).flatten()
            x = np.array(dic[key]['df']['sum_tr']).flatten()
            yerr = np.array(dic[key]['df']['alt_unc']).flatten() 
            xerr = np.array(dic[key]['df']['sum_unc']).flatten()
        else:
            key = 'dmap'
            y = np.array(dic[key]['df']['alt_tr']).flatten()
            x = np.array(dic[key]['df']['sum_tr']).flatten()
            yerr = np.array(dic[key]['df']['alt_unc']).flatten() 
            xerr = np.array(dic[key]['df']['sum_unc']).flatten()  
        
    
        plt.title('({}). {}'.format(letters[i],label),fontsize=fontsize)
    
        x = x[np.isfinite(y)]
        xerr = xerr[np.isfinite(y)]
        yerr = yerr[np.isfinite(y)]
        y = y[np.isfinite(y)]
    
        y = y[np.isfinite(x)]
        xerr = xerr[np.isfinite(x)]
        yerr = yerr[np.isfinite(x)]
        x = x[np.isfinite(x)]
    
    
        plt.plot(clim, clim, ls="--", c=".1")
        w=ws[j]/2
        j=j+1
        plt.plot([cmin-w,cmax-w], [cmin+w,cmax+w], ls="--", c="pink")
        plt.plot([cmin+w,cmax+w], [cmin-w,cmax-w], ls="--", c="pink")
    
        plt.errorbar(x,y,
                     yerr=yerr,
                     xerr = xerr,
                     #c=c,
                     # s=1,
                     alpha=0.1,
                     zorder=0,
                     capsize=3,capthick=2,ecolor='gray',lw=2,fmt='none')
        plt.scatter(x,y,
                    alpha=0.5,
                    marker='o',
                    # facecolors='none', edgecolors='b'
                   )
    
    
        plt.xlim(clim)
        plt.ylim(clim)
        if len(x)>500:
            idx = random.sample(list(np.arange(0,len(x))),500)
            xx = x[idx]
            yy = y[idx]
        else:
            xx=x
            yy=y
    
        #ax.annotate("$R^2$ = {:.2f}".format(metrics.r2_score(y-np.nanmean(y),x-np.nanmean(x))), (clim[0]+1,clim[1]-3))
        r,p = st.pearsonr(xx, yy)
        if p<0.05:
            if p<0.0001:
                ax.annotate("Pearsons r = {:.2f}**".format(r), (clim[0]+0.5,clim[1]-2),fontsize=textsize)
            else:
                 ax.annotate("Pearsons r = {:.2f}*".format(r), (clim[0],clim[1]-2),fontsize=textsize)  
        else:
             ax.annotate("Pearsons r = {:.2f}".format(r), (clim[0],clim[1]-2),fontsize=textsize)
        ax.annotate("RMSE = {:.2f}".format(np.sqrt(metrics.mean_squared_error(y,x))),(clim[0]+0.5,clim[1]-3.5),fontsize=textsize)
    
        plt.xlabel('', fontsize=labelsize)
        plt.ylabel('Altimetry \n(mm/yr)', fontsize=labelsize)
        if i==5:
            plt.xlabel('Sum of Componentns \n(mm/yr)', fontsize=labelsize)
    
    #####################
    ######################
    ### Rsidual maps
    #####################
    #####################
    clim=1
    dataset = das_res_map
    cmap=cmap_trend
    cmin=-clim;cmax=clim
    titles=titles
    clabel='mm/yr'
    # offset_y = -0.2
    
    
    plot_type = 'pcolor'
    proj='robin'
    #fig = plt.figure(figsize=fsize,dpi=100)
    inds = [0,2,4]
    plt.rc('text', usetex=True)
    plt.rc('font', family=settings['font'])
    
    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=settings['lon0'])
    else:
        proj=ccrs.PlateCarree()
    for idata,data in enumerate(dataset):
        ax = plt.subplot(gs[inds[idata]], projection=proj
                         #Mercator()
                         )
        #ax.background_img(name='pop', resolution='high')
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
            lv=np.arange(cmin,cmax+interval,interval)
            mm=plt.contourf(lon,lat,data,levels=lv,
                      transform = ccrs.PlateCarree(),cmap=cmap)
    
            plt.pcolormesh(lon,lat,data,
                    vmin=cmin,vmax=cmax,
                    zorder=0,
                    transform = ccrs.PlateCarree(),cmap=cmap)
    
        
        
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=settings['landcolor']))
    
        # d01 box
        
        title = '({}). '.format(letters[idata+idata])+str(titles[idata])
        plt.title(title,fontsize=fontsize)
    
    
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()
    y2x_ratio = (ymax-ymin)/(xmax-xmin) * nrow/ncol
    # x2y_ratio = (xmax-xmin)/(ymax-ymin) * nrow/ncol
    
    # Apply new h/w aspect ratio by changing h
    # Also possible to change w using set_figwidth()
    fig.set_figheight(wi * y2x_ratio)
    # fig.set_figwidth(15)
    
    plt.tight_layout()
    # # fig.subplots_adjust(aright=0.8)
    cbar_ax2 = fig.add_axes([0.1, 0.06, 0.25, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=ticksize, family=settings['font'])    
    cbar2.ax.tick_params(labelsize=labelsize) 

    #plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG S3 - residuals
def make_figure_residuals(save=True,
                path_to_figures = path_figures,
                figname = 'residuals_unc_distribution',
                figfmt='png'):
    
    dic = load_data()
    global settings
    settings = set_settings()
    # fsize = settings['fsize']
    
    clim=1
    fig = plt.figure(figsize=(15,10))
    ax = plt.subplot(211)
    key='som'
    # plt.xticks(data_x)    
    df= pd.DataFrame( {'unc':np.hstack(dic[key]['df']['res_unc']),
                        'trend':np.hstack(dic[key]['df']['res_tr']) })
    # N_cells = len(df)
    df.dropna(inplace=True)
    # N_cells_oceans = len(df)
    df.reset_index(inplace=True)
    
    data_x = df.index
    data_y = df.unc
    data_y_sc = np.array(df.trend)
    
    
    A = np.array(np.hstack(dic[key]['df']['alt_tr']))
    A_error = np.array(np.hstack(dic[key]['df']['alt_unc']))
    B = np.array(np.hstack(dic[key]['df']['sum_tr']))
    B_error = np.array(np.hstack(dic[key]['df']['sum_unc']))
    
    C = np.array([compare_values(A[i],A_error[i],B[i],B_error[i]) for i in range(len(A))])
    res_closed = len(C[C==1])
    res_max = len(data_y_sc)
    plt.ylabel("mm/yr")
    plt.title('(a). SOM Residuals: {} out of {} closed'.format(res_closed,res_max))
    
    # ax.set_xlim(np.nanmin(data_x),np.nanmax(data_x))
    # bars
    rects = ax.bar(data_x, data_y, 
                    color='None', 
                    # alpha=0,
                    edgecolor='black')
    #% %
    # scatter
    #% %
    sc=ax.scatter(data_x,np.abs(data_y_sc),
                    s=100,
                    c=data_y_sc,
                  cmap=cmap_trend,
                   vmin=-clim,vmax=clim)
    
    ax.set_xticks(data_x)
    ax.set_xticklabels(np.arange(1,data_x.max()+2))
    
    plt.xlim([data_x.min()-1,data_x.max()+1])
    
    
    
    ax = plt.subplot(212)
    # plt.xticks(data_x)    
    key='dmap'
    _,_,df3 = prep_dmaps(info=False)
    data_x = df3.index
    data_y = df3.res_unc
    data_y_sc = np.array(df3.res_tr)
    
    
    A = np.array(df3['Altimetry'])
    A_error = np.array(df3['Altimetry_err'])
    B = np.array(df3['Sum'])
    B_error = np.array(df3['Sum_err'])
    
    C = np.array([compare_values(A[i],A_error[i],B[i],B_error[i]) for i in range(len(A))])
    res_closed = len(C[C==1])
    res_max = len(data_y_sc)
    
    plt.ylabel("mm/yr")
    plt.title('(b). dMAPS Residuals: {} out of {} closed'.format(res_closed,res_max))
    
    # ax.set_xlim(np.nanmin(data_x),np.nanmax(data_x))
    # bars
    rects = ax.bar(data_x, data_y, 
                    color='None', 
                    alpha=0.5,
                    edgecolor='black')
    #% %
    # scatter
    #% %
    sc=ax.scatter(data_x,np.abs(data_y_sc),
                    s=100,
                    c=data_y_sc,
                  cmap=cmap_trend,
                   vmin=-clim,vmax=clim)
    x_tick = np.arange(0,100,10)
    x_tick[0] = 1
    ax.set_xticks(x_tick)
    # ax.set_xticklabels(np.arange(1,data_x.max()+2))
    
    
    
    plt.xlim([data_x.min()-1,data_x.max()+1])
    
    #plt.xlim([-1,94])
    plt.xlabel('Domain number')
    
    cbar_ax2 = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar2 = plt.colorbar(sc,cax=cbar_ax2)
    cbar2.set_label('Trend (mm/yr)',labelpad=15)
    
    
    
    
    
    # plt.savefig("bar_chart_with_colorbar_03.png", bbox_inches='tight')
    
    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    # plt.close()

    return

#%% FIG 4 - budget drivers
def make_figure_drivers(save=True,
                path_to_figures = path_figures,
                figname = 'budget_drivers',
                figfmt='png'
                ):
    
    settings = set_settings()
    df,df2 = prep_drivers()
    dic = load_data()
    fontsize=settings['titlesize']
    labelsize=settings['labelsize']
    legendsize = settings['legendsize']
    offset=0.15
    fig = plt.figure(figsize=(15,15),dpi=settings['dpi'])
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
    yerr=df['{}_err'.format(name)]
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
    yerr=df['{}_err'.format(name)]
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
    
    plt.xlabel('SOM Domain number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    plt.ylabel('mm/yr',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.text(17,6,'({})'.format(settings['letters'][0]),fontsize=settings['titlesize'])
    
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
    
    plt.xlabel('$\delta$-MAPS Domain number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    plt.ylabel('mm/yr',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.text(17,6,'({})'.format(settings['letters'][1]),
             fontsize=settings['titlesize'])
    
    
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
        
        plt.title('({}). Domains {} and {}'.format(settings['letters'][i+2],
                                                   icluster_som, n_dmap[i]),
                  fontsize=settings['titlesize'])
        plt.ylim(-200,200)
        if icluster_som ==1:
            # plt.legend()
            plt.ylabel('mm',fontsize=settings['titlesize'])
    
        i=i+1
        plt.xlabel('Time',fontsize=settings['titlesize'])
        plt.legend(fontsize=legendsize,loc = 'lower left')
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        
    plt.tight_layout()
    
    
    ######
   #  plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%%
def make_figure_drivers_v2(save=True,
                path_to_figures = path_figures,
                figname = 'budget_drivers',
                figfmt='png'
                ):
    
    settings = set_settings()
    df,df2 = prep_drivers()
    dic = load_data()
    fontsize=settings['titlesize']
    labelsize=settings['labelsize']
    legendsize = settings['legendsize']
    offset=0.15
    fig = plt.figure(figsize=(15,20),dpi=settings['dpi'])
    nrow=8;ncol=2
    colors_dic = settings['colors_dic']
    ymin=-0.5;ymax=7.8
    xtext = 16
    ytext=ymax-1.5
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
    yerr=df['{}_err'.format(name)]
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
    yerr=df['{}_err'.format(name)]
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
    # plt.legend(fontsize=legendsize,loc = 'upper left')
    
    plt.xlabel('SOM Domain number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    plt.ylabel('mm/yr',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.ylim(ymin,ymax)
    plt.text(xtext,ytext,'({})'.format(settings['letters'][0]),fontsize=settings['titlesize'])
    ax.legend(bbox_to_anchor=(0.25, -0.2),
              ncol=2,
              fontsize=legendsize,)
    
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
            legend=False
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
                # label=name
                )
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
                color=colors_dic[name],
                # label=name
                )
    plt.errorbar(x+offset,y,
                 yerr=yerr,
                 alpha=0.5,
                 capsize=3,capthick=2,
                 ecolor=colors_dic[name],
                 #lw=2,
                 zorder=1,
                 fmt='none')
    # plt.legend(fontsize=legendsize,loc = 'upper left')
    
    plt.xlabel('$\delta$-MAPS Domain number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    plt.ylabel('mm/yr',fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.ylim(ymin,ymax)
    plt.text(xtext,ytext,'({})'.format(settings['letters'][1]),
             fontsize=settings['titlesize'])

    
    
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
    j=0
    n_dmap = [82,45]
    for icluster_som, icluster_dmap in zip([1,12],[87,49]):
        ax = plt.subplot2grid((nrow,ncol),(4+j,0),rowspan=2,colspan=2)
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
                        # label=' d-MAPS'
                    )
        
        plt.title('({}). Domains {} and {}'.format(settings['letters'][i+2],
                                                   icluster_som, n_dmap[i]),
                  fontsize=settings['titlesize'])
        plt.ylim(-200,200)
        if icluster_som ==1:
            # plt.legend()
            plt.ylabel('mm',fontsize=settings['titlesize'])
            ax.legend(bbox_to_anchor=(0.25, -0.2),
                      ncol=2,
                      fontsize=legendsize,)
            
    
        i=i+1
        j=j+2
        plt.xlabel('Time',fontsize=settings['titlesize'])
        # plt.legend(fontsize=legendsize,loc = 'lower left',ncol=2)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        
    plt.tight_layout()
    
    
    ######
   #  plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return
#%%
## dmaps drivers - all clusters
def make_figure_drivers_dmaps(save = True, 
                   path_to_figures = path_figures,
                   figname = 'budget_drivers_dmaps',
                   figfmt='png'):
    # legendsize = 13
    df_dmap, mask_dmap, df3 = prep_dmaps(info=False)
    settings = set_settings()
    fontsize=settings['titlesize']
    labelsize=settings['labelsize']
    colors_dic = settings['colors_dic']
    fig = plt.figure(figsize=(15,20),dpi=settings['dpi'])
    
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
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.xticks(rotation = 0)
    # plt.legend(fontsize=legendsize,loc = 'upper left')
    
    plt.xlabel('Dmaps Domain number',fontsize=fontsize,)
    plt.xticks(rotation = 0) 
    
    
    # ax.legend(title='SUBJECT',title_fontsize=30,loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(bbox_to_anchor =(0.1,-0.35), loc='lower center', ncol=2)
    
    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

## all time series
def make_figure_all_ts(save=True,
                path_to_figures = path_figures,
                figname = 'budget_timeseries',
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
    
    fig = plt.figure(figsize=(20,20),dpi=settings['dpi'])
    for i in range(n_cluster):
        icluster = i+1
        ax = plt.subplot(y,x,icluster)
    
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
            
        
        plt.title('({}). Domains {};{}'.format(
            letters[i],icluster,df3[df3['cluster_n']==sel[i]].index[0]),
            fontsize=settings['titlesize'])
        plt.ylim(-200,200)
        if icluster ==18:
            plt.legend(fontsize=settings['legendsize'])
        if icluster >15:
            plt.xlabel('Time',fontsize=settings['labelsize'])
        if i%3==0:
            plt.ylabel('mm',fontsize=settings['labelsize'])
        ax.tick_params(axis='both', which='major', labelsize=settings['ticklabelsize'])
        
    plt.tight_layout()
    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG 5
def make_figure_stats(save=True,
                path_to_figures = path_figures,
                figname = 'budget_stats',
                figfmt='png'):
    settings = set_settings()
    # path = '/Volumes/LaCie_NIOZ/data/budget/'
    # path = '/Users/ccamargo/Desktop/manuscript_SLB/data/'
    #path = '/Volumes/LaCie_NIOZ/data/budget/'
    df = pd.read_pickle(path_to_data+"budget-stats.p")
    df = df.drop(df[df.comb == '(alt,sum)'].index)
    df['budget_name'] = ['1 degree' if label=='1degree'  
                         else 'SOM' if label=='som'  
                         else '$\delta$-MAPS' 
                         for label in df['budget'] ]
    df['comb'] = [combo.replace('alt,','') 
                  for combo in df['comb']]
    df['combo'] = [combo.replace('bar','GRD') 
               # if 'bar' in combo 
               # else combo
               for combo in df['comb']]
    # plot grouped boxplot
    varis = ['r','nRMSE']
    ncol=len(varis)
    nrow = 1
    fig = plt.figure(figsize=(10,10))
    for i,var in enumerate(varis):
        ax = plt.subplot(ncol,nrow,i+1)
        sns.boxplot(x = df['combo'],
                    y = df[var],
                    hue = df['budget_name'],
                    showfliers = False,
                    palette = 'Set2')
        
        if var=='R2':
            plt.ylim(-2,2)
        if var =='r':
            plt.ylim(-1,1)
        if var=='nRMSE':
            plt.ylim(0,0.3)
        plt.ylabel(var,fontsize=settings['labelsize'])
        plt.xlabel('')
        if i==0:
            plt.legend(loc='lower left',fontsize=settings['textsize'])
            plt.text(6.25,-0.9,'({})'.format(settings['letters'][i]),
                     fontsize=settings['textsize'],fontweight="bold")
        else: 
            plt.legend(loc='upper left',fontsize=settings['textsize'])
            plt.text(6.25,0.28,'({})'.format(settings['letters'][i]),
                     fontsize=settings['textsize'],fontweight="bold")   
        ax.tick_params(axis='both', which='major', labelsize=settings['labelsize']-4)
        plt.xticks(rotation = 15)
    plt.tight_layout()
#     plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG SI - stats 2x2 and 5x5
def make_figure_stats_SI(save=True,
                path_to_figures = path_figures,
                figname = 'stats_SI',
                figfmt='png'
                ):
    #% %

    settings = set_settings()
    letters = settings['letters']
    path = path_to_data
    df =  pd.read_pickle(path+"budget-stats_complete.p")
    # plot grouped boxplot
    # df = df.sort_values(by=['budget_name'])
    varis = ['r','nRMSE']
    df['comb'] = [combo.replace('alt,','') 
                  for combo in df['comb']]
    ncol=len(varis)
    nrow = 1
    textsize = 13
    fig = plt.figure(figsize=(10,10))
    for i,var in enumerate(varis):
        plt.subplot(ncol,nrow,i+1)
        sns.boxplot(x = df['comb'],
                    y = df[var],
                    hue = df['budget_name'],
                    showfliers = False,
                    palette = 'Set2')
        
        if var=='R2':
            plt.ylim(-2,2)
        if var =='r':
            plt.ylim(-1,1)
        if var=='nRMSE':
            plt.ylim(0,0.3)
        plt.ylabel(var,fontsize=settings['labelsize'])
        plt.xlabel('')
        if i==0:
            plt.legend(loc='lower left')
            plt.text(6.25,-0.9,'({})'.format(letters[i]),fontsize=textsize,fontweight="bold")
        else: 
            plt.legend(loc='upper left')
            plt.text(6.25,0.28,'({})'.format(letters[i]),fontsize=textsize,fontweight="bold")   
        plt.xticks(rotation = 15)
    plt.tight_layout()
    # plt.show()
    
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG 6 
def make_figure_combs(save=True,
                path_to_figures = path_figures,
                figname = 'budget_combination',
                figfmt='png'
                ):
    dic = load_data()
    #% % sensitivity to budget combination 
    da = xr.open_dataset(path_to_data+'combinations.nc')
    budget = 'som'
    mask_clusters = dic[budget]['mask']
    n_clusters = dic[budget]['n']
    
    n_pos = len(da.comb)
    cluster_combos_res = np.zeros((n_clusters,n_pos))
    cluster_combos_unc = np.zeros((n_clusters,n_pos))
    cluster_combos_sig = np.zeros((n_clusters,n_pos))
    
    for ipos in range(n_pos):
        res=np.array(da.res[ipos])
        unc = np.array(da.unc[ipos])
        mat = np.zeros((n_clusters))
        mat2 = np.zeros((n_clusters,))
        # mat3 = np.zeros((n_clusters))
        
        for i in range(n_clusters):
            icluster = i+1
            mask=np.array(mask_clusters)
            mask[np.where(mask!=icluster)]=np.nan
            mask[np.isfinite(mask)]=1
            
            # tmp[i,mask==1] = cluster_mean(np.array(dic[label]['trend']),mask, lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
            # tmp2[i,mask==1] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
            mat[i] = cluster_mean(res,mask,lat=np.array(da.lat),lon=np.array(da.lon),norm=False )
            mat2[i] = cluster_mean(unc,mask,lat=np.array(da.lat),lon=np.array(da.lon),norm=False )
            # mat3[i] = cluster_mean(np.array( np.abs(dic['alt']['unc'] - sum_comps_unc) ),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
            cluster_combos_res[i,ipos] = mat[i]
            cluster_combos_unc[i,ipos] = mat2[i]
            if mat[i]<mat2[i]:
                cluster_combos_sig[i,ipos] = 0 # zero if closed
            else:
                cluster_combos_sig[i,ipos] = 1 # 1 if open
    
        #% % open best dataset
    # pwd ="/Volumes/LaCie_NIOZ/data/budget/"
    dicb = pd.read_pickle(path_to_data+"best_rmse.pkl")
    print(dicb)
    variables = np.array(da['dataset'])
    n = []
    for var in variables:
        if var=='ste': var='steric'
        if var=='bar': var='barystatic'
        if var=='dyn': var = 'dynamic'
        
        if var=='dynamic':
            n.append('ENS') # we only have trends for ENS for now
        elif var=='steric':
            n.append(dicb[var].upper())
        else:
            n.append(dicb[var])
    
    names = np.array(da.names)

    #% %
    
    fontsize=settings['titlesize']
    size=75
    y = cluster_combos_res.flatten()
    y_sig = cluster_combos_sig.flatten()
    x = range(len(y))
    col= ['salmon' if sig==1 else 'mediumslateblue' for sig in y_sig]
    #% %
    fig = plt.figure(dpi=settings['dpi'],figsize=(15,10))
    ax=plt.subplot(111)
    # plot vertical lines to separate clusters
    xs = np.linspace(1,len(x),19)
    for xc in xs:
        plt.axvline(x=xc, color='gray', linestyle='--',alpha=0.5)
    # scatter all options:
    alpha=0.1
    plt.scatter(x,y,c=col,alpha=alpha)
    p1 = round(len(y_sig[y_sig==0])*100/len(y_sig))
    p2 = round(len(y_sig[y_sig==1])*100/len(y_sig))
    print('{}% closes, {}% doesnt'.format(p1,p2))
    
    # plot for legend:
    ind=np.where(y_sig==1)[0][0]
    plt.scatter(x[ind],y[ind],
                c=col[ind],
                alpha=alpha,label='open')
    ind=np.where(y_sig==0)[0][0]
    plt.scatter(x[ind],y[ind],
                c=col[ind],
                alpha=alpha,label='closed')
    
    # plot ensemble
    col= ['salmon' if sig==1 else 'blue' for sig in y_sig]
    
    ipos = [i for i in range(len(da.comb)) 
            if np.all(names[i,:]==['ENS'])][0]
    ipos = [i for i in range(len(da.comb)) 
            if np.all(names[i]==['ENS','ENS','IMB_WGP','ENS'])][0]
    y2 = np.full_like(y,np.nan)
    y2[ipos:len(y):n_pos] = y[ipos:len(y):n_pos]
    y2_sig = np.full_like(y,np.nan)
    y2_sig[ipos:len(y):n_pos] = y_sig[ipos:len(y):n_pos]
    
    # col= ['salmon' if sig==1 else 'mediumslateblue' for sig in cluster_combos_sig[:,ipos]]
    plt.scatter(x,y2,
                # c='black',
                marker='s',
                c=col,
                s=size,
                # label='ENS'
                )
    if np.any(y_sig[ipos:len(y):n_pos]==1):
        ind=np.where(y2_sig==0)[0][0]
        plt.scatter(x[ind],y[ind],
                    marker='s',
                    c=col[ind],
                    # alpha=alpha,
                    label='ENS ')
    if np.any(y_sig[ipos:len(y):n_pos]==0):
        ind=np.where(y2_sig==0)[0][0]
        plt.scatter(x[ind],y[ind],
                    marker='s',
                    
                    c=col[ind],
                    # alpha=alpha,
                    label='ENS ')
    
    # plot best rmse commbination
    col= ['salmon' if sig==1 else 'darkviolet' for sig in y_sig]
    
    ipos = [i for i in range(len(da.comb)) 
            if np.all(names[i]==n)][0]
    y2 = np.full_like(y,np.nan)
    y2[ipos:len(y):n_pos] = y[ipos:len(y):n_pos]
    y2_sig = np.full_like(y,np.nan)
    y2_sig[ipos:len(y):n_pos] = y_sig[ipos:len(y):n_pos]
    
    # col= ['salmon' if sig==1 else 'mediumslateblue' for sig in cluster_combos_sig[:,ipos]]
    plt.scatter(x,y2,
                # c='black',
                marker='^',
                c=col,
                s=size*2,
                # label='ENS'
                )
    if np.any(y_sig[ipos:len(y):n_pos]==1):
        ind=np.where(y2_sig==1)[0][0]
        plt.scatter(x[ind],y[ind],
                    marker='s',
                    c=col[ind],
                    # alpha=alpha,
                    label='RMSE ')
    if np.any(y_sig[ipos:len(y):n_pos]==0):
        ind=np.where(y2_sig==0)[0][0]
        plt.scatter(x[ind],y[ind],
                    marker='^',
                    
                    c=col[ind],
                    # alpha=alpha,
                    label='RMSE ')
    plt.ylabel('mm/yr',fontsize=settings['labelsize'])
    
    ax.set_xticks(xs[0:len(xs)-1]+200)
    ax.set_xticklabels(np.arange(1,19))
    ax.tick_params(axis='both', which='major', labelsize=settings['ticklabelsize'])
    
    plt.axhline(y = 0, color = 'gray', linestyle = '--')
    
    plt.xlabel('SOM domain number',fontsize=settings['labelsize'])
    
    plt.legend(fontsize=settings['labelsize'])
    # plt.show()


    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% FIG SI - dynamic SL
def make_figure_dyn(save=True,
                path_to_figures = path_figures,
                figname = 'dynSLs',
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
    labels= ['GRACE ENS','REANALYSIS']

    # plot
    dpi=settings['dpi']
    wi=15;hi=8
    # dimlat=180;dimlon=360
    # fontsize=25
    # ticksize=20
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

    labels= ['GRACE AVG','Reanalysis', 'GRACE JPL','GRACE CSR']
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
        
        plt.title('({}).{}'.format(letters[i],name),size=settings['titlesize'])
        
    
        
    plt.tight_layout()
    cbar_ax2 = fig.add_axes([0.13, -0.15, 0.75, 0.05])
    cbar2 = plt.colorbar(csf, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label(label='Trends \n {}-{} (mm/yr)'.format(t0,t1),
                    size=settings['labelsize'], family=settings['font'])
    cbar2.ax.tick_params(labelsize=settings['ticksize']) 
    
    # plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')

    plt.close()

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
    
   #  plt.show()

    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()


    return


#%% FIG SI - SOM 4x4 and 9x9
def make_figure_SOM(save=True,
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
        
        if n==9:
            cmap=generate_colormap()
        # plt.figure()
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
            
        plt.title('SOM {}x{}'.format(n,n),fontsize=settings['titlesize'])
    
        cbar = plt.colorbar(cs1,orientation='vertical')
        cbar.set_label(label='Domains',size=settings['labelsize'], 
                       family=settings['font'])
   
    
   #  plt.show()
    
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

    return

#%% REBUTTLE
# path = '/Volumes/LaCie_NIOZ/data/budget/'
def plot_open_regions(save=True,
                      path_to_figures = path_figures,
                      figname = 'open_dmaps',
                      figfmt='png'):
    settings = set_settings()
    dic = load_data()
    lat = dic['dims']['lat']['lat']
    lon = dic['dims']['lon']['lon']
    
    key = 'dmap'
    # df = dic[key]['df']
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
    # clim=1
    cmap='tab20'
    # clabel='Budget Residual (mm/yr)'
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

#% %
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

def plot_residual_size(save=True,
                      path_to_figures = path_figures,
                      figname = 'resXsize',
                      figfmt='png'):
    # path = '/Volumes/LaCie_NIOZ/data/budget/'
    dic = load_data()
    settings = set_settings()
    lat = np.array(dic['dims']['lat']['lat'])
    lon = np.array(dic['dims']['lon']['lon'])
    llon,llat = np.meshgrid(lon,lat)
    # lonv = llon.flatten()
    # latv = llon.flatten()
    dimlat = len(lat)
    dimlon = len(lon)
    #% % SOM & dmaps
    for j,key in enumerate(['som','dmap']):
        # key = 'som'
        # res = dic[key]['res']['trend'] # cluster residual trend
        # unc = dic[key]['res']['unc'] # cluster uncertainty
        mask = dic[key]['mask'] # clusters mask
        df = dic[key]['df']
        # get area grid
        surf, area = get_area(lat,lon,np.ones((dimlat,dimlon)) )
        
        cluster_area = np.zeros((len(df)))
        # central_lat = np.full_like(cluster_area,0)
        # central_lon = np.full_like(cluster_area,0)
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
    


    fig = plt.figure()
    x_name = 'size'
    xmin = min(dic['som']['df'][x_name].min(),dic['dmap']['df'][x_name].min(),)
    xmax = max(dic['som']['df'][x_name].max(),dic['dmap']['df'][x_name].max(),)

    y_name = 'res_tr'
    ymin = min(dic['som']['df'][y_name].min(),dic['dmap']['df'][y_name].min(),)
    ymax = max(dic['som']['df'][y_name].max(),dic['dmap']['df'][y_name].max(),)
    x_offset = 10**6
    y_offset = 0.5
    million = 10**6
    colors=['red','blue']
    labels={'dmap': '$\delta$-MAPS',
            'som':'SOM'}
    for j,key in enumerate(['dmap','som',]):
        df = dic[key]['df']
        
        threshold = 1
        y = np.array(df[y_name])
        n = len(y)
        y[(y>threshold) | (y<-threshold)]
        n_out = len(y[(y>threshold) | (y<-threshold)])
        n_in = n-n_out
        print('\n{}:'.format(key))
        print('Out of {}, {} are within ±{} and {} are not'.format(
            n,n_in,threshold,n_out))
        print('in %: {:.1f}% in and {:.1f}% out'.format((n_in*100)/n, (n_out*100)/n))
        # plt.subplot(1,2,int(j+1))
        plt.scatter(df[x_name]/million,df[y_name],
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
    plt.xlabel('Area (million $km^2$)')
    plt.ylabel('Residual trend \n (mm/yr)')
    # plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()

def plot_residual_unc_size(save=True,
                      path_to_figures = path_figures,
                      figname = 'resuncXsize',
                      figfmt='png'):
    # path = '/Volumes/LaCie_NIOZ/data/budget/'
    dic = load_data()
    settings = set_settings()
    lat = np.array(dic['dims']['lat']['lat'])
    lon = np.array(dic['dims']['lon']['lon'])
    llon,llat = np.meshgrid(lon,lat)
    lonv = llon.flatten()
    latv = llon.flatten()
    dimlat = len(lat)
    dimlon = len(lon)
    #% % SOM & dmaps
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
    
    #%  % 1 degree
    trend_1d  = np.array(dic['res']['trend'])
    unc_1d  = np.array(dic['res']['unc'])
    
    area = sl.get_grid_area(trend_1d)
    area=area.flatten()
    t1d = trend_1d.flatten()
    u1d = unc_1d.flatten()
    area=area[np.isfinite(t1d)]
    u1d=u1d[np.isfinite(u1d)]
    t1d=t1d[np.isfinite(t1d)]
    
#% %
    fig = plt.figure(figsize=(10,10))
    x_name = 'size'
    xmin = min(dic['som']['df'][x_name].min(),dic['dmap']['df'][x_name].min(),)
    xmax = max(dic['som']['df'][x_name].max(),dic['dmap']['df'][x_name].max(),)

    plt.subplot(211)
    y_name = 'res_tr'
    ymin = min(dic['som']['df'][y_name].min(),dic['dmap']['df'][y_name].min(),)
    ymax = max(dic['som']['df'][y_name].max(),dic['dmap']['df'][y_name].max(),)
    x_offset = 10**6
    y_offset = 0.5
    million = 10**6
    colors=['red','blue','green']
    labels={'dmap': '$\delta$-MAPS',
            'som':'SOM',
            '1d':'1 degree'}
    
    for j,key in enumerate(['dmap','som',
                            #'1d'
                            ]):
        if key=='1d':
            df = pd.DataFrame({x_name:area,
                              'res_tr':t1d,
                              'res_unc':u1d})
        else:
            df = dic[key]['df']
        # plt.subplot(1,2,int(j+1))
        plt.scatter(df[x_name]/million,df[y_name],
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
    plt.xlabel('Area (million $km^2$)')
    plt.ylabel('Residual trend \n (mm/yr)')
    
    plt.subplot(212)

    y_name = 'res_unc'
    ymin = min(dic['som']['df'][y_name].min(),dic['dmap']['df'][y_name].min(),)
    ymax = max(dic['som']['df'][y_name].max(),dic['dmap']['df'][y_name].max(),)
    x_offset = 10**6
    y_offset = 0.5
    million = 10**6
    # colors=['red','blue']
    # labels={'dmap': '$\delta$-MAPS',
    #         'som':'SOM'}
    for j,key in enumerate(['dmap','som',
                            # '1d'
                            ]):
        if key=='1d':
            df = pd.DataFrame({x_name:area,
                              'res_tr':t1d,
                              'res_unc':u1d})
        else:
            df = dic[key]['df']
        # df = dic[key]['df']
        # plt.subplot(1,2,int(j+1))
        plt.scatter(df[x_name]/million,df[y_name],
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
    plt.xlabel('Area (million $km^2$)')
    plt.ylabel('Residual uncertainty \n (mm/yr)')
    # plt.show()
    #%  %
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()
#%%
def plot_residual_opensize(save=True,
                      path_to_figures = path_figures,
                      figname = 'resXsize',
                      figfmt='png'):
    # path = '/Volumes/LaCie_NIOZ/data/budget/'
    
    dic = load_data()
    settings = set_settings()
    lat = np.array(dic['dims']['lat']['lat'])
    lon = np.array(dic['dims']['lon']['lon'])
    llon,llat = np.meshgrid(lon,lat)
    # lonv = llon.flatten()
    # latv = llon.flatten()
    dimlat = len(lat)
    dimlon = len(lon)
    #% %  SOM & dmaps
    for j,key in enumerate(['som','dmap']):
        # key = 'som'
        # res = dic[key]['res']['trend'] # cluster residual trend
        # unc = dic[key]['res']['unc'] # cluster uncertainty
        mask = dic[key]['mask'] # clusters mask
        df = dic[key]['df']
        # get area grid
        surf, area = get_area(lat,lon,np.ones((dimlat,dimlon)) )
        
        cluster_area = np.zeros((len(df)))
        # central_lat = np.full_like(cluster_area,0)
        # central_lon = np.full_like(cluster_area,0)
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
        df['marker']=['o' if s==1  else '*' for s in cluster_sig ]
        dic[key]['df'] = df
        
    


    fig = plt.figure()
    x_name = 'size'
    xmin = min(dic['som']['df'][x_name].min(),dic['dmap']['df'][x_name].min(),)
    xmax = max(dic['som']['df'][x_name].max(),dic['dmap']['df'][x_name].max(),)

    y_name = 'res_tr'
    ymin = min(dic['som']['df'][y_name].min(),dic['dmap']['df'][y_name].min(),)
    ymax = max(dic['som']['df'][y_name].max(),dic['dmap']['df'][y_name].max(),)
    x_offset = 10**6
    y_offset = 0.5
    million = 10**6
    colors=['red','blue']
    labels={'dmap': '$\delta$-MAPS',
            'som':'SOM'}
    #% %
    for j,key in enumerate(['dmap','som',]):
        df = dic[key]['df']
        
        threshold = 1
        y = np.array(df[y_name])
        n = len(y)
        y[(y>threshold) | (y<-threshold)]
        n_out = len(y[(y>threshold) | (y<-threshold)])
        n_in = n-n_out
        print('\n{}:'.format(key))
        print('Out of {}, {} are within ±{} and {} are not'.format(
            n,n_in,threshold,n_out))
        print('in %: {:.1f}% in and {:.1f}% out'.format((n_in*100)/n, (n_out*100)/n))
        # plt.subplot(1,2,int(j+1))
        # plt.scatter(df[x_name]/million,df[y_name],
        #             alpha=0.5,
        #             marker='o',
        #             color=colors[j],
        #             label=labels[key]
        #             # facecolors='none', edgecolors='b'
        #            )
        for i,row in df.iterrows():
            if compare_values(row['alt_tr'],row['alt_unc'],row['sum_tr'],row['sum_unc'])==1:
                
                f=colors[j]
                m='.'
                plt.scatter(row[x_name]/million,row[y_name],
                            alpha=0.5,
                            marker=m,
                            facecolors=f,
                            color=colors[j],
                            # label=labels[key]
                            label='{} - closed'.format(str(labels[key]))

                            # facecolors='none', edgecolors='b'
                           )
            else:
                f='none'
                m='*'
                # print(row[x_name]/million)
            # if i==0:
                plt.scatter(row[x_name]/million,row[y_name],
                            alpha=0.5,
                            marker=m,
                            facecolors=f,
                            color=colors[j],
                            label='{} - open'.format(str(labels[key]))
                            # facecolors='none', edgecolors='b'
                           )
            # else:
            #     plt.scatter(row[x_name]/million,row[y_name],
            #                 alpha=0.5,
            #                 marker=m,
            #                 facecolors=f,
            #                 color=colors[j],
            #                 # label=labels[key]
            #                 # facecolors='none', edgecolors='b'
            #                )
            
        # plt.title(key)
        #% %
        # plt.xlim(xmin-x_offset,xmax+x_offset)
        plt.ylim(ymin-y_offset,ymax+y_offset)
    # plt.legend()
    handles, labls = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labls, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel('Area (million $km^2$)')
    plt.ylabel('Residual trend \n (mm/yr)')
    # plt.show()
    #% %
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=settings['dpi'],bbox_inches='tight')
    plt.close()
#%% CALL ALL
# make_figure_trends(figname='fig1')
# make_figure_uncs(figname='figB1')
# # make_figure_gmsl()
# make_figure_clusters(figname='fig2')
# make_figure_hist(figname='figB2')
# make_figure_scatter(figname='fig3')
# make_figure_residuals(figname='figSI')
# # make_figure_drivers(figname='fig4')
# make_figure_drivers_dmaps(figname='figB3')
# make_figure_all_ts(figname='figB4')
# make_figure_stats(figname='fig5')
# make_figure_stats_SI(figname='figB6')
# make_figure_combs(figname='fig6')
# # make_figure_dyn(figname='figA1a')
# make_figure_dyn_dif(figname='figA1')
# make_figure_SOM(figname='figB7')

# # make_figure_hist_desc()
# # make_figure_scatter_desc()

# #% %  REBUTTAL
# plot_open_regions(figname='figR1')
# plot_residual_size(figname='figB5')
plot_residual_opensize(figname='figB5')

# # plot_residual_unc_size(figname='figR3')
# make_figure_drivers_v2(figname='fig4')



