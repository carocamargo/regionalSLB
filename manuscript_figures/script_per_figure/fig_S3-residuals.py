#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:04:26 2022

@author: ccamargo
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import cmocean as cm
import xarray as xr
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.cm import ScalarMappable
cmap_trend = cm.cm.balance
#%%
# make_figure(save=False)
path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'

def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'residuals_unc_distribution',
                figfmt='png'):
    
    dic = load_data(fmt = 'pkl')
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
    plt.xlabel('cluster number')
    
    cbar_ax2 = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar2 = plt.colorbar(sc,cax=cbar_ax2)
    cbar2.set_label('Trend (mm/yr)',labelpad=15)
    
    
    
    
    
    # plt.savefig("bar_chart_with_colorbar_03.png", bbox_inches='tight')
    
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

