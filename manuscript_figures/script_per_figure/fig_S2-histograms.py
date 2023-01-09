#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:39:14 2022

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
from matplotlib.gridspec import GridSpec
# from matplotlib.cm import ScalarMappable
cmap_trend = cm.cm.balance
cmap_unc = cm.tools.crop(cmap_trend,0,3,0)
# from matplotlib import cm as cmat
# import matplotlib.colors as col

import seaborn as sns
import scipy.stats as st
# from scipy import stats
import sklearn.metrics as metrics
import random

import warnings
warnings.filterwarnings("ignore","Mean of empty slice", RuntimeWarning)

import string

#%%
# make_figure(save=False)
path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'
def make_figure_SI(save=True,
                path_to_figures = path_figures,
                figname = 'budget_res_histogram',
                figfmt='png'
                ):
    # plot trends for each component
    clim=5
    dic = load_data(fmt = 'pkl')
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
        plt.ylabel('Number regions')
        # plt.title(title[i])
        title = '({}). '.format(letters[i+i])+str(titles[i])
        plt.title(title,fontsize=15)
        
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
        plt.title(title,fontsize=15)
        
        i=i+1
        plt.legend(loc='upper left')
    plt.show()   
    plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return ws
def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'residuals_map_scatter',
                figfmt='png'
                ):
    
    ws = make_figure_SI(save=False)
    dic = load_data(fmt = 'pkl')
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    interval = 0.1
    global settings
    settings = set_settings()
    # colors_dic = settings['colors_dic']
    letters = settings['letters']
    
    wi=10
    hi=15
    fig = plt.figure(figsize=(wi,hi),dpi=300)
    nrow=3;ncol=2
    gs = GridSpec(nrow, ncol)
    textsize = 13
    labelsize=18
    fontsize=20
    ticksize=15
    df_dmap, mask_dmap, _ = prep_dmaps(info=False)
    
    #% %  Plot residual clusters
    #% % make list with datasets
    das_res_map = [dic['res']['trend'],
               dic['dmap']['res']['trend'] * mask_dmap,
               dic['som']['res']['trend'],
                    ]
    
    titles = ["1 degree residuals",
                r"$\delta$-MAPS residuals ", 
              r"SOM residuals ",
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
    labels = ['1 degree',  'dmap','SOM']
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
    plt.rc('font', family='serif')
    
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
    x2y_ratio = (xmax-xmin)/(ymax-ymin) * nrow/ncol
    
    # Apply new h/w aspect ratio by changing h
    # Also possible to change w using set_figwidth()
    fig.set_figheight(wi * y2x_ratio)
    # fig.set_figwidth(15)
    
    plt.tight_layout()
    # # fig.subplots_adjust(aright=0.8)
    cbar_ax2 = fig.add_axes([0.1, 0.06, 0.25, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=ticksize, family='serif')    
    cbar2.ax.tick_params(labelsize=labelsize) 

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