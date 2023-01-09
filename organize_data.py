#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:06:55 2022

make data to publish

@author: ccamargo
"""
import pandas as pd
import numpy as np 
import xarray as xr
path_save = '/Users/ccamargo/Desktop/manuscript_SLB/data_to_publish/v1/'

path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
# path_save = path_to_data+'pub/'
#%% load data 
# path = '/Volumes/LaCie_NIOZ/data/budget/'
path = path_to_data
# file = 'budget_v2'
file = 'budget'

fmt = 'pkl'
dic = pd.read_pickle(path+file+'.'+fmt)
#%% Budget Components
def save_components(dic,path_save):
    comps = ['alt','steric','barystatic','dynamic']
    lat = dic['dims']['lat']['lat']
    lon = dic['dims']['lon']['lon']
    time = dic['dims']['time']['tdec']
    mask = np.array(dic['landmask'])
    trends = np.zeros((len(comps),len(lat),len(lon)))
    uncs = np.zeros((len(comps),len(lat),len(lon)))
    ts = np.zeros((len(comps),len(time),len(lat),len(lon)))
    
    for i,c in enumerate(comps):
        trends[i] = np.array(dic[c]['trend'] * mask)
        uncs[i] = np.array(dic[c]['unc']*mask)
        ts[i] = np.array(dic[c]['ts']*mask)
        
    ds = xr.Dataset(data_vars={
                    'trends':(('comp','lat','lon'),trends),
                    'uncs':(('comp','lat','lon'),uncs),
                    'ts':(('comp','time','lat','lon'),ts),
                    'mask':(('lat','lon'),mask)
                    },
                     coords={'lat':dic['dims']['lat']['xr'],
                             'lon':dic['dims']['lon']['xr'],
                             'time':dic['dims']['time']['xr'],
                             'comp':['Altimetry','Steric','GRD','Dynamic']
                             }
                     )
    ds['trends'].attrs = {'axis': 'X,Y',
                         'long_name': 'trends',
                         'standard_name': 'sea-level trend',
                         'units': 'mm/yr'
                         }
    ds['uncs'].attrs = {'axis': 'X,Y',
                         'long_name': 'uncertainties',
                         'standard_name': 'sea-level uncertainty',
                         'units': 'mm/yr'
                         }
    ds['ts'].attrs = {'axis': 't,X,Y',
                         'long_name': 'time_series',
                         'standard_name': 'sea-level time-series',
                         'units': 'mm'
                         }
    ds['time'].attrs ={'axis':'t',
                       'long_name':'time',
                       'standard_name':'time',
                        'unit':'months since 1993'
                       }
    ds['comp'].attrs = {'long_name':'components',
                         'standard_name':'budget components',
                         'Altimetry':'Total sea-level from altimetry, ensemble of AVISO, CMEMS, CSIRO and MEASURES',
                         'GRD': 'Mass-driven SLC from ensemble of Camargo et al, 2021 (Earth System Dynamics)',
                         'Steric': 'Density-driven SLC from ensemble of 15 datasets of Camargo et al., 2020 (JGR:Oceans), complemented with deep-steric of Purkey and Johnson (2010)',
                         'Dynamic':'Dynamic driven SLC from ocean reanalysis',
                         }
    ds.attrs ={'metadata':'Sea-level trends, uncertainties and time series from 1993-2016',
               'source':'Camargo et al. (2022)',
               'Observations':'Source of observations is described in Camargo et al., 2020 (JGR:Oceans), Camargo et al., 2021 (Earth System Dynamic) and Camargo et al., 2022 (Journal of Climate)',
               'contact':'carolina.camargo@nioz.nl'
               }
    ds.to_netcdf(path_save+'budget_components_ENS.nc')
    return

#%% masks

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

def prep_dmaps(dic):
    lats = []
    lons = []
    size = []
    c = []
    n = []
    a = []
    key='dmap'
    # dic = load_data(fmt = 'pkl')
    lon = dic['dims']['lon']['lon']
    lat = dic['dims']['lat']['lat']
    million = 1e+6

    grid_area = get_grid_area(dic['landmask'])/million

    key='dmap'
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
    df2 = pd.DataFrame({'Altimetry_trend':df2['alt_tr'],
                      'Altimetry_unc':df2['alt_unc'],
                       
                       'Steric_trend':df2['steric_tr'],
                       'Steric_unc':df2['steric_unc'],
                       'GRD_trend':df2['barystatic_tr'],
                       'GRD_unc':df2['barystatic_unc'],
                      'Dynamic_trend':df2['dynamic_tr'],
                      'Dynamic_unc':df2['dynamic_unc'],
                      
                      
                       
                     'Sum_trend':df2['sum_tr'],
                     'Sum_unc':df2['sum_unc'],
                     
                      
                        'Residual_trend':df2['res_tr'],
                        'Residual_unc':df2['res_unc']
                       })
    
    
    df2['cluster_n'] = [int(i) for i in dic['dmap']['df']['cluster_n']]
    df2['Sum_trend'] = df2['Steric_trend'] + df2['Dynamic_trend'] + df2['GRD_trend']
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
    
    
    A = np.array(df3['Altimetry_trend'])
    A_error = np.array(df3['Altimetry_unc'])
    B = np.array(df3['Sum_unc'])
    B_error = np.array(df3['Sum_trend'])
    
    C = np.array([compare_values(A[i],A_error[i],B[i],B_error[i]) for i in range(len(A))])
    # res_closed = len(C[C==1])
    df3['closure'] = C
    ((df3['area']*df3['closure']).sum() * 100)/df3['area'].sum()
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

    return dic['dmap']['mask'], df3


def load_SOMs(
        path,
        file
        ):
    ds = xr.open_dataset(path+file)
    mask = np.array(ds.bmu_map)
    return mask

def save_masks(dic,path_save):
    ds = xr.Dataset(data_vars={
                    'landmask':(('lat','lon'),dic['landmask']),
                    },
                     coords={'lat':dic['dims']['lat']['xr'],
                             'lon':dic['dims']['lon']['xr'],
                             }
                     )
    
    mask_dmap, df3 = prep_dmaps(dic)
    ds['dmaps_k5']=(('lat','lon'),mask_dmap)
    ds['dmaps_k5'].attrs={'metadata':'Domains identified with delta-maps algorithm',
                          'number_domains':'92',
                          'k':5}
    ds['som_2x3x3'] = (('lat','lon'),dic['som']['mask'])
    ds['som_2x3x3'].attrs={'number_domains':18,
                           'metadata':'Domains identified with Self-organizing Maps,  on the Atlantic and Indo-Pacific Ocean basins separately'
                           }
    for i,n in enumerate([4,9]):
        fname = 'som_{}x{}_sig_init2_norm_range_train_n10_ngb_function_ep_mask_ocean_.nc'.format(n,n)
    
        mask = load_SOMs(
                        path = '/Volumes/LaCie_NIOZ/reg/SOM_MATLAB/output/',
                        file = fname
                        )
        ds['som_{}x{}'.format(n,n)] = (('lat','lon'),mask)
        ds['som_{}x{}'.format(n,n)].attrs={'number_domains':n*n,
                               'metadata':'Domains identified with Self-organizing Maps on global oceans'
                               }
    
    ds.attrs={
        'metadata':'Clustering applied on  satellite altimetry time-series (CMEMS2022), for 1993-2019, pre-processed by removing the global mean trend, seasonality and by applying a spatial Gaussian filter of 300km width.' 
        
        }
    ds.to_netcdf(path_save+'masks.nc')
    return
#%% delta maps and  SOM tables
def save_domains(dic,path_save):

    # SOM
    key='som'
    df = dic[key]['df']
    df.rename(columns = {'barystatic_tr':'GRD_trend', 
                         'barystatic_unc':'GRD_unc',
                         'alt_tr':'Altimetry_trend',
                         'alt_unc':'Altimetry_unc',
                         'res_tr':'Residual_trend',
                         'res_unc':'Residual_unc',
                         'steric_tr':'Steric_trend',
                         'steric_unc':'Steric_unc',
                         'dynamic_tr':'Dynamic_trend',
                         'dynamic_unc':'Dynamic_unc',
                         'cluster_n':'Domain_number',
                         'sum_tr':'Sum_trend',
                         'sum_unc':'Sum_unc',
                         
                         }, 
              inplace = True)
    df.to_excel(path_save+"SOM_trends.xlsx",
             ) 
    df.to_pickle(path_save+"SOM_trends.pkl")
    
    # dmaps
    _, df2 = prep_dmaps(dic)
    df2.to_excel(path_save+"dmaps_trends.xlsx",
         ) 
    df2.rename(columns={'cluster_n':'Domain_number'},
               inplace=True)
    df2.drop(columns=['closure'],inplace=True)
    df2.to_excel(path_save+"dmaps_trends.xlsx",
             ) 
    df2.to_pickle(path_save+"dmaps_trends.pkl")
    return

#%% budget sensitivy
# def save_budget_sen(dic,path_save):
    
#     return
#%% call
save_components(dic,path_save)
save_masks(dic,path_save)
save_domains(dic,path_save)
