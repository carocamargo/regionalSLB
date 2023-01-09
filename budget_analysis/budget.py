#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:14:11 2022

@author: ccamargo
"""
import numpy as np
import xarray as xr
import os
import pandas as pd
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")

from utils_SLB import cluster_mean, sum_linear, sum_square, get_dectime
import warnings
warnings.filterwarnings("ignore","Mean of empty slice", RuntimeWarning)

path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
fname = 'budget'
#%% get budget components

dic = {}

period = ['1993-2017'] # full years
y0,y1=period[0].split('-')
t0='{}-01-01'.format(int(y0))
t1='{}-12-31'.format(int(y1)-1)
path = '/Volumes/LaCie_NIOZ/data/budget/trends/' 
path2 = '/Volumes/LaCie_NIOZ/data/budget/ts/' 
# flist = [file for file in os.listdir(path) if not file.startswith('.')]
flist=['steric.nc', 'alt.nc', 'barystatic.nc', 'dynamic.nc']
gia_cor = True
dims = {}
for file in flist:
    comp = file.split('.')[0]
    # print(comp)
    
    ds=xr.open_dataset(path+file)
    da = xr.open_dataset(path2+file)
    # print(da)
    ds = ds.sel(periods=period)
    da = da.sel(time=slice(t0,t1))
    
    if comp=='alt':
        if gia_cor:
            ds=xr.open_dataset(path+'alt_GIAcor.nc')
            ds=ds.sel(names=['ENS'])
            trend = np.array(ds.trends[0,:,:])
            unc = np.array(ds.uncs[0,:,:])
        else:
            ds=ds.sel(names=['ENS'])
            idx = np.where(ds.ICs =='bic_tp')[0][0]
            trend = np.array(ds.best_trend[0,0,idx,:,:])
            unc = np.array(ds.best_unc[0,0,idx,:,:])
        
        ts = np.array(da.sel(names='ENS')['SLA']) # mm
        tdec = get_dectime(da.time)
        dims['time'] = {'tdec':tdec,
                        'xr':da.time}
        dims['lat'] = {'lat':da.lat.values,
                        'xr':da.lat}
        dims['lon'] = {'lon':da.lon.values,
                        'xr':da.lon}
        #% % land mask
        da = da['sla_ens'][0,:,:]
        da = da.where((ds.lat>-66) & (ds.lat<66),np.nan)
        # da.plot()
        landmask = np.array(da.data)
        landmask[np.isfinite(landmask)]=1

        
    elif comp =='dynamic':
        idx = np.where(ds.ICs =='bic_tp')[0][0]
        trend = np.array(ds.best_trend[0,idx,:,:])
        unc = np.array(ds.best_unc[0,idx,:,:])
        # ts = np.array(da['ens_dyn_v1'] * 1000) # mm
        da=da.sel(names=['ENS'])
        ts = np.array(da['DSL'][0,:,:,:] * 1000) # mm

    elif comp=='steric':
        ds=ds.sel(names=['ENS'])
        trend = np.array(ds.trend_full[0,0,:,:])
        unc = np.array(ds.unc[0,0,:,:])
        ts = np.array(da.sel(names='ENS')['steric_full'] * 1000) # mm

        trend_up = np.array(ds.trend_up[0,0,:,:])
        unc_up = np.array(ds.unc[0,0,:,:])
        ts_up = np.array(da.sel(names='ENS')['steric_up']*1000)
        dic['steric_up'] = {'trend':trend_up,
                            'unc':unc_up,
                            'ts':ts_up}
        
    elif comp=='barystatic':
        # ds=ds.sel(names=['IMB_WGP'])
        ds = ds.sel(names=['ENS'])
        trend = np.array(ds.SLA[0,0,:,:])
        unc = np.array(ds.SLA_UNC[0,0,:,:])
        da=da['SLF_ASL'].sel(reconstruction='ENS')
        ts = np.array(da.data)
    
    dic[comp] = {'trend':trend,
                'unc':unc,
                'ts':ts
                }

dic['dims'] = dims
lat=np.array(ds.lat)
lon=np.array(ds.lon)

#%% sum of comps and residual
datasets = ['steric','barystatic','dynamic']
das_unc = []
das_trend = []
das_ts = []
for key in datasets:
    das_unc.append(dic[key]['unc'])
    das_trend.append(dic[key]['trend'])
    das_ts.append(dic[key]['ts'])
    
sum_comps_trend = sum_linear(das_trend)
sum_comps_unc = sum_square(das_unc)
sum_comps_ts = sum_linear(das_ts)

res_trend = sum_linear([dic['alt']['trend'], sum_comps_trend], how='subtract')
# res_unc = sum_square([dic['alt']['unc'], sum_comps_unc], how='subtract')
res_unc = sum_square([dic['alt']['unc'], sum_comps_unc])
res_ts = sum_linear([dic['alt']['ts'], sum_comps_ts], how='subtract')

dic['res']={'trend':res_trend,'unc':res_unc,'ts':res_ts}
dic['sum']={'trend':sum_comps_trend,'unc':sum_comps_unc,'ts':sum_comps_ts}

#%% make list with datasets
datasets = ['alt','steric','barystatic','dynamic','sum','res']
titles = [r"$\eta_{sat(cor)}$",
          r"$\eta_{SSL}$",
          r"$\eta_{BSL}$",
          r"$\eta_{DSL}$",
          r"$\sum(\eta_{SSL}+\eta_{BSL}+\eta_{DSL})$", 
          r"$\eta_{sat} -  \eta_{\sum}$"]
# \eta_{obs} = \eta_{SSL} = \eta_{BSL} + \eta_{DSL} 
# plt.title(r"$\eta$")
das_unc = []
das_trend = []
das_ts = []
for key in datasets:
    das_unc.append(dic[key]['unc'])
    das_trend.append(dic[key]['trend'])
    das_ts.append(dic[key]['ts'])
    
#%% Clusters

#%% SOM 19 regions
path = '//Volumes/LaCie_NIOZ/budget/regions/som/'
file= 'som_3x3_alt_1993_2019_n10_sig2_ep_atlantic_indopacific'
ds=xr.open_dataset(path+file+'.nc')

mask_clusters = np.array(ds.mask)
som = {'mask':mask_clusters}
n_clusters = len(np.unique(mask_clusters[np.isfinite(mask_clusters)]))
som['n']=n_clusters

#% % SOM 19 residuals
mat = np.zeros((n_clusters,len(datasets)))
mat2 = np.zeros((n_clusters,len(datasets)))
mat3 = np.zeros((n_clusters))

n = []

for j,label in enumerate(datasets):
    tmp = np.zeros((n_clusters,180,360))
    tmp2 = np.zeros((n_clusters,180,360))
    tmp3 = np.zeros((n_clusters,180,360))
    
    test = np.zeros((n_clusters,180,360))
    for i in range(n_clusters):
        icluster = i+1
        mask=np.array(mask_clusters)
        mask[np.where(mask!=icluster)]=np.nan
        mask[np.isfinite(mask)]=1
        
        # tmp[i,mask==1] = cluster_mean(np.array(dic[label]['trend']),mask, lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        # tmp2[i,mask==1] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        test[i,mask==1] = icluster
        mat[i,j] = cluster_mean(np.array(dic[label]['trend']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        mat2[i,j] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        tmp[i,mask==1] = mat[i,j]
        tmp2[i,mask==1] = mat2[i,j]
        if label =='res':
            u_res = np.sqrt(dic['alt']['unc']**2 +dic['sum']['unc']**2)
            mat3[i] = cluster_mean(np.array( u_res) ,mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
            tmp3[i,mask==1]=mat3[i]
               
    tr = np.sum(tmp,axis=0)
    tr[tr==0] = np.nan
    unc = np.sum(tmp2,axis=0)
    unc[unc==0] = np.nan
    
    som[label] = {'trend':tr,
                 'unc':unc,
                 }
    if label=='res':
        unc2 = np.sum(tmp3,axis=0)
        unc2[unc2==0] = np.nan
        som[label]['unc2']=unc2
        
som['res']['unc2_cl']=mat3


df_som = pd.DataFrame ({'cluster_n': np.unique(mask_clusters[np.isfinite(mask_clusters)]) })
for j,label in enumerate(datasets):
    df_som['{}_tr'.format(label)] = mat[:,j]
    df_som['{}_unc'.format(label)] = mat2[:,j]
df_som   

som['df']=df_som
dic['som']=som

#% %
#%% dmaps regions

ds=xr.open_dataset('/Volumes/LaCie_NIOZ/budget/regions/dmaps/dmaps_k5_k23.nc')

mask_clusters = np.array(ds.mask[0,:,:])
dmap = {'mask':mask_clusters}
n_clusters = len(np.unique(mask_clusters[np.isfinite(mask_clusters)]))
# plot_map(mask_clusters,cmax=n_clusters,lon0=210,cmap='prism',title='dMAPS k5 Clusters',clabel='cluster number')

# mask_comb = np.array(mask_clusters)
# x=5
# y=20
# plt.figure(figsize=(15,35))
# for i in range(0,int(np.nanmax(mask_clusters))):
#     icluster = i+1
#     ax = plt.subplot(y,x,icluster, projection = ccrs.Robinson(central_longitude=210))
#     ax.set_global()
#     mask=np.array(mask_clusters)
#     mask[np.where(mask!=icluster)]=np.nan
#     mm = ax.pcolormesh(ds.lon,\
#                        ds.lat,\
#                        mask,
#                        vmin=0, vmax=x*y, 
#                        transform=ccrs.PlateCarree(),
#                        #cmap='Spectral_r'
#                        cmap='jet'
#                       )
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', 
#                                                 edgecolor='gray', facecolor='papayawhip'))
#     plt.title('Cluster {}'.format(icluster))
# plt.show()


#% % dmaps residuals
mat = np.zeros((n_clusters,len(datasets)))
mat2 = np.zeros((n_clusters,len(datasets)))
mat3 = np.zeros((n_clusters))
n = []
dmap['n']=n_clusters

for j,label in enumerate(datasets):
    tmp = np.zeros((n_clusters,180,360))
    tmp2 = np.zeros((n_clusters,180,360))
    tmp3 = np.zeros((n_clusters,180,360))
    
    test = np.zeros((n_clusters,180,360))
    for i in range(n_clusters):
        icluster = i+1
        mask=np.array(mask_clusters)
        mask[np.where(mask!=icluster)]=np.nan
        mask[np.isfinite(mask)]=1
        
        # tmp[i,mask==1] = cluster_mean(np.array(dic[label]['trend']),mask, lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        # tmp2[i,mask==1] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        test[i,mask==1] = icluster
        mat[i,j] = cluster_mean(np.array(dic[label]['trend']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        mat2[i,j] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        tmp[i,mask==1] = mat[i,j]
        tmp2[i,mask==1] = mat2[i,j]
        if label =='res':
            u_res = np.sqrt(dic['alt']['unc']**2 +dic['sum']['unc']**2)
            mat3[i] = cluster_mean(np.array( u_res) ,mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
            tmp3[i,mask==1] = mat3[i]
                
    tr = np.sum(tmp,axis=0)
    tr[tr==0] = np.nan
    unc = np.sum(tmp2,axis=0)
    unc[unc==0] = np.nan
    
    dmap[label] = {'trend':tr,
                 'unc':unc,
                 }
    if label=='res':
        unc2 = np.sum(tmp3,axis=0)
        unc2[unc2==0] = np.nan
        dmap[label]['unc2']=unc2
        
dmap['res']['unc2_cl']=mat3

df_dmap = pd.DataFrame ({'cluster_n': np.unique(mask_clusters[np.isfinite(mask_clusters)]) })
for j,label in enumerate(datasets):
    df_dmap['{}_tr'.format(label)] = mat[:,j]
    df_dmap['{}_unc'.format(label)] = mat2[:,j]
df_dmap   

dmap['df']=df_dmap
dic['dmap']=dmap

#%%
# %% SOM 4x4 world
path = '//Volumes/LaCie_NIOZ/budget/regions/som/'
file= 'som_4x4_alt_1993_2019_n10_sig2_ep_world'
ds=xr.open_dataset(path+file+'.nc')

mask_clusters = np.array(ds.regions)
som = {'mask':mask_clusters}
n_clusters = len(np.unique(mask_clusters[np.isfinite(mask_clusters)]))

# plot_map(mask_clusters,cmax=n_clusters,lon0=210,title='SOM Clusters',clabel='cluster number')

# mask_comb = np.array(mask_clusters)
# x=4
# y=5
# plt.figure(figsize=(20,10))
# for i in range(0,int(np.nanmax(mask_clusters))):
#     icluster = i+1
#     ax = plt.subplot(y,x,icluster, projection = ccrs.Robinson(central_longitude=210))
#     ax.set_global()
#     mask=np.array(mask_clusters)
#     mask[np.where(mask!=icluster)]=np.nan
#     mm = ax.pcolormesh(ds.lon,\
#                        ds.lat,\
#                        mask,
#                        vmin=0, vmax=x*y, 
#                        transform=ccrs.PlateCarree(),
#                        #cmap='Spectral_r'
#                        cmap='jet'
#                       )
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', 
#                                                 edgecolor='gray', facecolor='papayawhip'))
#     plt.title('Cluster {}'.format(icluster))
# plt.show()
# cluster 10 is empty:
n_clusters = len(np.arange(0,int(np.nanmax(mask_clusters))))

#% % SOM 4x4 residuals
mat = np.zeros((n_clusters,len(datasets)))
mat2 = np.zeros((n_clusters,len(datasets)))
mat = np.zeros((n_clusters,len(datasets)))
mat2 = np.zeros((n_clusters,len(datasets)))
mat3 = np.zeros((n_clusters))

n = []

for j,label in enumerate(datasets):
    tmp = np.zeros((n_clusters,180,360))
    tmp2 = np.zeros((n_clusters,180,360))
    tmp3 = np.zeros((n_clusters,180,360))
    
    test = np.zeros((n_clusters,180,360))
    for i in range(n_clusters):
        icluster = i+1
        mask=np.array(mask_clusters)
        mask[np.where(mask!=icluster)]=np.nan
        mask[np.isfinite(mask)]=1
        
        # tmp[i,mask==1] = cluster_mean(np.array(dic[label]['trend']),mask, lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        # tmp2[i,mask==1] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        test[i,mask==1] = icluster
        mat[i,j] = cluster_mean(np.array(dic[label]['trend']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        mat2[i,j] = cluster_mean(np.array(dic[label]['unc']),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
        tmp[i,mask==1] = mat[i,j]
        tmp2[i,mask==1] = mat2[i,j]
        if label =='res':
            u_res = np.sqrt(dic['alt']['unc']**2 +dic['sum']['unc']**2)
            mat3[i] = cluster_mean(np.array( u_res),mask,lat=np.array(ds.lat),lon=np.array(ds.lon),norm=False )
            tmp3[i,mask==1]=mat3[i]
               
    tr = np.sum(tmp,axis=0)
    tr[tr==0] = np.nan
    unc = np.sum(tmp2,axis=0)
    unc[unc==0] = np.nan
    
    som[label] = {'trend':tr,
                 'unc':unc,
                 }
    if label=='res':
        unc2 = np.sum(tmp3,axis=0)
        unc2[unc2==0] = np.nan
        som[label]['unc2']=unc2
        
som['res']['unc2_cl']=mat3
    

df_som = pd.DataFrame ({'cluster_n': np.arange(0,int(np.nanmax(mask_clusters))) })
for j,label in enumerate(datasets):
    df_som['{}_tr'.format(label)] = mat[:,j]
    df_som['{}_unc'.format(label)] = mat2[:,j]
df_som   

som['df']=df_som
dic['som_4x4']=som

#% % make list with datasets
das_clusters = [dic['som_4x4']['res']['trend'],# dic['dmap']['res']['trend'],
                # dic['som']['res']['unc'],dic['dmap']['res']['unc'],
                ]
tr = np.array(dic['som_4x4']['res']['trend'])
unc = np.array(dic['som_4x4']['res']['unc'])
tr[np.abs(tr)<unc] = np.nan
das_clusters.append(tr)
# tr = np.array(dic['dmap']['res']['trend'])
# unc = np.array(dic['dmap']['res']['unc'])
# tr[np.abs(tr)<unc] = np.nan
# das_clusters.append(tr)

# titles = [r"SOM residuals ",# r"dMAPS residuals ", 
#           # r"unc", r"unc", 
#           'Significant residual', # "Significant residual"
                    
#           ]

# #% % plot trends for each component
# clim=5
# plot_map_subplots( das_clusters,
#              plot_type = 'pcolor',
#              lon=lon,lat=lat,
#              cmap=cmap_trend,
#              cmin=-clim,cmax=clim,
#              titles=titles,
#              clabel='nmm/yr',
#              lon0=210, offset_y = -0.2,
#              fontsize=25,
#              fsize=(15,10),
#              nrow=2,ncol=1
#              )
#%% 
mask = np.array(dic['som']['mask'])
mask[np.isfinite(mask)]=1
dic['landmask'] = mask
# # # load pickle module
import pickle
# path = '/Volumes/LaCie_NIOZ/data/budget/'
path = path_to_data
# # create a binary pickle file 
# f = open(path+"budget_v2.pkl","wb")
f = open(path+fname+".pkl","wb")


# # write the python object (dict) to pickle file
pickle.dump(dic,f)

# # close file
f.close()
