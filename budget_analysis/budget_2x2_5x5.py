#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:26:18 2022

@author: ccamargo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import sklearn.metrics as metrics
import random
import string
letters = list(string.ascii_lowercase)

path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'

#%%
def open_dataset(res = 5):
    # res = 2
    import xarray as xr
    path = '/Volumes/LaCie_NIOZ/data/budget/'
    path = path_to_data
    fname = 'budget_components_{}deg'.format(res)    
    da = xr.open_dataset(path+fname+'.nc')
        
    return da
import cmocean as cm
def plot(
        dataset,titles,lat,lon,
        clim=5,
        cmap=cm.cm.balance,
        clabel='Trend \nmm/yr',
        offset_y = -0.15,
        nrow=3,
        ncol=2,
        interval = 0.1,
        sig = False,
        proj = 'robin'
        ):
    
    # import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    # import matplotlib.pyplot as plt
    import sys
    sys.path.append("/Users/ccamargo/Documents/py_scripts/")
    import utils_SL as sl
    import string
    letters = list(string.ascii_lowercase)
    # Global plotting settings
    fontsize=25
    lon0=201;
    # fsize=(15,10)
    proj='robin'
    # land=True
    # grid=False
    landcolor='papayawhip'
    # extent = False
    plot_type = 'contour'
    cmin=-clim;cmax=clim
    nrow=8
    ncol=2
    proj = 'robin'
    plot_type = 'pcolor'
    fig = plt.figure(figsize=(15,12),dpi=100)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=lon0)
    else:
        proj=ccrs.PlateCarree()
    
    idata = 0
    ax = plt.subplot2grid((nrow,ncol), (0, 0), rowspan=3, projection=proj)
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
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=landcolor))
    
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
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=landcolor))
    
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
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=landcolor))
    
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
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=landcolor))
    
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
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='gray', facecolor=landcolor))
    
    plt.tight_layout()
    # # fig.subplots_adjust(right=0.8)
    cbar_ax2 = fig.add_axes([0.25, 0.2, 0.5, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=fontsize-5, family='serif')    
    cbar2.ax.tick_params(labelsize=fontsize-5) 
    
    plt.show()
def fig1():
    import numpy as np
    da = open_dataset()
    
    #% % make list with datasets
    datasets = ['altimetry',
                'sum',
                'steric',
                'GRD',
                # 'res',
                'dynamic',
                # 'sum','res'
               ]
    titles = [r"$\eta_{obs}$",
              r"$\sum(\eta_{SSL}+\eta_{GRD}+\eta_{DSL})$", 
              r"$\eta_{SSL}$", 
              r"$\eta_{GRD}$",
             #  r"$\eta_{obs} - \eta_{\sum(drivers)}$", 
              r"$\eta_{DSL}$",
             ]
    # \eta_{obs} = \eta_{SSL} = \eta_{BSL} + \eta_{DSL} 
    # plt.title(r"$\eta$")
    das_unc = []
    das_trend = []
    
    for key in datasets:
        
        das_unc.append(da[key+'_unc'].values)
        das_trend.append(da[key+'_trend'].values)
    
    plot(das_trend,titles,
         np.array(da.lat),np.array(da.lon))

def budget_stats():
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    # from scipy import stats
    import sklearn.metrics as metrics
    import warnings
    warnings.filterwarnings("ignore","Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore","divide by zero encountered in double_scalars", RuntimeWarning)
            

    stats = {}
    for res in [2,5]:
    
        key = '{}degree'.format(res)
        da = open_dataset(res=res)
        #mask:
        X = np.array(da['sum_trend']) + np.array(da['altimetry_trend'])
        mask = np.array(X).flatten()
        mask[np.isfinite(mask)] = 1
        nn = len(mask[np.isfinite(mask)])
        
        Y = np.array(da['altimetry_ts'])
        dimtime,dimlat,dimlon = Y.shape
        Y = Y.reshape(dimtime,dimlat*dimlon)
        
        varis = ['sum',
                    'steric',
                     'GRD',
                     'dynamic',
                 'steric+dynamic',
                 'GRD+dynamic',
                 'steric+GRD',
                 'steric+GRD+dynamic'
                           ]
        nvar = len(varis)
        n = len(mask)
        R2 = np.zeros((nn,nvar))
        R2.fill(np.nan)
        r = np.full_like(R2,np.nan)
        rp = np.full_like(R2,np.nan)
        RMSE = np.full_like(R2,np.nan)
        nRMSE = np.full_like(R2,np.nan)
        mRMSE = np.full_like(R2,np.nan)
        
        for j,var in enumerate(varis):
            if len(var.split('+'))==1:
                X = np.array(da[var+'_ts']).reshape(dimtime,dimlat*dimlon)
            else:
                X = np.array([da[subvar+'_ts'] for subvar in var.split('+')])
                X = np.sum(np.array(X),axis=0).reshape(dimtime,dimlat*dimlon)
                
                    
            # X = np.array(dic[var]['ts']).reshape(dimtime,dimlat*dimlon)
            ij = 0
            for i, m in enumerate(mask):
                if np.isfinite(m):
                    
                    y = np.array(Y[:,i])
                    y = np.array(y - np.nanmean(y))
                    ymean = np.nanmean(y)
                    SStot = np.nansum(np.array(y-ymean)**2)
                    x = np.array(X[:,i])
                    x = np.array(x - np.nanmean(x))
                    # plt.plot(x,label=var)
                    
                    if np.any(np.isnan(x)):
                        y = y[np.isfinite(x)]
                        x = x[np.isfinite(y)]
                    if np.any(np.isnan(y)):
                        x = x[np.isfinite(y)]
                        y = y[np.isfinite(y)]
                        
                    SSres = np.sum(np.array(y-x)**2)
        
                    R2[ij,j] = 1 - (SSres/SStot)
                    # print('{}:\n R2 = {:.3f}'.format(var, R2[i,j]))
        
                    r[ij,j],rp[ij,j] = st.pearsonr(x,y)
                    RMSE[ij,j] = np.sqrt(metrics.mean_squared_error(y,x))
                    yrange = np.array(np.nanmax(y) - np.nanmin(y))
                    nRMSE[ij,j] = np.array(RMSE[ij,j]/yrange)
                    mRMSE[ij,j] = np.array(RMSE[ij,j]/ymean)
                    ij = ij+1
                    # R = metrics.r2_score(y,x)
                    #print('R = {:.3f}'.format(R))
                    # print('r = {:.3f} ({:.2e})\n'.format(r[i,j],rp[i,j]))
        
        
        stats[key] = {'R2':R2,
                       'r':r,
                       'p_value':rp,
                        'RMSE':RMSE,
                        'nRMSE':nRMSE,
                        'mRMSE':mRMSE}
        
    # make dataframe
    combs = ['(alt,sum)','(alt,ste)','(alt,GRD)','(alt,dyn)',
            '(alt,ste+dyn)','(alt,GRD+dyn)','(alt,ste+GRD)','(alt,ste+dyn+GRD)' ]
    varis = ['R2','r','RMSE','nRMSE',
             # 'mRMSE'
            ]
    labels = [key for key in stats.keys()]
    for ilabel, label in enumerate(labels):
        var = varis[0]
        
        data = np.array(stats[label][var])
        i,j = data.shape
        if ilabel==0:
            df = pd.DataFrame({'comb':np.tile(combs,i),
                               'budget':np.tile(label,i*j)
                              })
            for var in varis:
                df[var] = np.array(stats[label][var]).flatten()
        else:
            df2 = pd.DataFrame({'comb':np.tile(combs,i),
                       'budget':np.tile(label,i*j)
                      })
            for var in varis:
                df2[var] = np.array(stats[label][var]).flatten()
            df = df.append(df2)
    return df

# path = '/Volumes/LaCie_NIOZ/data/budget/'
path = path_to_data
def complement_stats():
    import pandas as pd
    df = budget_stats()
    
    # path = '/Volumes/LaCie_NIOZ/data/budget/'
    # path = '/Users/ccamargo/Desktop/manuscript_SLB/data/'
    path = path_to_data
    #path = '/Volumes/LaCie_NIOZ/data/budget/'
    df2 = pd.read_pickle(path+"budget-stats.p")
    df2['comb'] = [combo.replace('bar','GRD') 
                   # if 'bar' in combo 
                   # else combo
                   for combo in df2['comb']]
    df = df2.append(df)
    df = df.drop(df[df.comb == '(alt,sum)'].index)
    df['budget_name'] = [
        '1x1' if label=='1degree'  
        else 'SOM' if label=='som'  
        else '2x2' if label=='2degree'
        else '5x5' if label=='5degree'
        else '$\delta$-MAPS' for label in df['budget'] 
        ]
    
    df.to_pickle(path+"budget-stats_complete.p")
    # df

    return df
complement_stats()
def plot_stats():
    #% %
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    df =  pd.read_pickle(path+"budget-stats_complete.p")
    # plot grouped boxplot
    # df = df.sort_values(by=['budget_name'])
    varis = ['r','nRMSE']
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
        plt.ylabel(var,fontsize=20)
        plt.xlabel('')
        if i==0:
            plt.legend(loc='lower left')
            plt.text(6.25,-0.9,'({})'.format(letters[i]),fontsize=textsize,fontweight="bold")
        else: 
            plt.legend(loc='upper left')
            plt.text(6.25,0.28,'({})'.format(letters[i]),fontsize=textsize,fontweight="bold")   
        
    plt.tight_layout()
    plt.show()

#%% plot

# res = 2
# key = '{}degree'.format(res)
# da2 = open_dataset(res=res)
# res = 5
# key = '{}degree'.format(res)
# da5 = open_dataset(res=res)
# masks = [np.array(da2['mask_1deg']),np.array(da5['mask_1deg'])]
# das_res = [da2['residual_trend'],
#            da5['residual_trend'],
#                 ]

# das_alt = [da2['altimetry_trend'],
#            da5['altimetry_trend'],
#                 ]

# das_sum = [da2['sum_trend'],
#            da5['sum_trend'],
#                 ]

# das_res_u = [da2['residual_unc'],
#              da5['residual_unc'],
#                 ]

# das_alt_u = [da2['altimetry_unc'],
#              da5['altimetry_unc'],
#                 ]

# das_sum_u = [da2['sum_unc'],
#              da5['sum_unc'],
#                 ]

# i=0
# ws = []
# clim=7.5
# title=['2 degrees', '5 degrees']
# titles = ['2 degree residuals',
#           '5 degree residuals', 
#           # r"unc", r"unc",                     
#           ]
# pos=[1,4]
# lims = [(0,3000),(0,500),]

# fig = plt.figure(figsize=(15,10))
# for i in range(2):
#     df = pd.DataFrame({'Altimetry':np.hstack(das_alt[i]),
#                        'Sum':np.hstack(das_sum[i]),
#                       # 'res':np.hstack(adas_res[i])
#                       })

#     df.dropna(inplace=True)
#     df.reset_index(inplace=True)
#     df = df.drop('index',axis=1)

#     ax = plt.subplot(2,3,pos[i])
#     for name in ['Altimetry','Sum']:
#         sns.histplot(df[name],kde=False, #label='Trend',
#                      alpha=0.5,
#                      label=name,
#                      # stat="percent", 
#                      # color=colors_dic[name],
#                     bins=np.arange(-clim, clim+0.1,0.5))
#     if i==2:
#         plt.xlabel('mm/yr')
#     else:
#         plt.xlabel('')
#     plt.ylabel('Number regions')
#     # plt.title(title[i])
#     title = '({}). '.format(letters[i+i])+str(titles[i])
#     plt.title(title,fontsize=15)
    
#     plt.ylim(lims[i])
#     plt.legend(loc='upper left')
# i=0
# pos=[2,5]

# for trend,unc in zip(das_res,das_res_u):
#     df= pd.DataFrame( {
#                     'Trend':np.hstack(trend),
#     'Unc':np.hstack(unc),
#     })
    
#     df.dropna(inplace=True)
#     df.reset_index(inplace=True)
#     df = df.drop('index',axis=1)
    
#     ax = plt.subplot(2,3,pos[i])
#     c = {'Trend':'darkgray',
#         'Unc':'lightpink'}
#     for var in ['Trend','Unc']:
#         sns.histplot(df[var],kde=False, #label='Trend',
#                      alpha=0.5,
#                      color=c[var],
#                      label=var,
#                      # stat="percent", 
#                     bins=np.arange(-clim, clim+0.1,0.5))

#     # plt.legend(prop={'size': 12})
#     if i==1:
#         plt.xlabel('mm/yr')
#     else:
#         plt.xlabel('')
#     plt.ylabel('')
#     plt.xlim(-clim,clim)
    
#     plt.ylim(lims[i])
#     var = 'Unc'
#     x = np.array(df[var])
#     ci_level=0.95
#     ci = st.norm.interval(alpha=ci_level, loc=np.mean(x), scale=x.std())
#     plt.axvline(x.mean(),color='k',linestyle='--')

#     plt.axvline(ci[0],c=c[var],linestyle='--',alpha=1,label='{}% CI'.format(ci_level*100))
#     plt.axvline(ci[1],c=c[var],linestyle='--',alpha=0.5)
    
#     ci_width = np.abs(ci[0]-ci[1])/2
#     ws.append(ci_width)
#     # plt.title(title[i]+': {:.3f}'.format( ci_width))
#     title = '({}). '.format(letters[i+i+1])+str(titles[i]+': {:.3f}'.format( ci_width))
#     plt.title(title,fontsize=15)
    
#     i=i+1
#     plt.legend(loc='upper left')
  

# #####################
# #####################
# ## Scatter plots
# #####################
# #####################
# labels = ['2 degrees',  '5 degrees']
# inds = [3,6]
# cmin=-5
# cmax=10
# clim = [cmin,cmax]
# j=0
# labelsize=18
# fontsize=20
# ticksize=15
# textsize = 13

# for i,label in zip(inds,labels):
#     ax = plt.subplot(2,3,i)
#     key ='som' # cluster
#     mask = masks[j]
#     mask_tmp = np.array(mask)
#     mask_tmp[np.isfinite(mask_tmp)]=1
#     if label =='2 degrees':
#         dic = da2
#     else:
#         dic = da5
        
#     y = np.array(dic['altimetry_trend'] * mask_tmp).flatten() 
#     x = np.array(dic['sum_trend'] * mask_tmp).flatten()
#     yerr = np.array(dic['altimetry_unc'] * mask_tmp).flatten() 
#     xerr = np.array(dic['sum_unc'] * mask_tmp).flatten()
    
#     plt.title('({}). {}'.format(letters[i],label),fontsize=fontsize)

#     x = x[np.isfinite(y)]
#     xerr = xerr[np.isfinite(y)]
#     yerr = yerr[np.isfinite(y)]
#     y = y[np.isfinite(y)]

#     y = y[np.isfinite(x)]
#     xerr = xerr[np.isfinite(x)]
#     yerr = yerr[np.isfinite(x)]
#     x = x[np.isfinite(x)]


#     plt.plot(clim, clim, ls="--", c=".1")
#     w=ws[j]/2
#     j=j+1
#     plt.plot([cmin-w,cmax-w], [cmin+w,cmax+w], ls="--", c="pink")
#     plt.plot([cmin+w,cmax+w], [cmin-w,cmax-w], ls="--", c="pink")

#     plt.errorbar(x,y,
#                  yerr=yerr,
#                  xerr = xerr,
#                  #c=c,
#                  # s=1,
#                  alpha=0.1,
#                  zorder=0,
#                  capsize=3,capthick=2,ecolor='gray',lw=2,fmt='none')
#     plt.scatter(x,y,
#                 alpha=0.5,
#                 marker='o',
#                 # facecolors='none', edgecolors='b'
#                )


#     plt.xlim(clim)
#     plt.ylim(clim)
#     if len(x)>500:
#         idx = random.sample(list(np.arange(0,len(x))),500)
#         xx = x[idx]
#         yy = y[idx]
#     else:
#         xx=x
#         yy=y

#     #ax.annotate("$R^2$ = {:.2f}".format(metrics.r2_score(y-np.nanmean(y),x-np.nanmean(x))), (clim[0]+1,clim[1]-3))
#     r,p = st.pearsonr(xx, yy)
#     if p<0.05:
#         if p<0.0001:
#             ax.annotate("Pearsons r = {:.2f}**".format(r), (clim[0]+0.5,clim[1]-2),fontsize=textsize)
#         else:
#              ax.annotate("Pearsons r = {:.2f}*".format(r), (clim[0],clim[1]-2),fontsize=textsize)  
#     else:
#          ax.annotate("Pearsons r = {:.2f}".format(r), (clim[0],clim[1]-2),fontsize=textsize)
#     ax.annotate("RMSE = {:.2f}".format(np.sqrt(metrics.mean_squared_error(y,x))),(clim[0]+0.5,clim[1]-3.5),fontsize=textsize)

#     plt.xlabel('', fontsize=labelsize)
#     plt.ylabel('Altimetry \n(mm/yr)', fontsize=labelsize)
#     if i==5:
#         plt.xlabel('Sum of Componentns \n(mm/yr)', fontsize=labelsize)

# plt.tight_layout()
# plt.show()
