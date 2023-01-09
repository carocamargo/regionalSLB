#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:30:43 2022

@author: ccamargo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")

from utils_SLB import cluster_mean# , plot_map_subplots, sum_linear, sum_square, get_dectime


path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'
# make_figure(save=False)

def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'budget_combination',
                figfmt='png'
                ):
    dic = load_data()
    #% % sensitivity to budget combination 
    da = xr.open_dataset('/Volumes/LaCie_NIOZ/data/budget/combinations.nc')
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
    pwd ="/Volumes/LaCie_NIOZ/data/budget/"
    dicb = pd.read_pickle(pwd+"best_rmse.pkl")
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
    
    fontsize=20
    size=75
    y = cluster_combos_res.flatten()
    y_sig = cluster_combos_sig.flatten()
    x = range(len(y))
    col= ['salmon' if sig==1 else 'mediumslateblue' for sig in y_sig]
    #% %
    fig = plt.figure(dpi=300,figsize=(15,10))
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
    plt.ylabel('mm/yr',fontsize=fontsize)
    
    ax.set_xticks(xs[0:len(xs)-1]+200)
    ax.set_xticklabels(np.arange(1,19))
    
    plt.axhline(y = 0, color = 'gray', linestyle = '--')
    
    plt.xlabel('SOM region number',fontsize=fontsize)
    
    plt.legend(fontsize=fontsize)
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


  
    
    