#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:40:09 2022

@author: ccamargo
"""


import numpy as np

# import scipy.optimize as opti
import xarray as xr

# import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import utils_SLE_v2 as sle

# from netCDF4 import Dataset
import pandas as pd

# import os

# import datetime as dt

# import cmocean as cm
# from mpl_toolkits.basemap import Basemap
# from matplotlib.gridspec import GridSpec
# from cartopy import crs as ccrs#, feature as cfeature

#% % packages for plotting
# from pandas.plotting import table
# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.basemap import Basemap
# from matplotlib.colors import ListedColormap

# # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
# col_dict={1:"black", # WN
#           2:"palegoldenrod", # PL
#           3:"lightpink", # PLWN
#           4:"orange", # AR1
#           5:"teal", # Ar5
#           6:"darkmagenta", # AR9
#           7:"skyblue", # ARf
#           8:"crimson" # GGM
#           }

# # We create a colormar from our list of colors
# cmapnm = ListedColormap([col_dict[x] for x in col_dict.keys()])
def lat2str(deg):
    # Source: https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
    # Adapted so that 0 has no indication of direction.
    minn = 60 * (deg - np.floor(deg))  # transform to minutes
    deg = np.floor(deg)  # degrees
    dirr = "N"
    if deg < 0:
        if minn != 0.0:
            deg += 1.0
            minn -= 60.0
        dirr = "S"
    elif deg == 0:
        dirr = ""
    return ("%d\N{DEGREE SIGN} %s") % (np.abs(deg), dirr)


def lon2str(deg):
    # Source: https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
    # Adapted so that 0 has no indication of direction.
    minn = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    dirr = ""  #'E'
    if deg < 0:
        if minn != 0.0:
            deg += 1.0
            minn -= 60.0
        dirr = ""  #'W'
    elif deg == 0:
        dirr = ""
    return ("%d\N{DEGREE SIGN} %s") % (np.abs(deg), dirr)


#% %
# ds_mask = xr.open_dataset('/Volumes/LaCie_NIOZ/data/ETOPO/ETOPO1_Ice-180x360.nc')
# ds_mask.z.plot(# vmin=-1,vmax=1
#                 );#plt.show()

# # ds_mask=ds_mask.sortby('lat',ascending=False)
# oceanmask=np.array(ds_mask.z)
# oceanmask[oceanmask>=0]=1
# oceanmask[oceanmask<=0]=np.nan
#% %
def ocean_mean(value, lat, lon):
    # value=np.array(ds.best_trend[0,:,:])
    ocean_lit = np.array([360000000, 361060000, 357000000, 360008310, 357000000])
    ocean_area = np.mean(ocean_lit) / 10 ** 5
    grid_area = sl.get_grid_area(np.ones((180, 360)))

    # plt.pcolor(oceanmask);#plt.show()

    # value=np.array(tws_gbw)
    # tdec=np.array(tdec_gwb)
    da = xr.Dataset(
        data_vars={"data": (("lat", "lon"), value)}, coords={"lat": lat, "lon": lon}
    )
    mu = (da.data * grid_area).sum(dim=("lat", "lon")) / ocean_area
    return mu.data


#%% periods
periods = [(2005, 2016), (1993, 2018), (2003, 2017)]
#%% run SLE
for period in periods:
    t0 = period[0]
    t1 = period[1] - 1

    pwd = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/hector/"
    # t0=2005
    # t1=2015
    pwd = pwd + "{}-{}/".format(t0, t1)

    flist = sl.get_filelist(pwd + "ranking/regional/", "*.nc")

    ic_idx = -1
    # ifile=0;file=flist[ifile]
    for ifile, file in enumerate(flist):
        ds = xr.open_dataset(file)
        name = file.split("/")[-1].split(".")[0]
        dataset = name.split("_")[1]
        # SLE
        X = np.array(ds.lon)
        Y = np.array(-ds.lat)
        data = np.array(ds.best_trend[ic_idx, :, :])
        if dataset == "IMB":
            data = -data
        # if name[ifile].split('_')[0]=='LWS':
        #     data = sle.height_to_EWH(data).reshape(180,360)
        slf_tr = sle.run_SLE(data, name + "_tr", var="asl")
        # sle.plot_contour_local(X,Y,slf_tr,fname=name+'_tr',
        #                        save=False)

        data = np.array(ds.best_unc[ic_idx, :, :])
        if dataset == "IMB":
            data = -data
        # if name[ifile].split('_')[0]=='LWS':
        #     data = sle.height_to_EWH(data).reshape(180,360)
        slf_unc = sle.run_SLE(data, name + "_unc", var="asl")
        # sle.plot_contour_local(X,Y,np.abs(slf_unc),fname=name+'_unc', # Unc should always be positive!
        #                                 save=True, savename=name+'_unc',
        #                                 path=pwd+'plot_SLE/')

        if ifile == 0:
            df_tr = pd.DataFrame(slf_tr.flatten(), columns=[name])
            df_unc = pd.DataFrame(slf_unc.flatten(), columns=[name])
        else:
            df_tr[name] = slf_tr.flatten()
            df_unc[name] = slf_unc.flatten()

    #% %
    path = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results/{}-{}/".format(t0, t1)
    pd.to_pickle(df_tr, path + "SLF_ASL_trend_{}-{}.p".format(t0, t1))
    pd.to_pickle(df_unc, path + "SLF_ASL_temporal_unc_{}-{}.p".format(t0, t1))


#%% make final dataset


def quadrsum(X):
    Y = np.zeros((X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        Y = Y + (X[i] * X[i])
    Y = np.sqrt(Y)
    return Y


#% %
for period in periods:
    #% %
    print("period: {}".format(period))
    # period=periods[-1]
    t0 = period[0]
    t1 = period[1] - 1
    print("Make final dataset")
    path = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results/{}-{}/".format(t0, t1)
    df_temporal_unc = pd.read_pickle(
        path + "SLF_ASL_temporal_unc_{}-{}.p".format(t0, t1)
    )

    # ds_trend =xr.open_dataset(path+'SLF_trend_ALL_NM_OLS_{}-{}.nc'.format(t0,t1))
    # ds_trend=ds_trend.sel(nm='OLS')
    # trend=np.array(ds_trend.trend).reshape(len(ds_trend.name),len(ds_trend.lat)*len(ds_trend.lon))
    # df_trend = pd.DataFrame(trend.T,columns=np.array(ds_trend.name))

    df_trend = pd.read_pickle(path + "SLF_ASL_trend_{}-{}.p".format(t0, t1))

    # ds_intrinsic_unc = xr.open_dataset(path+'intrinsic_unc_{}-{}.nc'.format(t0,t1))
    # ds_intrinsic_unc = xr.open_dataset(path+'intrinsic_unc_source_trend_SLF{}-{}.nc'.format(t0,t1))
    ds_intrinsic_unc = xr.open_dataset(
        path + "intrinsic_unc_prop_{}-{}.nc".format(t0, t1)
    )
    names = [name for name in np.array(ds_intrinsic_unc.name)]
    for iname, name in enumerate(names):
        if name == "GLWS_JPL":
            names[iname] = "GLA_JPL"
    ds_intrinsic_unc["name"] = names
    ds_noise_model = xr.open_dataset(
        path + "source_trend_temporal_unc_{}-{}.nc".format(t0, t1)
    )
    # ds_noise_model = ds_noise_model.sortby('lat',ascending=False)
    df_spatial_unc = pd.read_pickle(path + "spatial_unc_{}-{}.p".format(t0, t1))

    path2 = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results/"
    file = "normalized_spatial_unc.p"
    df_spatial_unc_norm = pd.read_pickle(path2 + file)

    names = [name for name in df_trend.columns]
    lat = np.array(ds_intrinsic_unc.lat)
    lon = np.array(ds_intrinsic_unc.lon)
    dimlat = len(lat)
    dimlon = len(lon)
    trends = np.zeros((len(names), len(lat), len(lon)))
    unc_type = ["temporal", "spatial", "intrinsic"]
    uncs = np.zeros((len(unc_type), len(names), len(lat), len(lon)))
    nm_sel = np.full_like(trends, 0)
    spatial_unc = np.full_like(trends, 0)
    ic_idx = -1
    # this should be the same IC index as in the script 2.e.SLE_OM_hectorsource.py,
    # when we inputed the best trend and uncertainty in the SLE!

    for iname, name in enumerate(names):
        trends[iname, :, :] = np.array(df_trend[name]).reshape(dimlat, dimlon)
        # trends[iname,:,:]=np.array(ds_noise_model.sel(name=name).best_trend[ic_idx,:,:])
        nm_sel[iname, :, :] = np.array(
            ds_noise_model.sel(name=name).ranks[ic_idx, :, :]
        )

        reg = name.split("_")[0]
        spatial_unc[iname, :, :] = np.array(df_spatial_unc_norm[reg]).reshape(
            dimlat, dimlon
        )
        for iunc, unc in enumerate(unc_type):
            if unc == "temporal":
                uncs[iunc, iname, :, :] = np.abs(
                    np.array(df_temporal_unc[name]).reshape(dimlat, dimlon)
                )
                # uncs[iunc,iname,:,:]=np.abs(ds_noise_model.sel(name=name).best_unc[ic_idx,:,:])

            elif unc == "spatial":
                uncs[iunc, iname, :, :] = np.abs(
                    np.array(df_spatial_unc[name.split("_")[0]]).reshape(dimlat, dimlon)
                )
            else:  # unc=='intrinsic'
                if name in np.array(ds_intrinsic_unc.name):
                    # uncs[iunc,iname,:,:]=np.abs(np.array(ds_intrinsic_unc.sel(name=name).intrinsic_unc_SLF_trend))
                    uncs[iunc, iname, :, :] = np.abs(
                        np.array(ds_intrinsic_unc.sel(name=name).intrinsic_unc_SLF)
                    )

    #% % sum the uncertainities in quadrature:
    unc_total = np.sqrt((uncs[0] ** 2) + (uncs[2] ** 2) + (uncs[1] ** 2))
    # unc_total = uncs.sum(axis=0)

    #% % make dataset
    da = xr.Dataset(
        data_vars={
            "trend": (("name", "lat", "lon"), trends),
            "uncs": (("unc_type", "name", "lat", "lon"), uncs),
            "unc_total": (("name", "lat", "lon"), unc_total),
            "nm_sel": (("name", "lat", "lon"), nm_sel),
            "spatial_unc_norm": (("name", "lat", "lon"), spatial_unc),
        },
        coords={
            "lon": lon,
            "lat": lat,
            "name": names,
            "unc_type": unc_type,
            "nm": np.array(ds_noise_model.nm),
        },
    )

    # da.uncs[:,0,:,:].plot(col='unc_type',cmap='RdBu_r',vmin=-1,vmax=1)
    # plt.show()
    #% % save dataset
    da.to_netcdf(path + "final_dataset_ASL_{}-{}.nc".format(t0, t1))

    # % make and save combos
    print("Make combos")
    # ds=xr.open_dataset(path+'final_dataset_{}-{}.nc'.format(t0,t1))
    ds = da
    ds = ds.sortby("lat", ascending=False)

    lon = np.array(ds.lon)
    lat = np.array(ds.lat)
    llon, llat = np.meshgrid(lon, -lat)
    df = pd.DataFrame(np.hstack(llon), columns=["lon"])
    df["lat"] = np.hstack(llat)

    #% %
    combos = [
        ["JPL"],
        ["CSR"],
        ["IMB", "WGP"],
        ["IMB", "GWB", "ZMP"],
        ["UCI", "WGP"],
        ["UCI", "GWB", "ZMP"],
    ]
    reconstr = ["JPL", "CSR", "IMB+WGP", "IMB+GWB+ZMP", "UCI+WGP", "UCI+GWB+ZMP"]
    # i=0
    for title, combo in zip(reconstr, combos):

        #% %
        names = [
            name
            for name in np.array(ds.name)
            for comb in combo
            if name.split("_")[1] == comb
        ]
        regions = [
            name.split("_")[0]
            for name in np.array(ds.name)
            for comb in combo
            if name.split("_")[1] == comb
        ]
        # trend=np.nansum(np.array(ds.trend.sel(name=names)),axis=0)
        # unc=quadrsum(np.array(ds.unc_total.sel(name=names)))
        df["{}_trend_tot".format(title)] = np.hstack(
            np.nansum(np.array(ds.trend.sel(name=names)), axis=0)
        )
        df["{}_unc_tot".format(title)] = np.hstack(
            quadrsum(np.array(ds.unc_total.sel(name=names)))
        )
        # da=ds.sel(name=names)
        for i, unc_typ in enumerate(np.array(ds.unc_type)):
            # print(unc_typ)
            df["{}_unc_{}".format(title, unc_typ)] = quadrsum(
                np.array(ds.uncs[i].sel(name=names))
            ).flatten()
        for i, reg in enumerate(regions):
            df["{}_unc_{}".format(title, reg)] = quadrsum(
                np.array(ds.uncs.sel(name=names[i]))
            ).flatten()

        df.to_pickle(path + "OM_reconstructions_ASL_{}-{}.p".format(t0, t1))
