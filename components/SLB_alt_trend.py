#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:51:15 2022

@author: ccamargo
"""

# import pandas as pd
import numpy as np
import xarray as xr

# import sys
# sys.path.append("~/Documents/py_scripts/")
# import utils_SL as sl
import os
import cmocean as cm
import matplotlib.pyplot as plt

#%%
def sel_best_NM(IC, field, field2):
    """Given a field that has dimensions of len(nm),len(lat),len(lon),
    and an information critera that has the same dimensions, select the best noise model for this field.
    Return the selected field and the scoring of the noise models, both with dimensions (len(lat),len(lon))
    """
    dimnm, dimlat, dimlon = field.shape
    field = field.reshape(dimnm, dimlat * dimlon)
    field2 = field2.reshape(dimnm, dimlat * dimlon)

    mask = np.array(field[0, :])
    IC = IC.reshape(dimnm, dimlat * dimlon)

    best_field = np.zeros((dimlat * dimlon))
    best_field.fill(np.nan)
    score = np.full_like(best_field, np.nan)
    best_field2 = np.full_like(best_field, np.nan)

    for icoord in range(dimlat * dimlon):
        if np.isfinite(mask[icoord]):
            target = IC[:, icoord]
            ic = np.zeros((dimnm))
            logic = np.zeros((dimnm))

            for inm in range(dimnm):
                logic[inm] = np.exp((np.nanmin(target) - target[inm]) / 2)
                if logic[inm] > 0.5:
                    ic[inm] = 1
            score[icoord] = int(np.where(ic == np.nanmax(ic))[0][0])

            best_field[icoord] = field[int(score[icoord]), icoord]
            best_field2[icoord] = field2[int(score[icoord]), icoord]

    return (
        score.reshape(dimlat, dimlon),
        best_field.reshape(dimlat, dimlon),
        best_field2.reshape(dimlat, dimlon),
    )


#%%
def add_attrs(
    ds, variables=["lat", "lon"], latname="lat", lonname="lon", depthname="depth"
):
    if "lat" in variables:
        ds[latname].attrs = {
            "axis": "Y",
            "long_name": "latitude",
            "standard_name": "latitude",
            "unit_long": "degrees north",
            "units": "degrees_north",
        }
    if "lon" in variables:
        ds[lonname].attrs = {
            "axis": "X",
            "long_name": "longitude",
            "standard_name": "longitude",
            "step": 0.25,
            "unit_long": "degrees east",
            "units": "degrees_east",
        }
    if "depth" in variables:
        ds[depthname].attrs = {
            "axis": "Z",
            "long_name": "depth",
            "positive": "down",
            "standard_name": "depth",
            "unit_long": "meter",
            "units": "m",
        }

    return


#%%

from matplotlib.colors import ListedColormap

# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
col_dict = {
    1: "black",  # WN
    2: "palegoldenrod",  # PL
    3: "lightpink",  # PLWN
    4: "orange",  # AR1
    5: "teal",  # Ar5
    6: "darkmagenta",  # AR9
    7: "skyblue",  # ARf
    8: "crimson",  # GGM
}

# We create a colormar from our list of colors
cmapnm = ListedColormap([col_dict[x] for x in col_dict.keys()])
#%%
plot = False
new_vars = ["NM_score", "best_trend", "best_unc"]
variables = ["trend", "unc"]

periods = [(1993, 2017)]
datasets = ['ENS', 'csiro','cmems',
            # 'aviso',
            'measures', 'slcci'
            ]
inds = ["aic", "bic", "bic_c", "bic_tp"]
dimlat=180;dimlon=360
scores = np.zeros((len(datasets),len(periods), len(inds), dimlat, dimlon))
best_trends = np.full_like(scores, 0)
best_uncs = np.full_like(scores, 0)

for ip, period in enumerate(periods):
    t0, t1 = period

    path = "/Volumes/LaCie_NIOZ/data/altimetry/trends/{}-{}/".format(t0, t1)
    for iname,name in enumerate(datasets):
        flist = [file for file in os.listdir(path) if file.split("_")[1] == name]
        nm = []
        cmap = cm.cm.balance
        clim = 5000
        cmin = -clim
        cmax = clim
        cmap2 = cm.tools.crop(cmap, 0, cmax, 0)
        for file in flist:
            ds = xr.open_dataset(path + file)
            if plot:
                fig = plt.figure(dpi=300, figsize=(15, 5))
                ax = plt.subplot(121)
                ds = ds.where((ds.lat >= -66) & (ds.lat <= 66), np.nan)
                ds.trend.plot(vmin=cmin, vmax=cmax, ax=ax, cmap=cmap)
                # plt.show()
                ax = plt.subplot(122)
                ds.unc.plot(vmin=0, vmax=cmax, ax=ax, cmap=cmap2)
                plt.show()
            nm.append(np.array(ds.nm))
        if len(nm) != 8:
            print("Missing Noise Model !!!!!")
    
        #% % plottrend an unc for all noise models
        ds = xr.open_dataset(path + "ALT_{}.nc".format(name))
        ds = ds.where((ds.lat >= -66) & (ds.lat <= 66), np.nan)
        if plot:
            ds.trend.plot(col="nm", col_wrap=4, vmin=cmin, vmax=cmax, cmap=cmap)
            plt.show()
            ds.unc.plot(col="nm", col_wrap=4, vmin=0, vmax=cmax, cmap=cmap2)
            plt.show()
    
        #% % select best noise model
        
    
        for i, ind in enumerate(inds):
            score, best_trend, best_unc = sel_best_NM(
                np.array(ds[ind]), np.array(ds.trend), np.array(ds.unc)
            )
            #% %
            if plot:
                cmap = cm.cm.balance
                clim = 5
                cmin = -clim
                cmax = clim
                cmap2 = cm.tools.crop(cmap, 0, cmax, 0)
    
                plt.figure(figsize=(20, 10))
                nrow = 2
                ncol = 2
    
                plt.subplot(nrow, ncol, 1)
                plt.pcolor(score + 1, vmin=1, vmax=8, cmap=cmapnm)
                cbar = plt.colorbar(  # ticks=np.arange(0.5,len(ds.nm)+0.5),
                    ##shrink=0.95,
                    # label='Preferred Noise Model',fontsize=15,
                    orientation="vertical"
                )
                cbar.ax.set_yticklabels(nm, fontsize=15)
                plt.title("NM selection - {}".format(ind), fontsize=20)
    
                plt.subplot(nrow, ncol, 2)
                plt.pcolor(best_trend / 1000, vmin=cmin, vmax=cmax, cmap=cmap)
                cbar = plt.colorbar()
                cbar.set_label(label="mm/yr", fontsize=15)
                plt.title("Trend", fontsize=20)
    
                plt.subplot(nrow, ncol, 3)
                plt.pcolor(best_unc / 1000, vmin=0, vmax=cmax, cmap=cmap2)
                cbar = plt.colorbar()
                cbar.set_label(label="mm/yr", fontsize=15)
                plt.title("Unc", fontsize=20)
    
                plt.subplot(nrow, ncol, 4)
                plt.pcolor(
                    (ds.sel(nm="AR1").trend / 1000) - (best_trend / 1000),
                    vmin=-1,
                    vmax=1,
                    cmap=cm.cm.curl,
                )
                cbar = plt.colorbar()
                cbar.set_label(label="dif", fontsize=15)
                plt.title("AR1 trend- Best Trend", fontsize=20)
    
                plt.tight_layout()
                plt.show()
    
            scores[iname,ip, i] = score
            best_trends[iname,ip, i] = best_trend
            best_uncs[iname,ip, i] = best_unc

da = xr.Dataset(
    data_vars={
        "NM_score": (("names","periods", "ICs", "lat", "lon"), scores),
        "best_trend": (("names","periods", "ICs", "lat", "lon"), best_trends/1000),
        "best_unc": (("names","periods", "ICs", "lat", "lon"), best_uncs/1000),
    },
    coords={
        "ICs": inds,
        "lat": ds.lat,
        "lon": ds.lon,
        "nm": ds.nm,
        "names":datasets,
        "periods": ["{}-{}".format(period[0], period[1]) for period in periods],
    },
)
# for var in variables:
#     new_var = "{}_{}-{}".format(var, t0, t1)
#     da[new_var] = ds[var] / 1000
#     new_vars.append(new_var)
# #% %
# for var in new_vars:
#     if var == "NM_score":
#         da[var].attrs["units"] = "noise model number"
#     else:
#         da[var].attrs["units"] = "mm/yr"
da.attrs["metadata"] = "Altimetry Trends and uncertainties computed with Hector"


path_save = "/Volumes/LaCie_NIOZ/data/budget/trends/"

# da['best_trend'][:,0,0,:,:].plot(col='names',col_wrap=2,vmin=0,vmax=8)
da.to_netcdf(path_save + "alt.nc")

#%%
