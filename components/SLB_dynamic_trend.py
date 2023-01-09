#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:34:50 2022

@author: ccamargo
"""


# import pandas as pd
import numpy as np
import xarray as xr

import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")
# import utils_SL as sl
# import utils_SLB
# import os
import cmocean as cm
import matplotlib.pyplot as plt

from utils_SLB import sel_best_NM, make_cmapnm
cmapnm = make_cmapnm()

#%%
plot = True
new_vars = ["NM_score", "best_trend", "best_unc"]
variables = ["trend", "unc"]

periods = [(1993, 2017)]
for ip, period in enumerate(periods):
    t0, t1 = period

    path = "/Volumes/LaCie_NIOZ/data/dynamicREA/trends/{}-{}/".format(t0, t1)
    #% % plottrend an unc for all noise models
    ds = xr.open_dataset(path + "ens_dyn_v1.nc")
    ds = ds.where((ds.lat >= -66) & (ds.lat <= 66), np.nan)
    clim=5
    cmin=-clim;cmax=clim
    cmap=cm.cm.balance
    cmap2 = cm.tools.crop(cmap, 0, cmax, 0)

    if plot:
        ds.trend.plot(col="nm", col_wrap=4, vmin=cmin, vmax=cmax, cmap=cmap)
        plt.show()
        ds.unc.plot(col="nm", col_wrap=4, vmin=0, vmax=cmax, cmap=cmap2)
        plt.show()
    nm = np.array(ds.nm)
    #% % select best noise model
    inds = ["aic", "bic", "bic_c", "bic_tp"]
    scores = np.zeros((len(periods), len(inds), len(ds.lat), len(ds.lon)))
    best_trends = np.full_like(scores, 0)
    best_uncs = np.full_like(scores, 0)

    for i, ind in enumerate(inds):
        score, best_trend, best_unc = sel_best_NM(
            np.array(ds[ind]), np.array(ds.trend), np.array(ds.unc)
        )
        #% %
        if plot:

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
            plt.pcolor(best_trend, vmin=cmin, vmax=cmax, cmap=cmap)
            cbar = plt.colorbar()
            cbar.set_label(label="mm/yr", fontsize=15)
            plt.title("Trend", fontsize=20)

            plt.subplot(nrow, ncol, 3)
            plt.pcolor(best_unc , vmin=0, vmax=cmax, cmap=cmap2)
            cbar = plt.colorbar()
            cbar.set_label(label="mm/yr", fontsize=15)
            plt.title("Unc", fontsize=20)

            plt.subplot(nrow, ncol, 4)
            plt.pcolor(
                (ds.sel(nm="AR1").trend) - (best_trend),
                vmin=-1,
                vmax=1,
                cmap=cm.cm.curl,
            )
            cbar = plt.colorbar()
            cbar.set_label(label="dif", fontsize=15)
            plt.title("AR1 trend- Best Trend", fontsize=20)

            plt.tight_layout()
            plt.show()

        scores[ip, i] = score
        best_trends[ip, i] = best_trend
        best_uncs[ip, i] = best_unc

    #% % add data to dataset
    if ip == 0:
        da = xr.Dataset(
            data_vars={
                "NM_score": (("periods", "ICs", "lat", "lon"), scores),
                "best_trend": (("periods", "ICs", "lat", "lon"), best_trends),
                "best_unc": (("periods", "ICs", "lat", "lon"), best_uncs),
            },
            coords={
                "ICs": inds,
                "lat": ds.lat,
                "lon": ds.lon,
                "nm": ds.nm,
                "periods": ["{}-{}".format(period[0], period[1]) for period in periods],
            },
        )
    for var in variables:
        new_var = "{}_{}-{}".format(var, t0, t1)
        da[new_var] = ds[var] / 1000
        new_vars.append(new_var)
#% %
for var in new_vars:
    if var == "NM_score":
        da[var].attrs["units"] = "noise model number"
    else:
        da[var].attrs["units"] = "mm/yr"
da.attrs["metadata"] = "Dynamic REA ENS Trends and uncertainties computed with Hector"


path_save = "/Volumes/LaCie_NIOZ/data/budget/trends/"
da.to_netcdf(path_save + "dynamic.nc")

#%%
