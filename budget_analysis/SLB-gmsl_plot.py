#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:07:16 2021

@author: ccamargo
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
# from utils_minisom import *
import utils_SL as sl

#%% open dataset
path = "/Users/ccamargo/Desktop/budget/data/"
file = "SLA_alt+comps_1993-2016_trends.nc"

ds = xr.open_dataset(path + file)
gmsl = np.array(ds.sla.mean(dim=("lat", "lon")))
time = np.array(ds.time)
#%% plot gmsl
plt.plot(time, gmsl)
plt.legend(np.array(ds.contribution))
plt.title("Global Mean Sea Level")
plt.xlabel("time")
plt.ylabel("mm")
plt.show()

#%% trend
tdec, _ = sl.get_dec_time(time)
for i in range(len(np.array(ds.contribution))):
    y = np.array(gmsl[:, i] - np.nanmean(gmsl[:, i]))
    trend, error, acc, trend_with_error, std_trend = sl.get_OLS_trend(tdec, y)
    print(np.array(ds.contribution[i]))
    print(trend)
    name = np.array(ds.contribution[i])
    plt.plot(y, label="{}".format(name))
    out = sl.get_ts_trend(tdec, y, plot=False)
    plt.plot(out[1], label=" {} trend: {}".format(name, np.round(out[0], 3)))
plt.legend()
plt.show()

#%%
path = "/Volumes/LaCie_NIOZ/reg/world/data/budget/"
files = [
    "MSLA_CMEMS_glb_merged_1993-2017_res1deg.nc",
    "sSLA_ENS_glb_1993-2017_res1deg.nc",
    "bSLA_ALL_glb_1993-2016_res1deg.nc",
]
ds = xr.open_dataset(path + files[0])
yy = np.array(ds.sla[0:288, :, :].mean(dim=("lat", "lon")))

path = "/Volumes/LaCie_NIOZ/data/total/"
file = "sla_alt_1993-2016.nc"
ds = xr.open_dataset(path + file)
tdec = np.array(ds.tdec)
time = np.array(ds.time)
for i in range(len(ds.name)):
    name = np.array(ds.name[i])
    y = (
        np.array(
            ds.sla[i, :, :, :]
            .where((ds.lat < 66) & (ds.lat > -66), np.nan)
            .mean(dim=("lat", "lon"))
        )
        * 1000
    )
    y = np.array(y - np.nanmean(y))
    out = sl.get_ts_trend(tdec, y, plot=False)
    plt.plot(y, label="{}".format(name))
    plt.plot(out[1], label=" {} trend: {}".format(name, np.round(out[0], 3)))
y = gmsl[:, 0] - np.nanmean(gmsl[:, 0])
out = sl.get_ts_trend(tdec, y, plot=False)
name = "CMEMS"
plt.plot(y, label="{}".format(name))
plt.plot(out[1], label=" {} trend: {}".format(name, np.round(out[0], 3)))

y = np.array(yy - np.nanmean(yy)) * 1000
out = sl.get_ts_trend(tdec, y, plot=False)
name = "CMEMS orig"
plt.plot(y, label="{}".format(name))
plt.plot(out[1], label=" {} trend: {}".format(name, np.round(out[0], 3)))
plt.ylabel("mm")
plt.xlabel("months since 1993")
plt.legend()
plt.show()
