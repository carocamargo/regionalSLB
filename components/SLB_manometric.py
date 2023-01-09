#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:01:18 2022

@author: ccamargo
"""


import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import utils_SLE_v2 as sle

# path = '/Volumes/LaCie_NIOZ/data/barystatic/regrid/'

# flist = [file for file in os.listdir(path) if file.startswith('GRA')]

# file=flist[0]
# ds = xr.open_dataset(path+file)
# ds.lwe_thickness[0,:,:].plot()


def from_trend_to_ts(trend, time):
    """
    Make time series from a linear trend
    1D or 3D time series

    """

    def timeseries(trend, time):
        """
        1D time series
        """
        A = np.ones((len(time), 2))
        A[:, 1] = np.array(time - np.mean(time))
        pred = np.array(A * trend)

        return pred[:, 1]

    if len(trend.shape) == 2:  # lat,lon
        dimlat, dimlon = trend.shape
        trend = trend.flatten()
        ts = np.zeros((len(time), dimlat * dimlon))
        for i in range(len(trend)):
            if trend[i]:
                ts[:, i] = timeseries(trend[i], time)
        ts = ts.reshape(len(time), dimlat, dimlon)
    else:
        ts = timeseries(trend, time)

    return ts


#%% get OM from mascons, and fingerprints
datasets = ["JPL", "CSR"]
dimlat = 180
dimlon = 360
periods = [(2003, 2016), (2005, 2015)]
OM_ts = np.zeros((len(datasets), 188, dimlat, dimlon))
OM = np.zeros((len(datasets), len(periods), dimlat, dimlon))
fing = np.zeros((len(datasets), len(periods), dimlat, dimlon))
for iname, name in enumerate(datasets):
    ds = xr.open_dataset(
        "/Volumes/LaCie_NIOZ/data/barystatic/use/"
        + "GRA-Mascon-{}_300kmbuf_sel_180x360.nc".format(name)
    )
    # print(ds.time)
    # ds= ds.sel(time=slice(to,ti))
    data = np.array(ds.ocean) * 10  # mm of water thickness
    # # Transform data to mm of SL:
    # for i in range(len(data)):
    #     data[i,:,:]=sle.EWH_to_height(data[i,:,:]).reshape(dimlat,dimlon)

    OM_ts[iname] = np.array(data)
#%%
da = xr.Dataset(
    data_vars={
        "manometric_sl": (("names", "time", "lat", "lon"), OM_ts),
        # 'fingerprint':(('names','lat','lon'),fing),
        # 'dynamic_sl':(('names','lat','lon'),dyn_SL)
    },
    coords={
        "names": datasets,
        "time": ds.time,
        "periods": ["2003-2016", "2005-2015"],
        "names_idx": np.arange(0, len(datasets)),
        "lat": ds.lat,
        "lon": ds.lon,
    },
)
#%%
for j, period in enumerate(periods):
    t0, t1 = period
    to = str(t0) + "-01-01"
    ti = str(t1) + "-12-31"
    path = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results/{}-{}/".format(t0, t1)
    file = "OM_reconstructions_ASL_{}-{}.p".format(t0, t1)
    df = pd.read_pickle(path + file)

    for iname, name in enumerate(datasets):
        ds = xr.open_dataset(
            "/Volumes/LaCie_NIOZ/data/barystatic/use/"
            + "GRA-Mascon-{}_300kmbuf_sel_180x360.nc".format(name)
        )
        # Select time

        ds = da.sel(time=slice(to, ti))
        tdec, _ = sl.get_dec_time(np.array(da.time))
        out = sl.get_reg_trend(tdec, data, np.array(ds.lat), np.array(ds.lon))
        OM[iname, j, :, :] = np.array(out[0])

        fing[iname, j, :, :] = np.array(df["{}_trend_tot".format(name)]).reshape(
            dimlat, dimlon
        )
        # deep = from_trend_to_ts(np.array(ds2.steric), tdec) # mm
#%%
da["manometric_trend"] = (["name", "periods", "lat", "lon"], OM)
da["barystatic_trend"] = (["name", "periods", "lat", "lon"], fing)

da["dynamic_trend"] = da["manometric_trend"] - da["barystatic_trend"]
path_dyn = "/Volumes/LaCie_NIOZ/data/dynamicREA/"
path = "/Volumes/LaCie_NIOZ/data/budget/"

da.attrs["units"] = "mm/yr"
da.attrs[
    "metadata"
] = "Dynamic SL computed by removing the fingerprints (barystatic SL) from the ocean mass signal of GRACE"
da.attrs["script"] = "SLB_manometric.py"
da.to_netcdf(path_dyn + "manometric_sl_GRACE.nc")
da.to_netcdf(path + "manometric_sl_GRACE.nc")

# da.to_netcdf('/Volumes/LaCie_NIOZ/data/sterodynamic/dyn_sl_GRACE_{}-{}.nc'.format(t0,t1))
