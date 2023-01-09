#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:52:48 2022

@author: ccamargo
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. get filelist
path = "/Volumes/LaCie_NIOZ/data/steric/data/"
path_to_original_files = path + "original/"
flist = [file for file in os.listdir(path_to_original_files) if file.endswith(".nc")]

path_to_regrided_files = path + "regrid_180x360/"


#%% 2. Regrid:
# for file in flist:
#     fin=path_to_original_files+file
#     fout=path_to_regrided_files+file
#     command_list=str('cdo -L remapbil,r360x180 '+fin+' '+fout)
#     _tmp=os.system(command_list)
#%% landmask
ds = xr.open_dataset("/Volumes/LaCie_NIOZ/data/masks/ETOPO_mask.nc")
ds = ds.where((ds.lat > -66) & (ds.lat < 66), np.nan)
mask = np.array(ds.landmask)

ds = xr.open_dataset(
    "/Volumes/LaCie_NIOZ/data/barystatic/masks/"
    + "LAND_MASK_CRI-JPL_180x360_conservative.nc"
)
ds = ds.where((ds.lat > -66) & (ds.lat < 66), np.nan)
mask = np.array(ds.mask)
mask[mask == 1] = np.nan
mask[mask == 0] = 1
# %% 3. get data
flist = [file for file in os.listdir(path_to_regrided_files) if file.endswith(".nc")]
datasets = []
for file in flist:
    print(file)
    name = file.split(".nc")[0]
    ds = xr.open_dataset(path_to_regrided_files + file, decode_times=False)
    timespan = [ds.timespan]
    print(timespan)
    ti, tf = timespan[0].split(" to ")
    yf = int(tf.split("-")[0])
    mf = int(tf.split("-")[1])
    if mf == 12:
        yf = yf + 1
        mf = "01"
    else:
        mf = mf + 1
    tf = "{}-{}-28".format(yf, str(mf).zfill(2))
    if name == "Ishii":
        ti = "1990-01-31T00:00:00.000000"
        tf = "2019-01-31T00:00:00.000000"
        print("correct time: {} to {}".format(ti, tf))
    # tf = '{}-{}-{}'.format(time[-1].year,str(time[-1].month).zfill(2),time[-1].day +15)
    time = np.arange(ti, tf, dtype="datetime64[M]")
    ds["time"] = np.array(time)

    da = ds["data"].rename("sla_" + name)
    da.data = da.data * mask
    da.data = da.data - np.array(
        da.sel(time=slice("2005-01-01", "2016-01-01")).mean(dim="time")
    )
    datasets.append(da)
    # print(da)
#%% merge datasets
ds = xr.merge(datasets)
#% % select since 1993
ds = ds.sel(time=slice("1993-01-01", ds.time[-1]))
#% % compute ENS mean
var = [
    key
    for key in ds.variables
    if key.split("_")[0] == "sla" and len(key.split("_")) == 2
]
data = np.zeros((len(var), len(ds.time), len(ds.lat), len(ds.lon)))
data.fill(np.nan)
names = [v.split("_")[-1] for v in var]
for i, v in enumerate(var):
    data[i] = np.array(ds[v])
da = xr.Dataset(
    data_vars={"data": (("names", "time", "lat", "lon"), data)},
    coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time, "names": names},
)

# ds['sla_ens'] = (['time','lat','lon'],np.nanmean(datamu,axis=0))
ds["sla_ens"] = da.data.mean(dim="names")
ens = np.zeros((1, len(ds.time), len(ds.lat), len(ds.lon)))
ens.fill(np.nan)
ens[0] = np.array(ds.sla_ens)
data2 = np.vstack([data, ens])
names.append("ENS")
ds = ds.assign_coords({"names": names})
ds["SLA"] = (["names", "time", "lat", "lon"], data2)

ds.attrs["units"] = "meters"
ds.attrs["description"] = "Steric sea-level height (m)"
ds.attrs["time_mean"] = "Removed time mean from 2005-2015 (full years)"
ds.attrs["script"] = "SLB-steric.py"
#%% save
path_save = "/Volumes/LaCie_NIOZ/data/budget/"
ds.to_netcdf(path_save + "steric_upper.nc")
