#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:43:18 2022

make one file with all altimetry datasets 

Modified on Mon Nov 21 2022: 
    - CSIRO without GIA trend
    - not include AVISO in Ensemble (same as CMEMS)

@author: ccamargo
"""

import os
import xarray as xr
import numpy as np
import pandas as pd

path_save = "/Volumes/LaCie_NIOZ/data/altimetry/"

#%% get datasets
flist = [file for file in os.listdir(path_save + "regrid/") if file.endswith(".nc")]
flist = [file for file in flist  if file.split('.')[0]!='aviso']
datasets = []

for file in flist:
    print(file)
    name = file.split(".nc")[0]
    ds = xr.open_dataset(path_save + "regrid/" + file)
    if name == "measures":
        var = ["sla", "sla_err"]
    elif name == "csiro":
        var = ["height"]
    else:
        var = ["sla"]

    time = pd.to_datetime(np.array(ds.time))
    ti = "{}-{}-{}".format(time[0].year, str(time[0].month).zfill(2), time[0].day)
    yf = time[-1].year
    if time[-1].month == 12:
        yf = time[-1].year + 1
        mf = "01"
    else:
        mf = time[-1].month + 1
    tf = "{}-{}-28".format(yf, str(mf).zfill(2))

    # tf = '{}-{}-{}'.format(time[-1].year,str(time[-1].month).zfill(2),time[-1].day +15)
    time = np.arange(ti, tf, dtype="datetime64[M]")
    ds["time"] = np.array(time)

    for v in var:
        if v == "height":
            da = ds[v].rename("sla_" + name)
        else:
            da = ds[v].rename(v + "_" + name)
        datasets.append(da)
        print(da)
#%% merge datasets

ds = xr.merge(datasets)
#%% select since 1993
ds = ds.sel(time=slice("1993-01-01", ds.time[-1]))
#%% compute ENS mean
var = [
    key
    for key in ds.variables
    if key.split("_")[0] == "sla" and len(key.split("_")) == 2
]
data = np.zeros((len(var), len(ds.time), len(ds.lat), len(ds.lon)))
data.fill(np.nan)
for i, v in enumerate(var):
    if ds[v].units == "m":
        data[i] = np.array(ds[v]) * 1000
    else:
        data[i] = np.array(ds[v])

names = [v.split("_")[-1] for v in var]
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
ds.SLA.attrs["units"] = "mm"
#%% save
path_save = "/Volumes/LaCie_NIOZ/data/budget/"
ds.to_netcdf(path_save + "alt.nc")
path_save = "/Volumes/LaCie_NIOZ/data/budget/ts/"
ds.to_netcdf(path_save + "alt.nc")
path_save = "/Volumes/LaCie_NIOZ/data/altimetry/"
ds.to_netcdf(path_save + "alt.nc")


