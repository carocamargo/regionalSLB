#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:41:01 2022

Dynamic SL part I:
    Ocean reanalysis
@author: ccamargo
"""


import numpy as np
import xarray as xr
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import matplotlib.pyplot as plt
import pandas as pd
import os

path_save = "/Volumes/SLB/calculations/dynamic_sl/"
#%%
dataset_name = "ARMOR3D"
path = "/Volumes/SLB/raw_data/steric/ARMOR3D/Data/"


flist = sl.get_filelist(path, ext="*.nc")
ds = xr.open_dataset(flist[0])
ds
var = "zo"
latname = "latitude"
lonname = "longitude"
depth = 0
time = 0
ds[var][time, depth, :, :].plot()
plt.show()
print(ds[var][time, depth, :, :].mean(dim=(latname, lonname)))

data = np.array(ds[var][time, depth, :, :])
mask = sl.get_dry_msk(data)  # 0 for land, 1 for ocean
# Get ocean area based on mask:
lat = np.array(ds[latname])
lon = np.array(ds[lonname])
surf, area = sl.get_ocean_area(lat, lon, mask, plot=False, info=False)
surf_m = surf * 1000000  # m2
area_m = area * 1000000  # m2

da = xr.Dataset(
    data_vars={"data": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
)
mu = (da.data * area_m * mask).sum(dim=("lat", "lon")) / surf_m
print(mu)
glb = np.array(mu.data)

da = xr.Dataset(
    data_vars={"data": (("lat", "lon"), data - glb)}, coords={"lat": lat, "lon": lon}
)
da.data.plot()
plt.show()
#%%
dataset_name = "ARMOR3D"
path = "/Volumes/SLB/raw_data/steric/ARMOR3D/Data/"
flist = sl.get_filelist(path, ext="*.nc")
print(xr.open_dataset(flist[0]))
var = "zo"
latname = "latitude"
lonname = "longitude"
timename = "time"
idepth = 0
ds = xr.open_dataset(flist[0])
londim = len(ds[lonname])
latdim = len(ds[latname])

subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}

dic = {dataset_name: subdic}

path = "/Volumes/SLB/raw_data/steric/GREP/v2/data/"
dataset_name = "foam"
flist = sl.get_filelist(path, ext="*.nc")
print(xr.open_dataset(flist[0]))
var = "zos_" + dataset_name
latname = "latitude"
lonname = "longitude"
timename = "time"
ds = xr.open_dataset(flist[0])
londim = len(ds[lonname])
latdim = len(ds[latname])

idepth = None
subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}
dic[dataset_name] = subdic

dataset_name = "oras"
var = "zos_" + dataset_name
subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}
dic[dataset_name] = subdic

dataset_name = "glor"
var = "zos_" + dataset_name
subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}
dic[dataset_name] = subdic

dataset_name = "cglo"
var = "zos_" + dataset_name
subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}
dic[dataset_name] = subdic

path = "/Volumes/SLB/raw_data/steric/SODA/Data_v3_3_2/ocean/"
dataset_name = "soda"
flist = sl.get_filelist(path, ext="*.nc")
print(xr.open_dataset(flist[0]))
var = "ssh"
latname = "yt_ocean"
lonname = "xt_ocean"
timename = "time"
ds = xr.open_dataset(flist[0])
londim = len(ds[lonname])
latdim = len(ds[latname])

idepth = None
subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}
dic[dataset_name] = subdic

path = "/Volumes/SLB/raw_data/steric/SODA/Data_v3_4_2/monthly/"
dataset_name = "soda_342"
flist = sl.get_filelist(path, ext="*.nc")
print(xr.open_dataset(flist[0]))
var = "ssh"
latname = "yt_ocean"
lonname = "xt_ocean"
timename = "time"
ds = xr.open_dataset(flist[0])
londim = len(ds[lonname])
latdim = len(ds[latname])

idepth = None
subdic = {
    "dataset": dataset_name,
    "path": path,
    "flist": flist,
    "var": var,
    "latname": latname,
    "lonname": lonname,
    "timename": timename,
    "idepth": idepth,
    "latdim": latdim,
    "londim": londim,
}
dic[dataset_name] = subdic
#%%
def gmsl(data, lat, lon):
    mask = sl.get_dry_msk(data)  # 0 for land, 1 for ocean
    # Get ocean area based on mask:

    surf, area = sl.get_ocean_area(lat, lon, mask, plot=False, info=False)
    surf_m = surf * 1000000  # m2
    area_m = area * 1000000  # m2

    da = xr.Dataset(
        data_vars={"data": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
    )
    mu = (da.data * area_m * mask).sum(dim=("lat", "lon")) / surf_m
    glb = np.array(mu.data)
    return glb


#%%
datasets = [
    "ARMOR3D",
    "soda",
    "soda_342",
    "cglo",
    "foam",
    "oras",
    "glor",
]
for dataset in datasets:
    print(dataset)
    local_dic = dic[dataset]
    dataset_name = local_dic["dataset"]
    path = local_dic["path"]
    flist = local_dic["flist"]
    latname = local_dic["latname"]
    lonname = local_dic["lonname"]
    timename = local_dic["timename"]
    idepth = local_dic["idepth"]
    var = local_dic["var"]
    print(var)
    print(flist[0])
    latdim = local_dic["latdim"]
    londim = local_dic["londim"]

    #% %
    if "soda" in dataset:
        dyn_sl = np.zeros((len(flist) * 12, latdim, londim))
        time = np.zeros((len(flist) * 12))
    else:
        dyn_sl = np.zeros((len(flist), latdim, londim))
        time = np.zeros((len(flist)))
    ssh = np.full_like(dyn_sl, 0)
    glb = np.full_like(time, 0)

    j = 0
    for i, file in enumerate(flist):
        ds = xr.open_dataset(file)
        if "soda" in dataset:
            for itime in range(len(ds[timename])):
                ssh[j, :, :] = np.array(ds[var][itime, :, :])
                lat = np.array(ds[latname])
                lon = np.array(ds[lonname])
                glb[j] = gmsl(ssh[j, :, :], lat, lon)

                dyn_sl[j, :, :] = np.array(ssh[j, :, :] - glb[j])
                time[j] = np.array(ds[timename][itime])
                j = j + 1
        else:
            itime = 0
            if idepth == 0:
                ssh[i, :, :] = np.array(ds[var][itime, idepth, :, :])
            else:
                ssh[i, :, :] = np.array(ds[var][itime, :, :])
            lat = np.array(ds[latname])
            lon = np.array(ds[lonname])
            glb[i] = gmsl(ssh[i, :, :], lat, lon)

            dyn_sl[i, :, :] = np.array(ssh[i, :, :] - glb[i])
            time[i] = np.array(ds[timename])
    #% %
    da = xr.Dataset(
        data_vars={
            "dyn_sl": (("time", "lat", "lon"), dyn_sl),
            "ssh": (("time", "lat", "lon"), ssh),
            "gmsl": (("time"), glb),
        },
        coords={"time": time, "lat": ds[latname].data, "lon": ds[lonname].data},
    )
    da.ssh.attrs = ds[var].attrs
    da["lat"].attrs = ds[latname].attrs
    da["lon"].attrs = ds[lonname].attrs

    da.dyn_sl.attrs["long_name"] = "dynamic sea-level height"
    da.dyn_sl.attrs["standard_name"] = "dynamic_height"
    if "soda" in dataset:
        da.dyn_sl.attrs["unit_long"] = ds[var].attrs["units"]
    else:
        da.dyn_sl.attrs["unit_long"] = ds[var].attrs["unit_long"]
    da.dyn_sl.attrs["units"] = ds[var].attrs["units"]
    da.attrs[
        "metadata"
    ] = "Dynamic sea-level height, calculated from sea-surface height by uniformly subtracting its time-varying global mean."
    da.attrs["source"] = dataset_name
    da.attrs["script"] = "dyn_sl.py"
    da.to_netcdf(path_save + dataset_name + ".nc")

#%% plot gmsl
flist = sl.get_filelist(path_save, ext="*.nc")
for f in flist:
    ds = xr.open_dataset(f)
    plt.figure()
    ds.gmsl.plot()
    plt.title(f.split("/")[-1])
    plt.show()
#%% regrid - make template
lon_name = "lon"
lat_name = "lat"
# to regrid later, we create a template
left_lon = 0
right_lon = 360
lower_lat = -89.5
upper_lat = 90
# left_lon = np.array(ds[lon_name].min())
# right_lon = np.array(ds[lon_name].max())
# if np.round(right_lon)==360:right_lon=360
# lower_lat = np.array(ds[lat_name].min())
# upper_lat = np.array(ds[lat_name].max())
command = "cdo -f nc -sellonlatbox,{},{},{},{} -random,r360x180 {}template.nc".format(
    left_lon, right_lon, lower_lat, upper_lat, path_save + "/regrid/"
)
os.system(command)
#%% regrid
for f in flist:
    fname = f.split("/")[-1].split(".nc")[0]
    #% % 4. regrid to 1 degree
    res = 1
    infile = path_save + fname + ".nc"
    ofile = path_save + "regrid/" + fname + "_res1deg.nc"
    command = "cdo -remapbil,{}template.nc {} {}".format(
        path_save + "regrid/", infile, ofile
    )
    os.system(command)

#%% correct time
filepath = path_save + "regrid/"
flist = [
    file
    for file in os.listdir(filepath)
    if not file.startswith("temp") and not file.startswith(".")
]


start = [1993, 1990, 1990, 1993, 1993, 1993, 1993]
end = [2020, 2017, 2017, 2019, 2019, 2019, 2019]
datasets = [
    "ARMOR3D",
    "soda",
    "soda_342",
    "cglo",
    "foam",
    "oras",
    "glor",
]
for i, file in enumerate(datasets):
    print(file)
    # time = pd.date_range(start='{}-01-01'.format(start[i]),
    #                      end='{}-01-01'.format(end[i]+1),
    #                      freq='M')
    # print(len(time))

    ds = xr.open_dataset(filepath + file + "_res1deg.nc")
    flist.remove(file + "_res1deg.nc")
    ds["time"] = pd.date_range(
        start="{}-01-01".format(start[i]), end="{}-01-01".format(end[i] + 1), freq="M"
    )
    ds.to_netcdf(path_save + "regrid_cor_time/" + file + ".nc")
    # print(file)
    # print(len(ds.time))

#%% make one file with all datasets
filepath = path_save + "regrid_cor_time/"
for i, file in enumerate(datasets):
    print(file)
    ds = xr.open_dataset(filepath + file + ".nc")
    print(len(ds.lat))
    print(ds.lat.min())
    print(ds.lat.max())
flist = [
    file
    for file in os.listdir(filepath)
    if not file.startswith("temp") and not file.startswith(".")
]

# longest time series since 1993 is from ARMOR (goes untuil dec 2020)
time = np.array(
    pd.date_range(
        start="{}-01-01".format(start[0]), end="{}-01-01".format(end[0] + 1), freq="M"
    )
)
dyn_sl = np.zeros((len(time), len(ds.lat), len(ds.lon), len(datasets)))
dyn_sl.fill(np.nan)
ssh = np.full_like(dyn_sl, np.nan)
gmsl = np.zeros((len(time), len(datasets)))
gmsl.fill(np.nan)

for i, file in enumerate(datasets):
    print(file)
    ds = xr.open_dataset(filepath + file + ".nc")
    # Select time
    t0 = 1993

    t1 = int(max(ds.groupby("time.year").max().year)) + 1
    to = str(t0) + "-01-01"
    ti = str(t1) + "-01-01"
    ds = ds.sel(time=slice(to, ti))

    dyn_sl[0 : len(ds.time), :, :, i] = np.array(ds.dyn_sl)
    ssh[0 : len(ds.time), :, :, i] = np.array(ds.ssh)
    gmsl[0 : len(ds.time), i] = np.array(ds.gmsl)
    #% %
da = xr.Dataset(
    data_vars={
        "ssh_anom": (("time", "lat", "lon", "name"), dyn_sl),
        "ssh": (("time", "lat", "lon", "name"), ssh),
        "gmsl": (("time", "name"), gmsl),
    },
    coords={"time": time, "lat": ds.lat, "lon": ds.lon, "name": datasets},
)
da.attrs[
    "metadata"
] = "SSH from reanalysis, and SSH anomaly computed by removing the time-varying global mean (gmsl variable)"
da.attrs["script"] = "dyn_sl.py"
da.to_netcdf("/Volumes/LaCie_NIOZ/data/dynamicREA/ssh_rea.nc")

#
