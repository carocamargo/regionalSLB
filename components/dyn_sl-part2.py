#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  3 10:50:57 2022

@author: ccamargo
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

path = "/Volumes/LaCie_NIOZ/data/dynamicREA/"
#%% compute ensemble
ds = xr.open_dataset(path + "ssh_rea.nc")
# drop armor because it's not a reanalysis
names = [
    name
    for name in np.array(ds.name)
    if not name == "ARMOR3D" and not name == "soda_342"
]
ds = ds.sel(name=names)
min_lat = -66
max_lat = 66
mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
# da = ds.where(mask_lat, drop=True) # remove anything out of our mask # dimension of latitude is 132
da = ds.where(
    mask_lat
)  # add nans on anything out of the mask # dimension of latitude is 180


dyn_sl = np.zeros((len(da.time), len(da.lat), len(da.lon), len(da.name) + 1))
ssh = np.zeros((len(da.time), len(da.lat), len(da.lon), len(da.name) + 1))
gmsl = np.zeros((len(da.time), len(da.name) + 1))

dyn_sl[:, :, :, 0 : len(da.name)] = np.array(da.ssh_anom)
ssh[:, :, :, 0 : len(da.name)] = np.array(da.ssh)
gmsl[:, 0 : len(da.name)] = np.array(ds.gmsl)

dyn_sl[:, :, :, len(da.name)] = np.array(da.ssh_anom.mean(dim="name"))
ssh[:, :, :, len(da.name)] = np.array(da.ssh.mean(dim="name"))
gmsl[:, len(ds.name)] = np.array(ds.gmsl.mean(dim="name"))
names.extend(["ens"])

#%% make dataset
xa = xr.Dataset(
    data_vars={
        "ssh_anom": (("time", "lat", "lon", "name"), dyn_sl),
        "ssh": (("time", "lat", "lon", "name"), ssh),
        "gmsl": (("time", "name"), gmsl),
    },
    coords={"time": da.time, "lat": da.lat, "lon": da.lon, "name": names},
)
#%% check ensemble
xa.ssh[-1, :, :, 1].plot(vmin=-10, vmax=10, cmap="RdBu_r")
plt.show()

# the last year is all NaNs (only Armor went until 2020, and we have removed it)
# let's remove it then:
xa = xa.sel(time=slice(None, "2020-01-01"))

#%% check global means
for i in range(len(xa.name)):
    plt.plot((xa.gmsl[:, i]), label=np.array(xa.name[i]))
    plt.legend()
    plt.show()
#%% remove sudden drop from c-glors
# make empty
dyn_sl = np.zeros((len(xa.time), len(da.lat), len(da.lon), len(da.name) + 1))
ssh = np.zeros((len(xa.time), len(da.lat), len(da.lon), len(da.name) + 1))
gmsl = np.zeros((len(xa.time), len(da.name) + 1))
# get data
dyn_sl[:, :, :, 0 : len(da.name)] = np.array(xa.ssh_anom[:, :, :, 0 : len(da.name)])
ssh[:, :, :, 0 : len(da.name)] = np.array(xa.ssh[:, :, :, 0 : len(da.name)])
gmsl[:, 0 : len(da.name)] = np.array(xa.gmsl[:, 0 : len(da.name)])

# correct cglors
i = 1
print(names[i])
plt.plot(xa.gmsl[:, i])
target = np.array(xa.gmsl[:, i])
ind = [ind for ind in np.where(target < -1)[0]]
ind.extend([317])
target[ind] = np.nan
plt.plot(target)
target[317] = np.nan
plt.plot(target)

gmsl[:, i] = target
ssh[ind, :, :, i] = np.nan
dyn_sl[ind, :, :, i] = np.nan

xd = xr.Dataset(
    data_vars={
        "dyn_sl": (("time", "lat", "lon", "name"), dyn_sl[:, :, :, 0 : len(da.name)]),
        "ssh": (("time", "lat", "lon", "name"), ssh[:, :, :, 0 : len(da.name)]),
        "gmsl": (("time", "name"), gmsl[:, 0 : len(da.name)]),
    },
    coords={
        "time": xa.time,
        "lat": da.lat,
        "lon": da.lon,
        "name": names[0 : len(da.name)],
    },
)
dyn_sl[:, :, :, len(da.name)] = np.array(xd.dyn_sl.mean(dim="name"))
ssh[:, :, :, len(da.name)] = np.array(xd.ssh.mean(dim="name"))
gmsl[:, len(ds.name)] = np.array(xd.gmsl.mean(dim="name"))
plt.plot(gmsl[:, len(ds.name)])

#%% make new dataset
xa = xr.Dataset(
    data_vars={
        "ssh_anom": (("time", "lat", "lon", "name"), dyn_sl),
        "ssh": (("time", "lat", "lon", "name"), ssh),
        "gmsl": (("time", "name"), gmsl),
    },
    coords={"time": xa.time, "lat": da.lat, "lon": da.lon, "name": names},
)
for i in range(len(xa.name)):
    plt.plot((xa.gmsl[:, i]), label=np.array(xa.name[i]))
    plt.legend()
    plt.show()

xa.to_netcdf(path + "ssh_rea_corrected.nc")


#%%
#%%
