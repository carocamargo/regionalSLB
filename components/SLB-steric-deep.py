#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:11:28 2022

Steric deep

@author: ccamargo
"""
# libraries
import xarray as xr
import os

#%%  functions
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


#%% PJ  - regrid

path = "/Volumes/LaCie_NIOZ/data/steric/deep_ocean/v2/"
file = "PJ.nc"
ds = xr.open_dataset(path + file)
ds.pj.plot()

path = "/Volumes/LaCie_NIOZ/data/steric/PJ/"
file = "PJ_updated"
ds = xr.open_dataset(path + file + ".nc")
ds.SLR[0, :, :].plot()  # 361x721
# ds.pj.plot()

add_attrs(ds)
infile = path + file + "_with_attrs"
ds.to_netcdf(infile + ".nc")

outfile = infile + "_180x360.nc"
command = "cdo -L remapbil,r360x180 {} {}".format(infile + ".nc", outfile)
os.system(command)
#%% Save PJ
path = "/Volumes/LaCie_NIOZ/data/steric/PJ/"
file = "PJ_updated"
infile = path + file + "_with_attrs"
outfile = infile + "_180x360.nc"
ds = xr.open_dataset(outfile)
ds.SLR[0, :, :].plot()
ds.SLR.sum(dim="depth").plot()
ds.SLR.mean(dim=("lat", "lon")).plot()
#% %
xd = xr.Dataset(
    data_vars={
        "steric": (("lat", "lon"), ds.SLR[0]),
        "steric_err": (("lat", "lon"), ds.SLR[0]),
    },
    coords={
        "lat": ds.lat,
        "lon": ds.lon,
    },
)
xd.attrs["metadata"] = "Deep Steric (>2000m) from Purkey & Johnson"
# xd.attrs["correction"] = 'Integrated value from 2000-6000m (before was only at 2000m)'
xd.attrs["source"] = "Update from Purkey & Johnson "
xd.attrs["timespan"] = "1990-2018 (full years)"
xd.to_netcdf("/Volumes/LaCie_NIOZ/data/steric/deep_ocean/" + "PJ.nc")

#%% chang
# import scipy.io

# path = "/Volumes/LaCie_NIOZ/data/steric/deep_ocean/Chang2019/"
# file = "TemperatureBelow2000mFitted2005_201907"
# mat = scipy.io.loadmat(path + file + ".mat")  # data in meters
# print(mat.keys())
# print([key for key in mat.keys() if not key.startswith("_")])
# # gpans=np.array(mat['gpans']) # depth,lat,lon,time


#%% Chang
