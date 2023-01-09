#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Pre-processing for cluster analysis
1. detrend
2. deseason
3. filter
4. regrid

Created on Fri Sep 24 14:55:33 2021

@author: ccamargo
"""


import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_dMAPS as dmaps
import utils_minisom as ms
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
from cartopy import crs as ccrs  # , feature as cfeature
import os

#%% get original data and copy to SOM folder:
path_original = "/Volumes/LaCie_NIOZ/data/altimetry/world/CMEMS/"
path_working = "/Volumes/LaCie_NIOZ/reg/world/data/"
file = "MSLA_CMEMS_glb_merged_1993-2019"

command = "cp {}{}.nc {}{}.nc".format(path_original, file, path_working, file)
os.system(command)
#%% open dataset and print some details:
ds = xr.open_dataset(path_working + file + ".nc")
print(ds.title)
print(
    "Time resolution : {} from {} to {} total of {} timesteps".format(
        ds.time_coverage_resolution,
        np.array(ds.time[0]),
        np.array(ds.time[-1]),
        len(ds.time),
    )
)
print(
    "Spatial resolutions: {} degrees \n longitude {} to {} \n latitude {} to {} ".format(
        ds.geospatial_lon_resolution,
        np.array(ds.longitude.min()),
        np.array(ds.longitude.max()),
        np.array(ds.latitude.min()),
        np.array(ds.latitude.max()),
    )
)
# ds.sla[0,:,:].plot()
ds.sla.mean(dim=("latitude", "longitude")).plot()
plt.title(file)
plt.show()
#%% data pre-processing:

lat_name = "latitude"
lon_name = "longitude"
lat_name_cdo = "lat"
lon_name_cdo = "lon"

var_name = "sla"
time_name = "time"
ifile = file

#% % 1. detrend
ofile1 = ifile + "_detrend"
dmaps.detrend(path_working, ifile, ofile1)
da = xr.open_dataset(path_working + ofile1 + ".nc")
da.sla.mean(dim=("latitude", "longitude")).plot()
plt.title(ofile1)
plt.show()

#% % 2. deseason
ofile2 = ofile1 + "_deseason"
dmaps.deseason(path_working, ofile1, ofile2)
da = xr.open_dataset(path_working + ofile2 + ".nc")
da.sla.mean(dim=("latitude", "longitude")).plot()
plt.title(ofile2)
plt.show()

# 3. apply 300km filter
d = 300  # 300km filter
d = d / 100
res = np.array(ds.geospatial_lon_resolution)
sig = int(1 / res)
#  apply to deseason and detrended dataset
files = [ofile1, ofile2]
for file in files:
    print("smoothing {}".format(file))
    dataset = xr.open_dataset(path_working + file + ".nc")
    data_in = np.array(dataset[var_name])
    ofile3 = file + "_{}kmfilter".format(int(d * 100))
    data_filtered = np.full_like(data_in, 0)
    for i in np.arange(len(dataset[time_name])):
        data_filtered[i, :, :] = ms.gaus_filter(data_in[i, :, :], sigma=sig * d)
    da = xr.Dataset(
        data_vars={var_name: ((time_name, lat_name, lon_name), data_filtered)},
        coords={
            time_name: ds[time_name],
            lon_name: ds[lon_name],
            lat_name: ds[lat_name],
        },
    )
    da.attrs = dataset.attrs
    da.attrs[
        "post-processing"
    ] = "SLA detrended, (deseasonalized), and  300km filtered."

    da.to_netcdf(path_working + ofile3 + ".nc")

    da.sla.mean(dim=("latitude", "longitude")).plot()
    plt.title(ofile3)
    plt.show()

#%% 4. regrid to 1 degree
res = 1
# a. first create a template of require lat/lon
left_lon = np.array(ds[lon_name].min())
right_lon = np.array(ds[lon_name].max())
if np.round(right_lon) == 360:
    right_lon = 360
lower_lat = np.array(ds[lat_name].min())
upper_lat = np.array(ds[lat_name].max())
command = "cdo -f nc -sellonlatbox,{},{},{},{} -random,r360x180 {}template.nc".format(
    left_lon, right_lon, lower_lat, upper_lat, path_working
)
os.system(command)
# b. get list of the 300km filters
flist = [
    filename
    for filename in os.listdir(path_working)
    if filename.endswith("300kmfilter.nc") and filename.startswith("MSLA")
]
# c. then use the template as the remap grid type
for file in flist:
    print("regridding {}".format(file))
    ifile = path_working + file
    ofile = path_working + file.split(".nc")[0] + "_res{}deg.nc".format(res)

    command = "cdo -remapbil,{}template.nc {} {}".format(path_working, ifile, ofile)
    os.system(command)
    da = xr.open_dataset(ofile)

    da.sla.mean(dim=("lat", "lon")).plot()
    plt.title(ofile)
    plt.show()
    path2 = "/Users/ccamargo/Desktop/budget/data/"
    da.to_netcdf(path2 + file.split(".nc")[0] + "_res{}deg.nc".format(res))


#%% select rom 1993-2016:
t0 = 2017
path = "/Volumes/LaCie_NIOZ/reg/world/data/"
files = [
    "MSLA_CMEMS_glb_merged_1993-2019_detrend_deseason_300kmfilter_res1deg.nc",
    "MSLA_CMEMS_glb_merged_1993-2019_detrend_300kmfilter_res1deg.nc",
]
for file in files:
    ds = xr.open_dataset(path + file)
    ds2 = ds.where(ds["time.year"] < t0, drop=True)
    ofile = file.split("2019")[0] + str(t0 - 1) + file.split("2019")[1]
    ds2.to_netcdf(path + ofile)
    ds2.sla.mean(dim=("lat", "lon")).plot()
    plt.title(ofile)
