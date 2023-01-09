#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:57:33 2021


Data Pre-processing for cluster analysis
- STERIC 
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

#% % get dataset we prepared for the budget, where steric and mass components
# are in the same matrix (so we dont need to deal with name issues)
path_original = "/Volumes/LaCie_NIOZ/reg/world/data/budget/"
file = "SLA_alt+comps_1993-2016_trends"
ds = xr.open_dataset(path_original + file + ".nc")
print(ds)
path_working = "/Volumes/LaCie_NIOZ/reg/world/data/"

lat_name = "lat"
lon_name = "lon"
lat_name_cdo = "lat"
lon_name_cdo = "lon"

var_name = "sla"
time_name = "time"
# loop over steric and barysatic time series
contrs = ["ste", "bar"]
for contr in contrs:
    ds2 = ds[var_name].sel(contribution=contr).to_dataset()
    # now SLA has dimensions time,lat,lon
    # save this file
    ofile = contr + "SL_1993-2016_res1deg"
    ds2.to_netcdf(path_working + ofile + ".nc")
    ds2.sla.mean(dim=["lat", "lon"]).plot()
    plt.title(ofile)
    plt.show()
    #% % 1. detrend
    ofile1 = ofile + "_detrend"
    dmaps.detrend(path_working, ofile, ofile1)
    da = xr.open_dataset(path_working + ofile1 + ".nc")
    da.sla.mean(dim=["lat", "lon"]).plot()
    plt.title(ofile1)
    plt.show()

    # 2. deseason
    ofile2 = ofile1 + "_deseason"
    dmaps.deseason(path_working, ofile1, ofile2)
    da = xr.open_dataset(path_working + ofile2 + ".nc")
    da.sla.mean(dim=["lat", "lon"]).plot()
    plt.title(ofile2)
    plt.show()

    # 3. apply 300km filter
    d = 300  # 300km filter
    d = d / 100
    res = 1
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
        da.sla.mean(dim=["lat", "lon"]).plot()
        plt.title(ofile3)
        plt.show()
