#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:32:02 2021

@author: ccamargo
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
# import utils_SL as sl
import utils_dMAPS as dmaps

#%%
path = "/Users/ccamargo/Desktop/budget/regions/deltamaps/"
path = '/Volumes/LaCie_NIOZ/budget/regions/dmaps/k'
folder = "5"

field = np.load(path + folder + "/domain_identification/domain_maps.npy")
ids = np.load(path + folder + "/domain_identification/domain_ids.npy")

domain_map = dmaps.get_domain_map(field)
plt.pcolor(
    domain_map, cmap="prism", vmin=np.nanmin(domain_map), vmax=np.nanmax(domain_map)
)

#%%
geofile = "/Users/ccamargo/Desktop/budget/data/MSLA_CMEMS_glb_merged_1993-2019_orig_deseason_detrend_300kmfilter_360x180.nc"
geofile = "/Users/ccamargo/Desktop/budget/data/MSLA_CMEMS_glb_merged_1993-2019_orig_deseason_detrend_300kmfilter_res1deg.nc"
ds = xr.open_dataset(geofile)
dmaps.plot_dMaps_output(
    geofile=geofile,
    fpath=path + folder,
    output="domain",
    outpath=None,
    show_seeds="homogeneity",
)
# domain_map = np.nansum(field,axis=0)
#%%
da = xr.Dataset(
    data_vars={
        "mask": (("lat", "lon"), domain_map),
    },
    coords={
        "lat": ds.lat,
        "lon": ds.lon,
    },
)

da.to_netcdf(path + "mpa_k5_tmp.nc")
ifile = path + "mpa_k5_tmp.nc"
ofile = path + "map_k5.nc"
command = "cdo -remapbil,{}template.nc {} {}".format(path, ifile, ofile)
import os

os.system(command)
da = xr.open_dataset(ofile)
#%%
landmask = np.array(ds.sla[0, :, :])
landmask[np.isfinite(landmask)] = 1
landmask[np.isnan(landmask)] = -1
d = np.array(domain_map)
d[np.isnan(d)] = 0
d[np.where(landmask == -1)] = -1
plt.figure()
plt.contourf(domain_map)

plt.contour(d, colors="black", levels=np.arange(-1, 97, 1), linewidths=0.25)
plt.show()
#%%