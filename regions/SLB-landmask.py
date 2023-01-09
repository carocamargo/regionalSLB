#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:18:45 2021

@author: ccamargo
"""


import xarray as xr
import numpy as np

path = "/Volumes/LaCie_NIOZ/reg/world/data/budget/"
files = [
    "MSLA_CMEMS_glb_merged_1993-2017_res1deg.nc",
    "sSLA_ENS_glb_1993-2017_res1deg.nc",
    "bSLA_ALL_glb_1993-2016_res1deg.nc",
]
ds = xr.open_dataset(path + files[0])
mask = np.array(ds.sla).min(axis=0)
len(mask[np.isnan(mask)])
mask[np.isfinite(mask)] = 1

ds = xr.open_dataset(path + files[1])
mask66 = np.array(ds.norm_steric_sl).min(axis=0)
len(mask66[np.isnan(mask66)])
mask66[np.isfinite(mask66)] = 1

mask_final = np.array(mask66 + mask)
da = xr.Dataset(
    data_vars={"mask": (("lat", "lon"), mask_final)},
    coords={"lat": ds.lat, "lon": ds.lon},
)
da.to_netcdf(path + "landmask.nc")
