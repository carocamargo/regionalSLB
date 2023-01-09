#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:27:40 2022

@author: ccamargo
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# import os

#% % open trends
path = "/Volumes/LaCie_NIOZ/data/steric/trends/"
file = "preferred_trend_and_uncertainty_180x360.nc"
ds = xr.open_dataset(path + file)
print(ds)
periods = ["2005-2015", "1993-2017"]
trend = np.zeros((len(ds.fname), len(periods), len(ds.lat), len(ds.lon)))
trend.fill(np.nan)
unc = np.full_like(trend, np.nan)
for i, period in enumerate(periods):
    y0, y1 = period.split("-")
    trend[:, i, :, :] = ds["trend_{}".format(y0)]
    unc[:, i, :, :] = ds["unc_{}".format(y0)]

dx = xr.Dataset(
    data_vars={
        "unc": (("names", "periods", "lat", "lon"), unc),
        "trend_up": (("names", "periods", "lat", "lon"), trend),
    },
    coords={
        "names": np.array(ds.fname),
        "periods": periods,
        "lat": ds.lat,
        "lon": ds.lon,
    },
)
#% % add deep
da = xr.open_dataset("/Volumes/LaCie_NIOZ/data/steric/deep_ocean/" + "PJ.nc")
# da = xr.open_dataset("/Volumes/LaCie_NIOZ/data/budget/steric_full.nc")
da
deep = np.array(da.steric)
plt.pcolor(deep)
deep[np.isnan(deep)] = 0
plt.pcolor(trend[0, 0, :, :] + deep)
trend_full = np.zeros((trend.shape))
for ip in range(len(periods)):
    for iname in range(len(ds.fname)):
        trend_full[iname, ip, :, :] = np.array(trend[iname, ip, :, :] + deep)
        # unc[iname,ip,:,:] = np.array(unc[iname,ip,:,:] + deep)

plt.pcolor(trend_full[iname, ip, :, :] - trend[iname, ip, :, :])
#%%
dx["trend_full"] = (["names", "periods", "lat", "lon"], trend_full)
dx["trend_deep"] = (["lat", "lon"], deep)

dx.trend_up.attrs["units"] = "mm/yr"
dx.trend_full.attrs["units"] = "mm/yr"
dx.trend_deep.attrs["units"] = "mm/yr"
dx.unc.attrs["units"] = "mm/yr"
#%%
d1 = np.array(dx['trend_full'][0,0,:,:])
d2 = np.array(dx['trend_up'][0,0,:,:])
plt.pcolor(d2-d1)
#%%
dx.attrs[
    "Summary"
] = "Full depth Best Steric trend and uncertainty according to the noise-model analysis \
    (see fig 7 and 8, Camargo et al, 2020, JGR:Oceans"
dx.attrs["units"] = "mm/year"
dx.attrs["deep"] = "Deep(>2000m) steric estimates from Purkey & Johnson, 2010"
dx.attrs[
    "Description"
] = "Linear trends from 1993-2017 and 2005-2015 (full years, jan-dec)"
dx.attrs["script"] = "SLB-steric_trends.py"
dx.attrs["author"] = "Carolina Camargo (carolina.camargo@nioz.nl)"
dx.attrs[
    "journal_reference"
] = " Camargo et al., 2020, JGR:Oceans,  https://doi.org/10.1029/2020JC016551"
dx.attrs[
    "dataset_reference"
] = "https://doi.org/10.4121/uuid:b6e5e4bc‐d382‐4b51‐837b‐c5cde4980bf3"
dx.to_netcdf("/Volumes/LaCie_NIOZ/data/budget/trends/steric.nc")
