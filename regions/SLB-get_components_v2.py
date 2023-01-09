#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:09:38 2022

@author: ccamargo
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import pandas as pd

#%% SLA

path_data = "/Volumes/LaCie_NIOZ/data/altimetry/world/CMEMS/"
path_save = "/Volumes/LaCie_NIOZ/reg/world/data/budget/"
path_template = "/Volumes/LaCie_NIOZ/reg/world/data/"

file = "MSLA_CMEMS_glb_merged_1993-2019"

# 1. regrid to 1 degree:
res = 1
# use already created template
ifile = path_data + file + ".nc"
ofile = path_save + file + "_res{}deg.nc".format(res)
command = "cdo -remapbil,{}template.nc {} {}".format(path_template, ifile, ofile)
os.system(command)

# select time:
# 1993-2017 (to match steric and mass)
t0_old = "1993"
t1_old = "2019"
t0 = 1993
t1 = 2017
infile = ofile
outfile = (
    path_save
    + ofile.split("/")[-1].split(t0_old)[0]
    + "{}-{}".format(t0, t1)
    + ofile.split("/")[-1].split(t1_old)[-1]
)
command = (
    "cdo selyear," + str(t0) + "/" + str(t1) + " " + str(infile) + " " + str(outfile)
)
os.system(command)

dsla = xr.open_dataset(outfile)
# dsla=xr.open_dataset(path_save+'MSLA_CMEMS_glb_merged_1993-2017_res1deg.nc')
#%% steric
path_data = "/Volumes/LaCie_NIOZ/data/steric/pub/files/"
file = "time_series_regional_1993-2017.nc"
file2 = "trends_regional_1993-2017.nc"
file2 = "preferred_trend_and_uncertainty.nc"


ds = xr.open_dataset(path_data + file, decode_times=False)
fname = 12
# #%% see difference between steric_sl and norm_steric_sl
# #% %
# var = 'steric_sl'
# var = 'norm_steric_sl'
# trend ,_= sl.get_reg_trend(np.array(ds.time),np.array(ds[var][fname,:,:,:])*1000,
#                            np.array(ds.lat),np.array(ds.lon))
# plt.pcolor(ds.lon,ds.lat,trend,vmin=-6,vmax=6,cmap='RdBu_r');plt.colorbar()
# plt.show()
#% % steric time series
var = "norm_steric_sl"
ds[var][fname, :, :, :].std(dim="time").plot()
print(ds[var])
# select ensemble
da = ds.sel(fname_index=fname, drop=True)
da[var][0, :, :].plot()
plt.show()
da.to_netcdf(path_save + "steric_SL_ENS_1993-2017_66deg.nc")
ifile = path_save + "steric_SL_ENS_1993-2017_66deg.nc"
ofile = path_save + "steric_SL_ENS_1993-2017_90deg.nc"
command = "cdo -remapbil,{}template.nc {} {}".format(path_template, ifile, ofile)
os.system(command)
ds = xr.open_dataset(ofile, decode_times=False)
ds[var][0, :, :].plot()
plt.show()
ds2 = ds.where((ds.lat < np.max(da.lat)) & (ds.lat > np.min(da.lat)), np.nan)
ds2[var][0, :, :].plot()
plt.show()

#% % steric trends
ds = xr.open_dataset(path_data + file2)
print(ds)
vars = list(ds.keys())
da2 = xr.Dataset(
    data_vars={var: (("lat", "lon"), ds[var] / 1000) for var in vars},
    coords={"lat": da.lat, "lon": da.lon},
)
for var in vars:
    da2[var].attrs["units"] = "m/y"


da2.attrs = ds.attrs
ifile = path_save + "steric_SL_ENS_trends_1993-2017-66deg.nc"
da2.to_netcdf(ifile)
ofile = path_save + "steric_SL_ENS_trend_1993-2017_90deg.nc"
command = "cdo -remapbil,{}template.nc {} {}".format(path_template, ifile, ofile)
os.system(command)
ds = xr.open_dataset(ofile)
var = "trend_1993"
ds[var][:, :].plot()
plt.show()
ds3 = ds.where((ds.lat < np.max(da.lat)) & (ds.lat > np.min(da.lat)), np.nan)
ds3[var][:, :].plot()
plt.show()
#% % combine trend and time series

dsteric = xr.merge([ds2, ds3])
dsteric["time"] = dsla.time
dsteric["lon"] = dsla.lon
dsteric["lat"] = dsla.lat
dsteric.attrs[
    "description"
] = "Steric SL trend and time series for 1993-2017 and 2005-2015 for ENS"
dsteric.attrs["units"] = "m"
dsteric.attrs["script"] = "SLB-get_components.py"
dsteric.to_netcdf(path_save + "sSLA_ENS_glb_1993-2017_res1deg.nc")
#%% ocean mass
path_data = "/Volumes/LaCie_NIOZ/data/barystatic/pub/"
file = "final_dataset_1993-2016.nc"  # trends & unc

path_data = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results/1993-2017/"
file = "final_dataset_1993-2017.nc"  # trends & unc
ds = xr.open_dataset(path_data + file)
ds
ds.trend[0, :, :].plot()

file = "OM_reconstructions_1993-2017.p"
df = pd.read_pickle(path_data + file)
rec = "IMB+WGP"
slf = np.array(df["{}_trend_tot".format(rec)]).reshape(180, 360)
unc = np.array(df["{}_unc_tot".format(rec)]).reshape(180, 360)
dbary = xr.Dataset(
    data_vars={
        "barystatic_SL": (("lat", "lon"), slf / 1000),
        "barystatic_unc": (("lat", "lon"), unc / 1000),
    },
    coords={
        "lat": dsteric.lat,
        "lon": dsteric.lon,
        "period": "1993-2017",
        "reconstruction": rec,
    },
)


dbary.to_netcdf(path_save + "bSLA_1993-2017_res1deg.nc")

#%%
