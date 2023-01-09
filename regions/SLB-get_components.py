#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:54:33 2021

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
ds = xr.open_dataset(path_data + file)
ds
ds.trend[0, :, :].plot()
path = "/Volumes/LaCie_NIOZ/data/barystatic/use/comb/"
# file = '16_input_datasets_buf_1993-2020_180x360_update_v2.nc' # variations at source (Land)
file = "18_SLF_monthly_1993-2020_180x360_update.nc"  # SLF tim series
file = "ALL_datasets_1993-2020_180x360_v3_update.nc"
ds2 = xr.open_dataset(path + file)
ds2.SLF[0, 0, :, :].plot()
plt.show()
#% % check dimensions
np.all(ds.lat == ds2.lat)
np.all(ds.lon == ds2.lon)
print(ds2.name)
ds2["name"] = [
    "AIS_IMB",
    "AIS_R19",
    "GLA_ZMP",
    "GLA_WGP",
    "AIS_CSR",
    "GIS_CSR",
    "LWS_CSR",
    "GLWS_CSR",
    "TCWS_CSR",
    "AIS_JPL",
    "GIS_JPL",
    "LWS_JPL",
    "GLWS_JPL",
    "TCWS_JPL",
    "GIS_IMB",
    "GIS_M19",
    "LWS_GWB",
    "LWS_WGP",
]
print(ds2.name)
#% %
trends_1993 = np.zeros((len(ds2.name), len(ds.lat), len(ds.lon)))
uncs_1993 = np.zeros((len(ds2.name), len(ds.lat), len(ds.lon)))

for i, name in enumerate(np.array(ds2.name)):
    # print(name)
    if name in np.array(ds.name):
        # print(name)
        trends_1993[i, :, :] = np.array(ds.trend.sel(name=name))
        uncs_1993[i, :, :] = np.array(ds.unc_total.sel(name=name))
#% % trends from 2003

file = "final_dataset_2003-2016.nc"
ds = xr.open_dataset(path_data + file)

trends_2003 = np.zeros((len(ds2.name), len(ds.lat), len(ds.lon)))
uncs_2003 = np.zeros((len(ds2.name), len(ds.lat), len(ds.lon)))

for i, name in enumerate(np.array(ds2.name)):
    # print(name)
    if name in np.array(ds.name):
        # print(name)
        trends_2003[i, :, :] = np.array(ds.trend.sel(name=name))
        uncs_2003[i, :, :] = np.array(ds.unc_total.sel(name=name))
#% % reconstructions
df = pd.read_pickle(path_data + "OM_reconstructions_2003-2016.p")
llon = np.array(df["lon"]).reshape(180, 360)[0, :]
llat = np.array(df["lat"]).reshape(180, 360)[:, 0]
np.all(llon == np.array(ds2.lon))
np.all(llat == np.array(ds2.lat))
#% % test
df = df.sort_values(by=["lat", "lon"], ascending=[False, True])
data = np.array(df["IMB+WGP_trend_tot"]).reshape(180, 360)
plt.pcolor(data)
plt.show()
#% %
data2 = np.array(ds.trend.sel(name=["AIS_IMB", "GIS_IMB", "GLA_WGP", "LWS_WGP"]))
data2 = np.sum(data2, axis=0)
plt.pcolor(data2)
plt.show()
#% %
plt.pcolor(data - data2)
plt.show()
#% %
df2 = pd.read_pickle(path_data + "OM_reconstructions_1993-2016.p")
df2 = df2.sort_values(by=["lat", "lon"], ascending=[False, True])

#% %
reconstr = ["JPL", "CSR", "IMB+WGP", "IMB+GWB+ZMP", "UCI+WGP", "UCI+GWB+ZMP"]
combos = [
    ["JPL"],
    ["CSR"],
    ["IMB", "WGP"],
    ["IMB", "GWB", "ZMP"],
    ["UCI", "WGP"],
    ["UCI", "GWB", "ZMP"],
]
rec_trends_2003 = np.zeros((len(reconstr), len(ds2.lat), len(ds2.lon)))
rec_uncs_2003 = np.zeros((len(reconstr), len(ds2.lat), len(ds2.lon)))
rec_SLF = np.zeros((len(reconstr), len(ds2.time), len(ds2.lat), len(ds2.lon)))
rec_trends_1993 = np.zeros((len(reconstr), len(ds2.lat), len(ds2.lon)))
rec_uncs_1993 = np.zeros((len(reconstr), len(ds2.lat), len(ds2.lon)))

i = 0
for title, combo in zip(reconstr, combos):

    #% %
    names = [
        name
        for name in np.array(ds2.name)
        for comb in combo
        if name.split("_")[1] == comb
    ]
    rec_SLF[i, :, :, :] = np.array(
        ds2.SLF.sel(name=["AIS_IMB", "GIS_IMB", "GLA_WGP", "LWS_WGP"])
    ).sum(axis=0)

    rec_trends_2003[i, :, :] = np.array(df[title + "_trend_tot"]).reshape(180, 360)
    rec_uncs_2003[i, :, :] = np.array(df[title + "_unc_tot"]).reshape(180, 360)
    rec_trends_1993[i, :, :] = np.array(df2[title + "_trend_tot"]).reshape(180, 360)
    rec_uncs_1993[i, :, :] = np.array(df2[title + "_unc_tot"]).reshape(180, 360)
    i = i + 1
    # for iname,name in enumerate(names):
    #     rec_SLF_1993[iname,:,:,:] =

#%% construct dataset
ind = np.arange(0, 180)
ind = ind[::-1]
dbary = xr.Dataset(
    data_vars={
        "SLF": (("name", "time", "lat", "lon"), ds2["SLF"] / 1000),
        "trend_1993": (("name", "lat", "lon"), trends_1993 / 1000),
        "unc_1993": (("name", "lat", "lon"), uncs_1993 / 1000),
        "trend_2003": (("name", "lat", "lon"), trends_2003 / 1000),
        "unc_2003": (("name", "lat", "lon"), uncs_2003 / 1000),
        "rec_SLF": (("reconstruction", "time", "lat", "lon"), rec_SLF / 1000),
        "rec_trend_1993": (("reconstruction", "lat", "lon"), rec_trends_1993 / 1000),
        "rec_unc_1993": (("reconstruction", "lat", "lon"), rec_uncs_1993 / 1000),
        "rec_trend_2003": (("reconstruction", "lat", "lon"), rec_trends_2003 / 1000),
        "rec_unc_2003": (("reconstruction", "lat", "lon"), rec_uncs_2003 / 1000),
    },
    coords={
        "name": ds2.name,
        "time": ds2.time,
        "reconstruction": reconstr,
        "lat": ind,  # ds2.lat,
        "lon": ds2.lon,
    },
)
dbary.attrs["trends_period"] = "trends from 1993-2016 and 2003-2016"
dbary.attrs["units"] = "meters and m/y"
dbary.trend_1993[0, :, :].plot()
plt.show()
dbary["lat"] = -dsla.lat
dbary.trend_1993[0, :, :].plot()
plt.show()

dbary.attrs[
    "description"
] = "Barystatic SL trend and time series for 1993-2016 and 2003-2016"
dbary.attrs["units"] = "m"
dbary.attrs["script"] = "SLB-get_components.py"
# dbary.to_netcdf(path_save + 'bSLA_ALL_glb_1993-2016_res1deg.nc')
#%% checking data
for r in ["JPL", "CSR", "IMB+WGP", "IMB+GWB+ZMP", "UCI+WGP", "UCI+GWB+ZMP"]:
    dbary.rec_SLF.sel(reconstruction=r).mean(dim=["lat", "lon"]).plot()
    ds = dbary.rec_SLF.sel(reconstruction=r).groupby("time.month").mean()
    # ds = dbary.SLF.sel(name='LWS_WGP').groupby("time.month").mean()

    ds.plot(
        col="month",
        col_wrap=4,  # each row has a maximum of 4 columns
        # The remaining kwargs customize the plot just as for not-faceted plots
        robust=True,
        cmap="RdBu_r",
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "pad": 0.1,
        },
    )
    plt.show()

for n in np.array(dbary.name):
    dbary.SLF.sel(name=n).mean(dim=["lat", "lon"]).plot()
    plt.show()
#%%
ds = dbary.rec_SLF.sel(reconstruction=r)
tdec, tdec0 = sl.get_dec_time(np.array(ds.time))
trend, _ = sl.get_reg_trend(
    tdec, np.array(ds.data * 1000), np.array(ds.lat), np.array(ds.lon)
)
plt.pcolor(ds.lon, ds.lat, trend, vmin=-6, vmax=6, cmap="RdBu_r")
plt.colorbar()

dbary.to_netcdf(path_save + "bSLA_ALL_glb_1993-2016_res1deg.nc")

#%%
