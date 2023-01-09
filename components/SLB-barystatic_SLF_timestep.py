#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:03:59 2022

Obtain time series of SLF

@author: ccamargo
"""


import numpy as np

# import scipy.optimize as opti
import xarray as xr

# import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import utils_SLE_v2 as sle

# from netCDF4 import Dataset
import pandas as pd

# import os

# import datetime as dt

# import cmocean as cm
# from mpl_toolkits.basemap import Basemap
# from matplotlib.gridspec import GridSpec
# from cartopy import crs as ccrs#, feature as cfeature

#% % packages for plotting
# from pandas.plotting import table
# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.basemap import Basemap
# from matplotlib.colors import ListedColormap

# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
# col_dict={1:"black", # WN
#           2:"palegoldenrod", # PL
#           3:"lightpink", # PLWN
#           4:"orange", # AR1
#           5:"teal", # Ar5
#           6:"darkmagenta", # AR9
#           7:"skyblue", # ARf
#           8:"crimson" # GGM
#           }

# # We create a colormar from our list of colors
# cmapnm = ListedColormap([col_dict[x] for x in col_dict.keys()])
def lat2str(deg):
    # Source: https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
    # Adapted so that 0 has no indication of direction.
    minn = 60 * (deg - np.floor(deg))  # transform to minutes
    deg = np.floor(deg)  # degrees
    dirr = "N"
    if deg < 0:
        if minn != 0.0:
            deg += 1.0
            minn -= 60.0
        dirr = "S"
    elif deg == 0:
        dirr = ""
    return ("%d\N{DEGREE SIGN} %s") % (np.abs(deg), dirr)


def lon2str(deg):
    # Source: https://github.com/matplotlib/basemap/blob/master/examples/customticks.py
    # Adapted so that 0 has no indication of direction.
    minn = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    dirr = ""  #'E'
    if deg < 0:
        if minn != 0.0:
            deg += 1.0
            minn -= 60.0
        dirr = ""  #'W'
    elif deg == 0:
        dirr = ""
    return ("%d\N{DEGREE SIGN} %s") % (np.abs(deg), dirr)


#% %
# ds_mask = xr.open_dataset('/Volumes/LaCie_NIOZ/data/ETOPO/ETOPO1_Ice-180x360.nc')
# ds_mask.z.plot(# vmin=-1,vmax=1
#                 );##plt.show()

# # ds_mask=ds_mask.sortby('lat',ascending=False)
# oceanmask=np.array(ds_mask.z)
# oceanmask[oceanmask>=0]=1
# oceanmask[oceanmask<=0]=np.nan
#% %
def ocean_mean(value, lat, lon):
    # value=np.array(ds.best_trend[0,:,:])
    ocean_lit = np.array([360000000, 361060000, 357000000, 360008310, 357000000])
    ocean_area = np.mean(ocean_lit) / 10 ** 5
    grid_area = sl.get_grid_area(np.ones((180, 360)))

    # plt.pcolor(oceanmask);##plt.show()

    # value=np.array(tws_gbw)
    # tdec=np.array(tdec_gwb)
    da = xr.Dataset(
        data_vars={"data": (("lat", "lon"), value)}, coords={"lat": lat, "lon": lon}
    )
    mu = (da.data * grid_area).sum(dim=("lat", "lon")) / ocean_area
    return mu.data


def glb_to_reg(value=1, mask=np.ones((180, 360))):

    df = pd.DataFrame(mask.flatten(), columns=["mask"])
    df["area"] = sl.get_grid_area(mask).flatten()

    df["regional_value"] = (np.full_like(mask, value).flatten() * df["mask"]) / len(
        mask[np.isfinite(mask)]
    )  # df['area']
    # df['regional_value'] = (np.full_like(mask,value).flatten() * df['mask'])/  df['area']

    # df=pd.DataFrame(mask.flatten(),columns=['mask'])
    # df['area'] = sl.get_grid_area(mask).flatten()
    # # df['area']=df['mask']
    # df['weighted_area']=(df['area']*df['mask'])/np.nansum(df['mask']*df['area'])
    # df['regional_value'] = (np.full_like(mask,value).flatten() * df['weighted_area'])

    return np.array(df["regional_value"]).reshape(mask.shape)


def glb_to_reg2(value=1, mask=np.ones((180, 360))):

    df = pd.DataFrame(mask.flatten(), columns=["mask"])
    df["area"] = sl.get_grid_area(mask).flatten()

    df["regional_value"] = (
        np.full_like(mask, value).flatten() * df["mask"]
    )  # df['area']
    # df['regional_value'] = (np.full_like(mask,value).flatten() * df['mask'])/  df['area']

    # df=pd.DataFrame(mask.flatten(),columns=['mask'])
    # df['area'] = sl.get_grid_area(mask).flatten()
    # # df['area']=df['mask']
    # df['weighted_area']=(df['area']*df['mask'])/np.nansum(df['mask']*df['area'])
    # df['regional_value'] = (np.full_like(mask,value).flatten() * df['weighted_area'])

    return np.array(df["regional_value"]).reshape(mask.shape)


import pickle


def load_dict(name, path):
    with open(path + name + ".pkl", "rb") as f:
        return pickle.load(f)


#%% 1. Transform mean values into regional:
#% %a.  open mask
name_mask = "masks_dict"
path_mask = "/Volumes/LaCie_NIOZ/data/barystatic/"
# path_mask=path+name
masks = load_dict(name_mask, path_mask)
print(masks.keys())

mask_keys = {
    "AIS_IMB": "AIS_regions",
    "AIS_UCI": "AIS_basins",
    "GIS_IMB": "GIS_regions",
    "GIS_UCI": "GIS_basins",
    "GLA_ZMP": "Glaciers",
}

# b. open mean values
path_data = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/input_data/"
file = "means_v2.nc"
ds = xr.open_dataset(path_data + file)
file = "regional_v3.nc"
ds2 = xr.open_dataset(path_data + file)
path_save = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/SLF_ts/"
#%%
dimlat = 180
dimlon = 360
names = ["GIS_IMB", "GIS_UCI", "AIS_IMB", "AIS_UCI", "GLA_ZMP"]
regional_mass_change = np.zeros((len(names), dimlat, dimlon))

for iname, name in enumerate(names):
    var = "{}_slc".format(name)
    da = ds[var]

    masks = load_dict(name_mask, path_mask)
    regs = ds["reg_{}".format(name)]
    # regional_mass_change = np.zeros((len(regs),dimlat,dimlon))
    regdim, timedim = da.data.shape
    if timedim == 40:
        time = np.array(ds.year)
    elif timedim == 324:
        time = np.array(ds.time)
    else:
        print("check dataset")
    regional_mass_change_EWH = np.zeros((len(regs), timedim, dimlat, dimlon))

    for ireg, reg in enumerate(np.array(regs)):
        print(reg)
        if name == "GLA_ZMP":
            reg = reg.split("_")[1]

        mask = masks[mask_keys[name]][reg]

        data = np.array(da.data[ireg])

        for itime in range(timedim):

            regional_mass_change_EWH[ireg, itime, :, :] = sle.height_to_EWH(
                glb_to_reg(np.array(data[itime]), mask)
            ).reshape(180, 360)

    # make regional dataset:
    da = xr.Dataset(
        data_vars={
            # 'SLF_ASL':(('reg','time','lat','lon'),regional_SLF_rsl),
            # 'SLF_RSL':(('reg','time','lat','lon'),regional_SLF_asl),
            "mass_change_EWH": (
                ("reg", "time", "lat", "lon"),
                regional_mass_change_EWH,
            ),
        },
        coords={
            "lon": ds2.lon,
            "lat": -ds2.lat,
            "reg": [str(reg) for reg in np.array(regs)],
            "time": np.array(time),
        },
    )
    if name == "AIS_IMB":
        da = da.drop_sel(reg="AIS")

    elif name == "GIS_IMB" or name == "GIS_UCI":
        da = da.drop_sel(reg="GIS")
    if name.split("_")[0] == "AIS":
        mask = masks["AIS_regions"]["AIS"]
    if name.split("_")[0] == "GIS":
        mask = masks["GIS_regions"]["GIS"]
    if name == "GLA_ZMP":
        mask_dic = masks["Glaciers"]
        mask_gla = np.zeros((180, 360))
        for key in mask_dic:
            mask_tmp = mask_dic[key]
            mask_tmp[np.isnan(mask_tmp)] = 0
            mask_gla = mask_gla + mask_tmp
        mask_gla[mask_gla == 0] = np.nan
        mask = mask_gla
    da = da.sum(dim="reg") * mask

    regional_SLF_rsl = np.zeros((timedim, dimlat, dimlon))
    regional_SLF_asl = np.zeros((timedim, dimlat, dimlon))

    for itime in range(timedim):
        # run SLE
        X = np.array(ds2.lon)
        Y = np.array(-ds2.lat)
        d = np.array(da.mass_change_EWH[itime, :, :])
        if name.split("_")[-1] == "IMB":
            d = -d

        regional_SLF_asl[itime, :, :] = sle.run_SLE(d, name, var="asl")
        # regional_SLF_rsl[itime,:,:] = sle.run_SLE(d,name,var='rsl')
        regional_SLF_rsl[itime, :, :], _ = sle.get_output(
            var="rsl",
            path="/Volumes/LaCie_NIOZ/PhD/Barystatic/SLM_run/",
            fname="{}__sle2r-cc".format(name),
        )

    da["SLF_ASL"] = (["time", "lat", "lon"], regional_SLF_asl)
    da["SLF_RSL"] = (["time", "lat", "lon"], regional_SLF_rsl)

    # save

    da.to_netcdf(path_save + name + ".nc")

#%% regional
ds2 = xr.open_dataset(path_data + file)
path_save = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/SLF_ts/"
#%%
dimlat = 180
dimlon = 360
names = np.array(ds2.name)

for iname, name in enumerate(names):
    da = ds2.sel(name=name)["SL_EWH"]

    time = np.array(ds2.time)
    timedim = len(time)
    regional_mass_change_EWH = np.array(da.data)
    regional_SLF_rsl = np.full_like(regional_mass_change_EWH, 0)
    regional_SLF_asl = np.full_like(regional_mass_change_EWH, 0)

    for itime in range(timedim):
        # run SLE
        X = np.array(ds2.lon)
        Y = np.array(-ds2.lat)
        d = np.array(regional_mass_change_EWH[itime, :, :])
        # regional_SLF_rsl[itime,:,:] = sle.run_SLE(d,name,var='rsl')
        regional_SLF_asl[itime, :, :] = sle.run_SLE(d, name, var="asl")
        regional_SLF_rsl[itime, :, :], _ = sle.get_output(
            var="rsl",
            path="/Volumes/LaCie_NIOZ/PhD/Barystatic/SLM_run/",
            fname="{}__sle2r-cc".format(name),
        )

    # make regional dataset:
    da = xr.Dataset(
        data_vars={
            "SLF_ASL": (("time", "lat", "lon"), regional_SLF_rsl),
            "SLF_RSL": (("time", "lat", "lon"), regional_SLF_asl),
            "mass_change_EWH": (("time", "lat", "lon"), regional_mass_change_EWH),
        },
        coords={
            "lon": ds2.lon,
            "lat": -ds2.lat,
            "time": np.array(time),
        },
    )

    da.to_netcdf(path_save + name + ".nc")

#%% combine all in one dataset
import os
from datetime import datetime

path_save = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/SLF_ts/"
flist = os.listdir(path_save)
i = 0
file = flist[i]
ds = xr.open_dataset(path_save + file)
datasets = []
for file in flist:
    name = file.split(".nc")[0]
    # print(name)
    da = xr.open_dataset(path_save + file)
    # print(da.time)
    # print('')
    varis = ["SLF_ASL", "SLF_RSL", "mass_change_EWH"]
    # new_vars = [var+'_'+name for var in varis]
    new_vars = {var: var + "_" + name for var in varis}
    da = da.rename(new_vars)
    if name.split("_")[-1] == "IMB":
        ti = "1992-01-01"
        tf = "2019-01-01"
        da["time"] = np.arange(ti, tf, dtype="datetime64[M]")
    elif name.split("_")[-1] == "ZMP" or name.split("_")[-1] == "UCI":
        # print(name)
        ti = "1979"
        tf = "2019"
        da["time"] = np.arange(ti, tf, dtype="datetime64[Y]")
    else:
        t = [
            datetime.utcfromtimestamp(
                (t - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
            )
            for t in np.array(da.time)
        ]
        y = [t.timetuple().tm_year for t in t]
        m = [t.timetuple().tm_mon for t in t]
        dt = [
            np.array("{}-{}".format(y, str(m).zfill(2)), dtype="datetime64")
            for y, m in zip(y, m)
        ]
        da["time"] = dt

        # print(da.time)
    datasets.append(da)
#% % merge datasets
da = xr.merge(datasets)
print(da)
#% %
da.attrs["units"] = "mm"
da.attrs["description"] = "Barystatic SLC in mm."

da = da.sel(time=slice("1993-01-01", da.time[-1]))

#% % make combos
combos = [
    "IMB_WGP",
    "IMB_ZMP_GWB",
    "UCI_WGP",
    "UCI_ZMP_GWB",
    "CSR",
    "JPL",
]
data_combo = np.zeros((len(combos), len(da.time), len(da.lat), len(da.lon)))
data_combo.fill(np.nan)

data_combo2 = np.full_like(data_combo, np.nan)
data_combo3 = np.full_like(data_combo, np.nan)

for ic, combo in enumerate(combos):
    datasets = combo.split("_")
    var = [
        key
        for key in da.variables
        if key.split("_")[0] == "SLF" and key.split("_")[1] == "ASL"
    ]
    var = [var for var in var if var.split("_")[-1] in datasets]
    #% % compute ENS mean
    data = np.zeros((len(var), len(da.time), len(da.lat), len(da.lon)))
    data.fill(np.nan)
    for i, v in enumerate(var):
        data[i] = np.array(da[v])

    var = [
        key
        for key in da.variables
        if key.split("_")[0] == "SLF" and key.split("_")[1] == "RSL"
    ]
    var = [var for var in var if var.split("_")[-1] in datasets]
    data2 = np.zeros((len(var), len(da.time), len(da.lat), len(da.lon)))
    data2.fill(np.nan)
    for i, v in enumerate(var):
        data2[i] = np.array(da[v])
    #% %
    var = [key for key in da.variables if key.split("_")[0] == "mass"]
    var = [var for var in var if var.split("_")[-1] in datasets]
    data3 = np.zeros((len(var), len(da.time), len(da.lat), len(da.lon)))
    data3.fill(np.nan)
    for i, v in enumerate(var):
        data3[i] = np.array(da[v])

    # sum components
    data_combo[ic] = data.sum(axis=0)
    data_combo2[ic] = data2.sum(axis=0)
    data_combo3[ic] = data3.sum(axis=0)
data_combo[data_combo == 0] = np.nan
data_combo2[data_combo2 == 0] = np.nan
data_combo3[data_combo3 == 0] = np.nan

for ic in range(len(combos)):
    data_combo[ic] = data_combo[ic] - np.nanmean(data_combo[ic, 144:276, :, :], axis=0)
    data_combo2[ic] = data_combo2[ic] - np.nanmean(
        data_combo2[ic, 144:276, :, :], axis=0
    )
    data_combo3[ic] = data_combo3[ic] - np.nanmean(
        data_combo3[ic, 144:276, :, :], axis=0
    )


#%%   ensemble
da = da.assign_coords({"combos": combos})
da["SLF_ASL_combos"] = (["combos", "time", "lat", "lon"], data_combo)
da["SLF_RSL_combos"] = (["combos", "time", "lat", "lon"], data_combo2)
da["MASS_CHANGE_combos"] = (["combos", "time", "lat", "lon"], data_combo3)

da["SLF_ASL_combo_ens"] = da["SLF_ASL_combos"].mean(dim="combos")
da["SLF_RSL_combo_ens"] = da["SLF_RSL_combos"].mean(dim="combos")
da["MASS_CHANGE_combo_ens"] = da["MASS_CHANGE_combos"].mean(dim="combos")

ens = np.zeros((1, len(da.time), len(da.lat), len(da.lon)))
ens.fill(np.nan)
ens[0] = np.array(da.SLF_ASL_combo_ens)
data = np.vstack([data_combo, ens])

ens2 = np.zeros((1, len(da.time), len(da.lat), len(da.lon)))
ens2.fill(np.nan)
ens2[0] = np.array(da.SLF_RSL_combo_ens)
data2 = np.vstack([data_combo2, ens])

ens3 = np.zeros((1, len(da.time), len(da.lat), len(da.lon)))
ens3.fill(np.nan)
ens3[0] = np.array(da.MASS_CHANGE_combo_ens)
data3 = np.vstack([data_combo3, ens])

combos.append("ENS")
da = da.assign_coords({"reconstruction": combos})
da["SLF_ASL"] = (["reconstruction", "time", "lat", "lon"], data)
da["SLF_RSL"] = (["reconstruction", "time", "lat", "lon"], data2)
da["MASS_CHANGE"] = (["reconstruction", "time", "lat", "lon"], data3)

#%% check ens
import matplotlib.pyplot as plt

ds = da["SLF_ASL"]
plt.figure(figsize=(15, 10), dpi=100)
for i, rec in enumerate(np.array(ds.reconstruction)):
    mu = np.array(ds[i, :, :, :].mean(dim=("lat", "lon")))
    plt.plot(mu, label=rec)
plt.ylim([-100, 20])
plt.legend()
plt.show()

#%% save it
path_budget = "/Volumes/LaCie_NIOZ/data/budget/"
da.to_netcdf(path_budget + "barystatic_timeseries.nc")
