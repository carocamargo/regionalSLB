#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:19:35 2022

@author: ccamargo
"""

import xarray as xr
import numpy as np

path_dyn = "/Volumes/LaCie_NIOZ/data/dynamicREA/"


#%% steric
path = "/Volumes/LaCie_NIOZ/data/budget/"
dssl = xr.open_dataset(path + "steric_full.nc")
# select reanalysis and ens
dssl = dssl.sel(names=["SODA", "CGlors", "FOAM", "ORAS", "GLORYS"])
# ens_up = dssl.steric_up.mean(dim='names')
# ens_full = dssl.steric_full.mean(dim='names')

names = [name for name in np.array(dssl.names)]
names.append("ENS")
data = np.zeros(())
ens_up = np.zeros((1, len(dssl.time), len(dssl.lat), len(dssl.lon)))
ens_up.fill(np.nan)
ens_full = np.full_like(ens_up, np.nan)

ens_up[0] = np.array(dssl.steric_up.mean(dim="names"))
ens_full[0] = np.array(dssl.steric_full.mean(dim="names"))

data_up = np.vstack([np.array(dssl.steric_up), ens_up])
data_full = np.vstack([np.array(dssl.steric_full), ens_full])

ds = xr.Dataset(
    data_vars={
        "steric_up": (("names", "time", "lat", "lon"), data_up),
        "steric_full": (("names", "time", "lat", "lon"), data_full),
        "steric_deep": (("time", "lat", "lon"), dssl.steric_deep),
    },
    coords={"names": names, "lat": dssl.lat, "lon": dssl.lon, "time": dssl.time},
)
ds.attrs = dssl.attrs
dssl = ds
# compute anomaly
dssl["ssl_gmsl"] = dssl["steric_full"].mean(dim=("lat", "lon"))
dssl["ssl_anom"] = dssl["steric_full"] - dssl["ssl_gmsl"]
# save it
dssl
dssl = dssl.drop_vars(["steric_up", "steric_deep"])
dssl.to_netcdf(path_dyn + "steric.nc")

#%%
from datetime import datetime


def get_time(time):
    t = datetime.utcfromtimestamp(
        (time - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
    )
    # for t in time]
    # t = [t.timetuple().tm_year + (t.timetuple().tm_yday/365) for t in t]
    return [t.year, t.month]


#%% SSH from rea
dssh = xr.open_dataset(path_dyn + "ssh_rea_corrected.nc")
t0 = np.array(dssh.time[0])
t1 = np.array(dssh.time[-1])
y0, m0 = get_time(t0)
ti = "{}-{}-01".format(y0, str(m0).zfill(2))
yf, mf = get_time(t1)
if mf == 12:
    yf = yf + 1
    mf = "01"
else:
    mf = mf + 1
tf = "{}-{}-28".format(yf, str(mf).zfill(2))

time = np.arange(ti, tf, dtype="datetime64[M]")
dssh["time"] = np.array(time)

names = np.array(dssh.name)
print(names)
# correct it to match the ones from steric
names = ["SODA", "CGlors", "FOAM", "ORAS", "GLORYS", "ENS"]
dssh["names"] = names

# change from time,lat,lon,name to name,time,lat,lon
timedim, latdim, londim, namedim = np.array(dssh.ssh_anom.shape)
ssh = np.zeros((namedim, timedim, latdim, londim))
ssh_anom = np.full_like(ssh, 0)
gmsl = np.zeros((namedim, timedim))
for i in range(namedim):
    ssh[i] = np.array(dssh.ssh[:, :, :, i])
    ssh_anom[i] = np.array(dssh.ssh_anom[:, :, :, i])
    gmsl[i] = np.array(dssh.gmsl[:, i])
#% %
ds = xr.Dataset(
    data_vars={
        "ssh": (("names", "time", "lat", "lon"), ssh),
        "ssh_anom": (("names", "time", "lat", "lon"), ssh_anom),
        "ssh_gmsl": (("names", "time"), gmsl),
    },
    coords={"lat": dssh.lat, "lon": dssh.lon, "time": dssh.time, "names": names},
)


#%% merge SSH and SSL
da = xr.merge([dssl, ds])
print(da)
da.attrs["units"] = "meters (m)"
da.attrs["description"] = "SSL: steric, SSH: sea-surface height form reanalysis"
da.attrs["steric"] = "Full columns steric (0-2000m from each dataset + >2000m from PJ)"
da.attrs[
    "ensemble"
] = "ENS, both of dynamic and steric is the ensemble of the 5 reanalysis"
da["DSL"] = da["ssh_anom"] - da["ssl_anom"]
da["DSL"].attrs = {"units": "m", "long_name": "Dynamic sea-level anomaly"}
da["SDSL"] = da["ssh_anom"] + da["ssl_gmsl"]
da["SDSL"].attrs = {"units": "m", "long_name": "Sterodynamic sea-level anomaly"}
da = da.rename({"steric_full": "ssl"})
da.attrs["script"] = "SLB_dynamic.py"
da.attrs["metadata"] = "Dynamic (DSL) and Sterodynamic SL from reanalysis. "
# save
da.to_netcdf(path_dyn + "dynamic_sl.nc")
da.to_netcdf(path + "dynamic_sl.nc")

#%%
names = ["SODA", "CGlors", "FOAM", "ORAS", "GLORYS"]
ens_dyn = da["DSL"].sel(names=names).mean(dim="names")
ens_sdsl = da["DSL"].sel(names=names).mean(dim="names")
ds = xr.Dataset(
    data_vars={
        "ens_dyn_v1": (("time", "lat", "lon"), ens_dyn),
        "ens_sdsl_v1": (("time", "lat", "lon"), ens_sdsl),
        "ens_dyn_v2": (("time", "lat", "lon"), np.array(da["DSL"][-1, :, :, :])),
        "ens_sdsl_v2": (("time", "lat", "lon"), np.array(da["SDSL"][-1, :, :, :])),
    },
    coords={"lat": da.lat, "lon": da.lon, "time": da.time, "names": "ENS"},
)
ds.attrs = da.attrs
ds.attrs[
    "v1"
] = "First we compute DSL and SDSL from each reanalysis, then take the mean (ensemble)"
ds.attrs[
    "v2"
] = "First we take the mean of SSH and SSL from the 5 reanlysis, then compute DSL and SDSL"
ds.attrs["reanlysis"] = names
ds.to_netcdf(path + "dynamic_sl_ENS.nc")
ds.to_netcdf(path_dyn + "dynamic_sl_ENS.nc")
