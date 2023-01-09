#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:11:57 2022

@author: ccamargo
"""

import xarray as xr
import sys

sys.path.append("/Users/ccamargo/Documents/py_scripts/")
import utils_SL as sl
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def get_dectime(time):
    t = [
        datetime.utcfromtimestamp(
            (t - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        )
        for t in time
    ]
    t = [t.timetuple().tm_year + (t.timetuple().tm_yday / 365) for t in t]
    return np.array(t)


def from_trend_to_ts(trend, time):
    """
    Make time series from a linear trend
    1D or 3D time series

    """

    def timeseries(trend, time):
        """
        1D time series
        """
        A = np.ones((len(time), 2))
        A[:, 1] = np.array(time - np.mean(time))
        pred = np.array(A * trend)

        return pred[:, 1]

    if len(trend.shape) == 2:  # lat,lon
        dimlat, dimlon = trend.shape
        trend = trend.flatten()
        ts = np.zeros((len(time), dimlat * dimlon))
        for i in range(len(trend)):
            if trend[i]:
                ts[:, i] = timeseries(trend[i], time)
        ts = ts.reshape(len(time), dimlat, dimlon)
    else:
        ts = timeseries(trend, time)

    return ts


#%% open datasets
path = "/Volumes/LaCie_NIOZ/data/budget/"
ds = xr.open_dataset(path + "steric_upper.nc")

ds2 = xr.open_dataset(("/Volumes/LaCie_NIOZ/data/steric/deep_ocean/" + "PJ.nc"))


#%% test adding deep steric
time = np.array(ds.time)
tdec = get_dectime(time)

#%%
mu = np.array(ds2.steric.mean())
ts = from_trend_to_ts(mu, tdec)
out = sl.get_ts_trend(tdec, ts, plot=False)
plt.plot(ts)
print(mu == out[0])
#%%
lat = np.array(ds.lat)
lon = np.array(ds.lon)

deep = from_trend_to_ts(np.array(ds2.steric), tdec)  # mm
tr_deep, _ = sl.get_reg_trend(tdec, deep, lat, lon)
plt.pcolor(tr_deep, vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title('Steric PJ deep\n {:.3f}'.format(np.nanmean(tr_deep)))
plt.show()


deep[np.isnan(deep)] = 0
steric_up = np.array(ds.sla_ens*1000)# meters >mm

steric_full = np.array((steric_up) + (deep) )  
tr, _ = sl.get_reg_trend(tdec, steric_full, lat, lon)
plt.pcolor(tr, vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title('Steric ENS full\n {:.3f}'.format(np.nanmean(tr)))
plt.show()

steric_up = np.array(ds.sla_ens*1000)# meters >mm
tr_up, _ = sl.get_reg_trend(tdec, steric_up, lat, lon)
plt.pcolor(tr_up, vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title('Steric ENS up\n {:.3f}'.format(np.nanmean(tr_up)))
plt.show()

tr_deep[np.isnan(tr_deep)]=0
plt.pcolor(tr_up+tr_deep,vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title('Steric ENS up+deep\n {:.3f}'.format(np.nanmean(tr_deep+tr_up)))
plt.show()
#%% test deep ts
tr, _ = sl.get_reg_trend(tdec, deep, lat, lon)
plt.pcolor(tr, vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title("Trend from time series\n" + str(np.nanmean(tr)))
plt.show()

plt.pcolor(ds2.steric, vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title("Original trend\n" + str(np.nanmean(ds2.steric)))
plt.show()

plt.pcolor(tr - np.array(ds2.steric), vmin=-0.1, vmax=0.1, cmap="RdBu_r")
plt.colorbar()
plt.title("Original - Reconstructed")
plt.show()
# ds['SLA_full'] = ds['SLA']+deep
#%% add too all datasest
deep = from_trend_to_ts(np.array(ds2.steric), tdec)  # mm
deep[np.isnan(deep)] = 0
steric_up = np.array(ds.SLA)
steric_full = np.array(steric_up + (deep / 1000))  # meters

tr, _ = sl.get_reg_trend(tdec, steric_full[-1] * 1000, lat, lon)
plt.pcolor(tr, vmin=-5, vmax=5, cmap="RdBu_r")
plt.colorbar()
plt.title(str(np.nanmean(tr)))
#%% make dataset
ds = xr.Dataset(
    data_vars={
        "steric_up": (("names", "time", "lat", "lon"), steric_up),
        "steric_deep": (("time", "lat", "lon"), deep),
        "steric_full": (("names", "time", "lat", "lon"), steric_full),
    },
    coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time, "names": ds.names},
)
ds.attrs["units"] = "meters (m)"
ds.attrs["description"] = "Steric sea-level height, full column (m)"
ds.attrs["time_mean"] = "Removed time mean from 2005-2015 (full years)"
ds.attrs["deep_steric"] = "Deep steric from Purkey & Johnson updated (at 2000m)"
ds.attrs["script"] = "SLB-steric_full.py"
#% % save
path_save = "/Volumes/LaCie_NIOZ/data/budget/"
ds.to_netcdf(path_save + "steric_full.nc")
