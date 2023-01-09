##############################
#### Changing parameters
##############################

ifolder = 0
pwd = "/export/lv1/user/ccamargo/data/"
dataset = "alt.nc"
var = "SLA"
periods = [(1993, 2017), (2003, 2016), (1993, 2016), (2005, 2015)]
t0 = 2003
t1 = 2017


##############################
#### load libraries
##############################

import xarray as xr
import numpy as np
import utils_hec as hec
import os
import pandas as pd

# import cmocean as cmo
# from numba import jit
import datetime as dt
from datetime import datetime


def get_dectime(time):
    t = [
        datetime.utcfromtimestamp(
            (t - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        )
        for t in time
    ]
    t = [t.timetuple().tm_year + (t.timetuple().tm_yday / 365) for t in t]
    return np.array(t)


##############################
#### Start Code
##############################

# open dataset
ds = xr.open_dataset(pwd + dataset)
ds

# get dimensions
lat = np.array(ds.lat)
lon = np.array(ds.lon)
dimlat = len(lat)
dimlon = len(lon)
lon, lat = np.meshgrid(lon, lat)
lon = np.hstack(lon)
# lon3=lon.flatten() # same thing
lat = np.hstack(lat)  # or lat.flatten()

# getnames
names = sorted(np.array(ds.names))
dimname = len(names)

# select only th working variable
ds = ds[var]

# set path to hector
path_to_hec = "/export/lv1/user/ccamargo/dump/" + str(ifolder) + "/"
os.chdir(path_to_hec)

sp = 30
# select time
for period in periods:
    t0, t1 = period
    print("Trend from {}-{}".format(t0, t1))
    da = ds.sel(time=slice("{}-01-01".format(t0), "{}-12-31".format(t1)))
    time = get_dectime(np.array(da.time))
    dimtime = len(time)
    time = get_dectime(
        np.array(pd.date_range(start="1/1/{}".format(t0), periods=dimtime, freq="M"))
    )

    # loop over datasets:
    for iname, name in enumerate(names):
        da2 = da.sel(names=name)

        data = np.array(da2.data * 1000)  # mm
        data2 = data.ravel().reshape(dimtime, dimlat * dimlon)

        # test diff noise models:
        NM = [
            "WN",
            "PL",
            "PLWN",
            "AR1",
            "AR5",
            "AR9",
            "ARF",
            "GGMWN",  # 'FNWN','RWFNWN'
        ]
        dimNM = len(NM)

        # allocate empty variables
        bic = np.full_like(np.zeros((dimNM, dimlat * dimlon)), np.nan)
        bic_c = np.full_like(bic, np.nan)
        bic_tp = np.full_like(bic, np.nan)
        aic = np.full_like(bic, np.nan)
        logL = np.full_like(bic, np.nan)
        N = np.full_like(bic, np.nan)
        tr = np.full_like(bic, np.nan)
        tr_err = np.full_like(bic, np.nan)

        # path_to_save ='//export/lv1/user/ccamargo/OM/hector/'+str(t0)+'-2016/'

        inm = 0
        n = NM[inm]
        # for inm, n in enumerate(NM):
        # loop over NM:
        for inm, n in enumerate(NM):
            print(n)
            #% % Loop over each lat-lon
            go = dt.datetime.now()
            for ilon in range(len(lon)):
                if np.any(np.isfinite(data2[:, ilon])):  # check if we have data:
                    # print('{}/{}'.format(ilon,len(lon)))
                    # print(ilon)
                    x = data2[:, ilon]
                    # create a .mom file for it:
                    hec.ts_to_mom(
                        x[np.isfinite(x)],
                        time[np.isfinite(x)],
                        sp=sp,
                        path=str(path_to_hec + "raw_files/"),
                        name=str(name),
                        ext=".mom",
                    )

                    # Create a .ctl file:
                    hec.create_estimatetrend_ctl_file(
                        name, n, sp=sp, GGM_1mphi=6.9e-07, LikelihoodMethod="FullCov"
                    )

                    # Run estimatetrend (hector)
                    os.system("estimatetrend > estimatetrend.out")

                    # get results:
                    out = hec.get_results()
                    # save results:
                    if out[0] != None:
                        tr[inm, ilon] = out[0]
                        tr_err[inm, ilon] = out[1]
                        N[inm, ilon] = out[2]
                        logL[inm, ilon] = out[3]
                        aic[inm, ilon] = out[4]
                        bic[inm, ilon] = out[5]
                        bic_c[inm, ilon] = out[6]
                        bic_tp[inm, ilon] = out[7]

            # save by NM
            dn = xr.Dataset(
                data_vars={
                    "trend": (("lat", "lon"), tr[inm, :].reshape(dimlat, dimlon)),
                    "unc": (("lat", "lon"), tr_err[inm, :].reshape(dimlat, dimlon)),
                    "bic": (("lat", "lon"), bic[inm, :].reshape(dimlat, dimlon)),
                    "aic": (("lat", "lon"), aic[inm, :].reshape(dimlat, dimlon)),
                    "logL": (("lat", "lon"), logL[inm, :].reshape(dimlat, dimlon)),
                    "bic_c": (("lat", "lon"), bic_c[inm, :].reshape(dimlat, dimlon)),
                    "bic_tp": (("lat", "lon"), bic_tp[inm, :].reshape(dimlat, dimlon)),
                    "N": (("lat", "lon"), N[inm, :].reshape(dimlat, dimlon)),
                },
                coords={
                    "nm": NM[inm],
                    "period": ["{}-{}".format(t0, t1)],
                    "fname": str(name),
                    "lat": ds.lat,
                    "lon": ds.lon,
                },
            )
            dn.attrs[
                "metadata"
            ] = " sea-level trends in mm/y from {} to {}, obtained with Hector".format(
                t0, t1
            )
            path_to_save = "{}/hector/{}-{}/".format(pwd, t0, t1)
            # os.system(' cd {}'.format(path_to_save))
            dn.to_netcdf(path_to_save + "ALT_" + str(name) + "_{}_".format(n) + ".nc")
            print("saved for: " + str(name) + " {}".format(n))

        print(dt.datetime.now() - go)
        # save all
        print("saving for: " + str(name))
        dx = xr.Dataset(
            data_vars={
                "trend": (("nm", "lat", "lon"), tr.reshape(len(NM), dimlat, dimlon)),
                "unc": (("nm", "lat", "lon"), tr_err.reshape(len(NM), dimlat, dimlon)),
                "bic": (("nm", "lat", "lon"), bic.reshape(len(NM), dimlat, dimlon)),
                "aic": (("nm", "lat", "lon"), aic.reshape(len(NM), dimlat, dimlon)),
                "logL": (("nm", "lat", "lon"), logL.reshape(len(NM), dimlat, dimlon)),
                "bic_c": (("nm", "lat", "lon"), bic_c.reshape(len(NM), dimlat, dimlon)),
                "bic_tp": (
                    ("nm", "lat", "lon"),
                    bic_tp.reshape(len(NM), dimlat, dimlon),
                ),
                "N": (("nm", "lat", "lon"), N.reshape(len(NM), dimlat, dimlon)),
            },
            coords={
                "nm": NM,
                "period": ["{}-{}".format(t0, t1)],
                "fname": str(name),
                "lat": ds.lat,
                "lon": ds.lon,
            },
        )
        dx.attrs[
            "metadata"
        ] = " sea-level trends in mm/y from {} to {}, obtained with Hector".format(
            t0, t1
        )
        path_to_save = "{}/hector/{}-{}/".format(pwd, t0, t1)
        # os.system(' cd {}'.format(path_to_save))
        dx.to_netcdf(path_to_save + "ALT_" + str(name) + ".nc")
        print("saved for: " + str(name))
