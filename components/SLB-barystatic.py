#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:05:49 2022

@author: ccamargo
"""

import pandas as pd
import numpy as np
import xarray as xr

periods = [# (2005, 2016), 
           (1993, 2018), (2003, 2017)]
reconstr = ["JPL", "CSR", "IMB+WGP", "IMB+GWB+ZMP", "UCI+WGP", "UCI+GWB+ZMP"]
trends = np.zeros((len(reconstr), len(periods)))
uncs = np.zeros((len(reconstr), len(periods)))
das = []
dimlat = 180
dimlon = 360
names = []
for j, title in enumerate(reconstr):
    print(title)
    trends = np.zeros((len(periods), dimlat, dimlon))
    uncs = np.zeros((len(periods), dimlat, dimlon))
    name = title.replace("+", "_")
    names.append(name)
    for i, period in enumerate(periods):
        #% %
        print("period: {}".format(period))
        # period=periods[-1]
        t0 = period[0]
        t1 = period[1] - 1
        # print('Open final dataset')
        path = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results_final/{}-{}/".format(
            t0, t1
        )
        # df= pd.read_pickle(path+'OM_reconstructions_ASL_{}-{}.p'.format(t0,t1))
        df = pd.read_pickle(path + "OM_reconstructions_{}-{}_ASL.p".format(t0, t1))

        trends[i] = np.array(df["{}_trend_tot".format(title)]).reshape(dimlat, dimlon)
        uncs[i] = np.array(df["{}_unc_tot".format(title)]).reshape(dimlat, dimlon)

    if j == 0:
        ds = xr.Dataset(
            data_vars={
                "sla_{}".format(name): (("periods", "lat", "lon"), trends),
                "sla_unc_{}".format(name): (("periods", "lat", "lon"), uncs),
            },
            coords={
                "periods": [
                    "{}-{}".format(period[0], period[1] - 1) for period in periods
                ],
                "lat": np.array(df["lat"]).reshape(dimlat, dimlon)[:, 0],
                "lon": np.array(df["lon"]).reshape(dimlat, dimlon)[0, :],
            },
        )
    else:
        ds["sla_{}".format(name)] = (["periods", "lat", "lon"], trends)
        ds["sla_unc_{}".format(name)] = (["periods", "lat", "lon"], uncs)

data = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
data2 = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
for i, v in enumerate(names):
    data[i] = np.array(ds["sla_" + v])
    data2[i] = np.array(ds["sla_unc_" + v])
    #% %
ds["sla_ENS"] = (["periods", "lat", "lon"], np.nanmean(data, axis=0))
ds["sla_unc_ENS"] = (["periods", "lat", "lon"], np.nanmean(data2, axis=0))

names.append("ENS")
data = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
data2 = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
for i, v in enumerate(names):
    data[i] = np.array(ds["sla_" + v])
    data2[i] = np.array(ds["sla_unc_" + v])

# var = [key for key in ds.variables if key.split('_')[0]=='sla' and len(key.split('_'))==2]
# name = [v.split('_')[-1] for v in var]
# data = np.zeros((len(var),len(ds.time),len(ds.lat),len(ds.lon)))
# for i,v in enumerate(var):
#     data[i] = np.array(ds[v])

ds = ds.assign_coords({"names": names})
ds["SLA"] = (["names", "periods", "lat", "lon"], data)
ds["SLA_UNC"] = (["names", "periods", "lat", "lon"], data2)

path_save = "/Volumes/LaCie_NIOZ/data/budget/"
ds.to_netcdf(path_save + "barystatic_asl_trends.nc")
path_save = "/Volumes/LaCie_NIOZ/data/budget/trends/"
ds.to_netcdf(path_save + "barystatic.nc")

#%% RSL
# trends = np.zeros((len(reconstr), len(periods)))
# uncs = np.zeros((len(reconstr), len(periods)))
# das = []
# dimlat = 180
# dimlon = 360
# names = []
# for j, title in enumerate(reconstr):
#     print(title)
#     trends = np.zeros((len(periods), dimlat, dimlon))
#     uncs = np.zeros((len(periods), dimlat, dimlon))
#     name = title.replace("+", "_")
#     names.append(name)
#     for i, period in enumerate(periods):
#         #% %
#         print("period: {}".format(period))
#         # period=periods[-1]
#         t0 = period[0]
#         t1 = period[1] - 1
#         # print('Open final dataset')
#         path = "/Volumes/LaCie_NIOZ/data/barystatic/revisions/results_final/RSL/"
#         df = pd.read_pickle(path + "OM_reconstructions_ASL_{}-{}.p".format(t0, t1))
#         # df= pd.read_pickle(path+'OM_reconstructions_{}-{}.p'.format(t0,t1))
#         trends[i] = np.array(df["{}_trend_tot".format(title)]).reshape(dimlat, dimlon)
#         uncs[i] = np.array(df["{}_unc_tot".format(title)]).reshape(dimlat, dimlon)

#     if j == 0:
#         ds = xr.Dataset(
#             data_vars={
#                 "sla_{}".format(name): (("periods", "lat", "lon"), trends),
#                 "sla_unc_{}".format(name): (("periods", "lat", "lon"), uncs),
#             },
#             coords={
#                 "periods": [
#                     "{}-{}".format(period[0], period[1] - 1) for period in periods
#                 ],
#                 "lat": np.array(df["lat"]).reshape(dimlat, dimlon)[:, 0],
#                 "lon": np.array(df["lon"]).reshape(dimlat, dimlon)[0, :],
#             },
#         )
#     else:
#         ds["sla_{}".format(name)] = (["periods", "lat", "lon"], trends)
#         ds["sla_unc_{}".format(name)] = (["periods", "lat", "lon"], uncs)

# data = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
# data2 = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
# for i, v in enumerate(names):
#     data[i] = np.array(ds["sla_" + v])
#     data2[i] = np.array(ds["sla_unc_" + v])
#     #% %
# ds["sla_ENS"] = (["periods", "lat", "lon"], np.nanmean(data, axis=0))
# ds["sla_unc_ENS"] = (["periods", "lat", "lon"], np.nanmean(data2, axis=0))

# names.append("ENS")
# data = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
# data2 = np.zeros((len(names), len(ds.periods), len(ds.lat), len(ds.lon)))
# for i, v in enumerate(names):
#     data[i] = np.array(ds["sla_" + v])
#     data2[i] = np.array(ds["sla_unc_" + v])

# # var = [key for key in ds.variables if key.split('_')[0]=='sla' and len(key.split('_'))==2]
# # name = [v.split('_')[-1] for v in var]
# # data = np.zeros((len(var),len(ds.time),len(ds.lat),len(ds.lon)))
# # for i,v in enumerate(var):
# #     data[i] = np.array(ds[v])

# ds = ds.assign_coords({"names": names})
# ds["SLA"] = (["names", "periods", "lat", "lon"], data)
# ds["SLA_UNC"] = (["names", "periods", "lat", "lon"], data2)

# path_save = "/Volumes/LaCie_NIOZ/data/budget/"
# ds.to_netcdf(path_save + "barystatic_rsl_trends.nc")
