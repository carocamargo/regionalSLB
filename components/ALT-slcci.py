#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:38:14 2022

@author: ccamargo
"""

import xarray as xr
import numpy as np
path = '/Volumes/LaCie_NIOZ/data/altimetry/trends/1993-2017/'
t0=1993
t1=2017
name = 'slcci'
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
dimlat=180
dimlon=360
# allocate empty variables
bic = np.full_like(np.zeros((dimNM, dimlat, dimlon)), np.nan)
bic_c = np.full_like(bic, np.nan)
bic_tp = np.full_like(bic, np.nan)
aic = np.full_like(bic, np.nan)
logL = np.full_like(bic, np.nan)
N = np.full_like(bic, np.nan)
trend = np.full_like(bic, np.nan)
unc = np.full_like(bic, np.nan)

# % %loop over NM:
for inm, n in enumerate(NM):
    print(n)
    ds = xr.open_dataset(path+'ALT_'+name+'_{}_.nc'.format(n))
    bic[inm] = np.array(ds['bic'])
    bic_c[inm] = np.array(ds['bic_c'])
    bic_tp[inm] = np.array(ds['bic_tp'])
    aic[inm] = np.array(ds['aic'])
    logL[inm] = np.array(ds['logL'])
    trend[inm] = np.array(ds['trend'])
    unc[inm] = np.array(ds['unc'])
    N[inm] = np.array(ds['N'])

#%%
dx = xr.Dataset(
    data_vars={
        "trend": (("nm", "lat", "lon"), trend),
        "unc": (("nm", "lat", "lon"), unc),
        "bic": (("nm", "lat", "lon"), bic),
        "aic": (("nm", "lat", "lon"), aic),
        "logL": (("nm", "lat", "lon"), logL),
        "bic_c": (("nm", "lat", "lon"), bic_c),
        "bic_tp": ( ("nm", "lat", "lon"),bic_tp),
        "N": (("nm", "lat", "lon"), N),
    },
    coords={
        "nm": NM,
        "period": ds.period,
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

# os.system(' cd {}'.format(path_to_save))
dx.to_netcdf(path + "ALT_" + str(name) + ".nc")
    
    