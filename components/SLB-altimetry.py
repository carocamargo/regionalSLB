#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 09:25:58 2022

Get altimetry datasets
& 
Regrid
@author: ccamargo

Modified on Mon Nov 21 2022: CSIRO without GIA trend
"""

#% %
import os
import xarray as xr
import numpy as np

path_save = "/Volumes/LaCie_NIOZ/data/altimetry/"
#%% SLcci
"""
The data can be accessed through ftp: . address: ftp.esa-sealevel-cci.org
. login : slcci
. passwd : slcci
"""
path = "/Volumes/SLB/raw_data/total/SL_cci/SeaLevel-ECV/V2.0_20161205/"

outfile = path_save + "slcci.nc"
dm = xr.open_mfdataset(path + "*nc", concat_dim="time")
dm.to_netcdf(outfile)

#%% AVISO - SSALTO/DUACS https://www.aviso.altimetry.fr/index.php?id=1526
"""
ftp-access.aviso.altimetry.fr/climatology/global/delayed-time/monthly_mean/msla_h/
user: caromlcamargo@gmail.com
psw: bNeGWt
downloaded on feb 1st 2022
"""
path = "/Volumes/SLB/raw_data/total/AVISO/v7/"
flist = sorted([file for file in os.listdir(path) if file.endswith(".nc")])
ds = xr.open_dataset(path + flist[0])
latdim = len(ds.latitude)
londim = len(ds.longitude)
timedim = len(flist)
sla = np.zeros((timedim, latdim, londim))
time = []
for i, f in enumerate(flist):
    print(i)
    ds = xr.open_dataset(path + f)
    sla[i] = np.array(ds.sla)
    time.append(np.array(ds.time)[0])

da = xr.Dataset(
    data_vars={"sla": (("time", "lat", "lon"), sla)},
    coords={"time": time, "lat": np.array(ds.latitude), "lon": np.array(ds.longitude)},
)
da.lat.attrs = ds.latitude.attrs
da.lon.attrs = ds.longitude.attrs
da.sla.attrs = ds.sla.attrs
da.time.attrs = ds.time.attrs
da.attrs = ds.attrs

outfile = path_save + "aviso.nc"
da.to_netcdf(outfile)
#%% CMEMS
"""
Dataset: SURFACE HEIGHTS AND DERIVED VARIABLES REPROCESSED (1993-ONGOING)
Product identifier: SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047
url: https://resources.marine.copernicus.eu/product-detail/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/INFORMATION
"ftp_url":"my.cmems-du.eu/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4-monthly/",
Variables:['sla']Time resolution: monthlyDownload date: 2021-09-10 18:20:34.057089}
"""

path = "/Volumes/LaCie_NIOZ/data/altimetry/world/CMEMS/sla/"
outfile = path_save + "cmems.nc"

command = "cdo mergetime {}*.nc {}".format(path, outfile)
os.system(command)

#%% CSIRO
"""
https://www.cmar.csiro.au/sealevel/sl_data_cmar.html
download date: 1st feb 2022
-IB correction applied, seasonal signal not removed, GIA correction applied
"""
path = "/Volumes/LaCie_NIOZ/data/altimetry/world/csiro/jb_iby_srn_gtn_giy.nc"
# CHANGE FOR NO GIA:
path = "/Volumes/LaCie_NIOZ/data/altimetry/world/csiro/jb_iby_srn_gtn_gin.nc"
    
ds = xr.open_dataset(path)
ds.to_netcdf(path_save + "csiro.nc")

#%% JPL MeASURES
"""
MEaSUREs Gridded Sea Surface Height Anomalies Version 1812

Command to download: wget -m https://podaac-opendap.jpl.nasa.gov/opendap/allData/merged_alt/L4/cdr_grid/
Url: https://podaac.jpl.nasa.gov/dataset/SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL1812

Citation: Zlotnicki, Victor; Qu, Zheng; Willis, Joshua. 2019. SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL1609. Ver. 1812. PO.DAAC, CA, USA. Dataset accessed [YYYY-MM-DD] at https://doi.org/10.5067/SLREF-CDRV2


Data accessed: 2021-09-09 (9th of September)
"""

path = "/Volumes/LaCie_NIOZ/data/altimetry/world/measures/"
years = np.arange(1992, 2020)
for year in years:
    flist = sorted(
        [
            file
            for file in os.listdir(path + "5day/")
            if file.endswith(".nc")
            and int(file.split("_")[-1][0:4]) == year
            and not file.startswith("._")
        ]
    )
    ds = xr.open_dataset(path + "5day/" + flist[0])
    latdim = len(ds.Latitude)
    londim = len(ds.Longitude)
    timedim = len(flist)
    sla = np.zeros((timedim, latdim, londim))
    sla_err = np.zeros((timedim, latdim, londim))

    time = []
    for i, f in enumerate(flist):
        print("{}/{}".format(i, timedim))
        ds = xr.open_dataset(path + "5day/" + f)
        sla[i] = np.transpose(np.array(ds.SLA[0]))
        sla_err[i] = np.transpose(np.array(ds.SLA_ERR[0]))
        time.append(np.array(ds.Time)[0])

    da = xr.Dataset(
        data_vars={
            "sla": (("time", "lat", "lon"), sla),
            "sla_err": (("time", "lat", "lon"), sla_err),
        },
        coords={
            "time": time,
            "lat": np.array(ds.Latitude),
            "lon": np.array(ds.Longitude),
        },
    )

    da.lat.attrs = ds.Latitude.attrs
    da.lon.attrs = ds.Longitude.attrs
    da.sla.attrs = ds.SLA.attrs
    da.sla_err.attrs = ds.SLA_ERR.attrs

    da.time.attrs = ds.Time.attrs
    da.attrs = ds.attrs
    da.attrs["files"] = flist
    print("Saving for {}".format(year))
    outfile = path + "years/" + "{}.nc".format(year)
    # outfile_5days = '/Volumes/LaCie_NIOZ/data/altimetry/world/measures/measures_5day_v2.nc'
    da.to_netcdf(outfile)
    outfile2 = path + "years/monavg/" + "{}.nc".format(year)
    print("Monthly mean {}".format(year))
    os.system("cdo monmean {} {}".format(outfile, outfile2))

dsm = xr.open_mfdataset(path + "years/monavg/" + "*nc", combine="by_coords")
dsm.to_netcdf(path_save + "measures.nc")
# outfile_5days = '/Volumes/LaCie_NIOZ/data/altimetry/world/measures/measures_5day.nc'
# dsm.to_netcdf(outfile_5days)
# outfile_month = path_save+'measures_month.nc'
# os.system('cdo monmean {} {}'.format(outfile_5days,outfile_month))
# #%%
# outfile = '/Volumes/LaCie_NIOZ/data/altimetry/world/measures/MSLA_MEASURES_merged_1992-2019_5day.nc'
# command = 'cdo mergetime {}*.nc {}'.format(path,outfile)
# os.system(command)

# # monthly mean
# infile=outfile
# outfile = path_save+'measures2.nc'
# command = 'cdo monmean {} {}'.format(infile,outfile)
# os.system(command)
# xr.open_dataset(outfile)
#%% regrid to 1degree
flist = [file for file in os.listdir(path_save) if file.endswith(".nc")]
for file in flist:
    os.system(
        "cdo -L remapbil,r360x180 "
        + str(path_save + file)
        + " "
        + str(path_save + "regrid/" + file)
    )
