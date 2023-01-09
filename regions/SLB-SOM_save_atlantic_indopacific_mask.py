#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:18:15 2022

@author: ccamargo
"""
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

cmap = cmocean.cm.balance
import matplotlib.pyplot as plt


def plot_map(
    data,
    plot_type="pcolor",
    lon=np.arange(0, 360),
    lat=np.arange(-90, 90),
    cmap="tab10",
    cmin=0,
    cmax=9,
    fsize=(15, 10),
    proj="robin",
    land=True,
    grid=True,
    title="",
    clabel="",
    lon0=0,
    landcolor="papayawhip",
    extent=False,
    interval=0.1,
    sig=False,
    unc=0.3,
):

    lon[-1] = 360
    plt.figure(figsize=(15, 10), dpi=100)
    if proj == "robin":
        proj = ccrs.Robinson(central_longitude=lon0)
    else:
        proj = ccrs.PlateCarree()
    ax = plt.subplot(
        111,
        projection=proj
        # Mercator()
    )
    # ax.background_img(name='pop', resolution='high')
    if extent:
        ax.set_extent(extent, ccrs.PlateCarree())
    else:
        ax.set_global()
    ##             min_lon,,max_lon,minlat,maxlat
    if plot_type == "pcolor":
        mm = ax.pcolormesh(
            lon,
            lat,
            data,
            vmin=cmin,
            vmax=cmax,
            transform=ccrs.PlateCarree(),
            # cmap='Spectral_r'
            cmap=cmap,
        )
    if plot_type == "contour":
        lv = np.arange(cmin, cmax + interval, interval)
        mm = plt.contourf(
            lon, lat, data, levels=lv, transform=ccrs.PlateCarree(), cmap=cmap
        )

        plt.pcolormesh(
            lon,
            lat,
            data,
            vmin=cmin,
            vmax=cmax,
            zorder=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
        )
    if sig:
        Z_insg = np.array(data)
        Z_insg[np.abs(data) > (np.array(unc))] = 1
        Z_insg[np.abs(data) < (np.array(unc))] = 0
        stip = plt.contourf(
            lon,
            lat,
            Z_insg,
            levels=[-1, 0, 1],
            colors="none",
            hatches=[None, "..."],
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

    if land:

        # ax.add_feature(cfeature.LAND, facecolor=landcolor)

        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "land", "50m", edgecolor="gray", facecolor=landcolor
            )
        )

        # resol = '50m'  # use data at this scale

        # land = cfeature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        # ax.add_feature(land, facecolor='beige')

    # d01 box
    if grid:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER
    plt.colorbar(mm, label=clabel, orientation="horizontal", shrink=0.9)
    plt.title(title, fontsize=20)

    #    plt.show()
    # plt.close()
    return


#%% som clusters
path = "//Volumes/LaCie_NIOZ/budget/regions/som/"
flist = [
    "som_3x3_alt_1993_2019_n10_sig2_ep_atlantic.nc",
    "som_3x3_alt_1993_2019_n10_sig2_ep_indopacific.nc",
]

mask = np.zeros((180, 360))
ds = xr.open_dataset(path + flist[0])
ds.bmu_map.max()
n_neurons = len(ds.neurons)
mask = np.array(ds.bmu_map)
mask[np.isnan(mask)] = 0
ds = xr.open_dataset(path + flist[1])
mask2 = np.array(ds.bmu_map + n_neurons)
mask2[np.isnan(mask2)] = 0
mask = np.array(mask + mask2)
mask[np.where(mask == 0)] = np.nan

title = "som_3x3_alt_1993_2019_n10_sig2_ep_atlantic_indopacific"
plot_map(mask, lon0=205, fsize=(15, 10), cmap="jet", title=title, cmax=18, grid=False)

mask_comb = np.array(mask)
x = 4
y = 5
plt.figure(figsize=(20, 10))
for i in range(0, int(np.nanmax(mask_comb))):
    icluster = i + 1
    ax = plt.subplot(y, x, icluster, projection=ccrs.Robinson(central_longitude=210))
    ax.set_global()
    mask = np.array(mask_comb)
    mask[np.where(mask != icluster)] = np.nan
    mm = ax.pcolormesh(
        ds.lon,
        ds.lat,
        mask,
        vmin=0,
        vmax=x * y,
        transform=ccrs.PlateCarree(),
        # cmap='Spectral_r'
        cmap="jet",
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "50m", edgecolor="gray", facecolor="papayawhip"
        )
    )
    plt.title("Cluster {}".format(icluster))

da = xr.Dataset(
    data_vars={"mask": (("lat", "lon"), mask_comb)},
    coords={"lat": ds.lat, "lon": ds.lon},
)
da.to_netcdf(path + title + ".nc")
