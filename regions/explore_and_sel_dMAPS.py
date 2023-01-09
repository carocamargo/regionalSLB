#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:10:48 2022

@author: ccamargo
"""
#
import xarray as xr
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
# from cartopy import crs as ccrs , feature as cfeature
# from matplotlib.gridspec import GridSpec
# from matplotlib.colors import ListedColormap
# from matplotlib import colors
# import cmocean as cm
# from cmcrameri import cm as cmf


# user defined functions
import sys
sys.path.append("/Users/ccamargo/Documents/github/SLB/")
from utils_SLB import plot_nmi_matrix, plot_dMaps_output, calc_nmi_matrix,get_domain_map
# import utils_dMAPS as dmaps


#%%
path =  "/Volumes/LaCie_NIOZ/reg/world/dmaps/world_v3/"

geofile = "MSLA_CMEMS_glb_merged_1993-2019_detrend_deseason_300kmfilter_res1deg_66lat.nc"

i = 1 # resolution
nmi = calc_nmi_matrix(path, 2, np.arange(4, 25))  # ,
plot_nmi_matrix(
    nmi, np.arange(4, 25), fname=path + "plot/nmi_matrix_res{}.png".format(i)
)
#%% for each k
ks = np.arange(4,25)
ds=xr.open_dataset(path+geofile)
plot=False
masks = np.zeros((len(ks),len(ds.lat),len(ds.lon)))
for i,k in enumerate(ks):
    print(k)
    folder = "k{}".format(k)

    field = np.load(path + folder + "/domain_identification/domain_maps.npy")

    domain_map = get_domain_map(field)
    masks[i] = domain_map

    if plot:
        plot_dMaps_output(
            geofile=path + geofile,
            fpath=path + "k{}/".format(k),
            output="domain",
            cmap="jet",
            title="domain_map_res{}_k{}_cmapjet".format(i, str(k).zfill(2)),
            outpath=path + "plot/",  # "k{}/".format(k)+'plot/',
            show_seeds="homogeneity",
        )
    
        plot_dMaps_output(
            geofile=path + geofile,
            fpath=path + "k{}/".format(k),
            title="homogeneity_map_res{}_k{}".format(i, str(k).zfill(2)),
            output="homogeneity",
            outpath=path + "plot/",
            show_seeds="homogeneity",
        )

        plt.pcolor(
            domain_map, cmap="prism", vmin=np.nanmin(domain_map), vmax=np.nanmax(domain_map)
        )
        plt.show()

#%%
da = xr.Dataset(
    data_vars={'mask':(('k','lat','lon'),masks)},
    coords={
        'k':ks,
        "lat": ds.lat,
        "lon": ds.lon,
    },
)
da.attrs['metadata'] = 'dMAPS clusters, with varying k (4 to 24)'
da.to_netcdf(path+'clusters.nc')
#%% landmask
# import xarray as xr
ds = xr.open_dataset(path + geofile)
mask = np.array(ds.sla[0, :, :])
mask[np.isfinite(mask)] = 1

mask66 = np.array(ds.sla[0, :, :].where((ds.lat > -66) & (ds.lat < 66), np.nan))
mask66[np.isfinite(mask66)] = 1
#%%
# import matplotlib.pyplot as plt
kline = np.arange(4, 25)
number_nans = np.zeros((len(kline)))
number_clustered_pixels = np.zeros((len(kline)))
perc_nans = np.zeros((len(kline)))
perc_clustered_pixels = np.zeros((len(kline)))
number_clusters = np.zeros((len(kline)))

for i, k in enumerate(kline):
    domains = np.load("{}k{}/domain_identification/domain_maps.npy".format(path, k))
    number_clusters[i] = domains.shape[0]
    domains = np.sum(domains, axis=0) * mask66
    number_nans[i] = len(domains[domains == 0])
    number_clustered_pixels[i] = len(domains[domains == 1])
    perc_nans[i] = len(domains[domains == 0]) / len(domains[np.isfinite(domains)])
    perc_clustered_pixels[i] = len(domains[domains == 1]) / len(
        domains[np.isfinite(domains)]
    )


#%%
plt.figure()
plt.scatter(kline, number_clusters, c=[perc_clustered_pixels * 100])
plt.xlabel("k")
plt.ylabel("number clusters")
plt.colorbar(label="perc of clustered pixels")
plt.show()

plt.figure()
plt.scatter(kline, number_clusters, c=[perc_nans * 100])
plt.xlabel("k")
plt.ylabel("number clusters")
plt.colorbar(label="% of not clustered pixels")
plt.show()

#%%
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection="3d")

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(kline, number_clusters, perc_nans, 'gray')
sc = ax.scatter3D(kline, perc_nans, number_clusters, c=perc_nans, cmap="plasma", s=150)
# plt.xlabel('k')
# plt.ylabel('number clusters')
# plt.zlabel('% not clustered pixels')
ax.set_xlabel("K")
ax.set_zlabel("number clusters")
ax.set_ylabel("% of not clusterd pixels")
# Add a color bar which maps values to colors.
fig.colorbar(sc, shrink=0.5, aspect=5)

#%%

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection="3d")

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
ax.plot3D(
    kline[number_clusters < 120],
    perc_nans[number_clusters < 120],
    number_clusters[number_clusters < 120],
    "gray",
)
sc = ax.scatter3D(
    kline[number_clusters < 120],
    perc_nans[number_clusters < 120],
    number_clusters[number_clusters < 120],
    c=perc_nans[number_clusters < 120],
    cmap="plasma",
    s=150,
)
# plt.xlabel('k')
# plt.ylabel('number clusters')
# plt.zlabel('% not clustered pixels')
ax.set_xlabel("K")
ax.set_zlabel("number clusters")
ax.set_ylabel("% of not clusterd pixels")
# Add a color bar which maps values to colors.
fig.colorbar(sc, shrink=0.5, aspect=5)

#%%
# import pandas as pd
df = pd.DataFrame(
    {
        "k": kline,
        "n_clusters": number_clusters,
        "perc_not_cluster": perc_nans,
        "perc_clusters": perc_clustered_pixels,
    }
)

print(df[(df.perc_clusters > 0.5) & (df.perc_not_cluster < 0.5)])
#%% save k=5 and k=23
da = da.sel(k=[5,23])
da.to_netcdf('/Volumes/LaCie_NIOZ/budget/regions/dmaps/dmaps_k5_k23.nc')

#%%
k=5
domains = np.load("{}k{}/domain_identification/domain_maps.npy".format(path, k))
#%%
for icluster in range(0,len(domains)):
    # mask = np.array(domains)
    # mask[mask!=icluster]=np.nan
    plt.pcolor(domains[icluster])
    plt.title('cluster = {}'.format(icluster))
    plt.show()






