#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:27:04 2022

@author: ccamargo
"""

def agree_test(a,a_sig,b,b_sig):
    '''
    Do measurmenets agree within uncertaities?
    1 = agree
    0 = disagree

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    a_sig : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    b_sig : TYPE
        DESCRIPTION.

    Returns
    -------
    agree : TYPE
        DESCRIPTION.

    '''
    
    if  a + a_sig > b - b_sig and a - a_sig < b + b_sig:
        agree = 1
    else:
        agree = 0  
    
    return agree

#%%
def zeta_test(a,a_sig,b,b_sig):
    '''
    Test whether values agree acocoridng to zeta test
    1 = agree
    0 = disagree
    0.5 = tension zone

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    a_sig : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    b_sig : TYPE
        DESCRIPTION.

    Returns
    -------
    score : TYPE
        DESCRIPTION.

    '''
    import numpy as np
    zeta = np.abs( 
        (a-b)/ (np.sqrt(a_sig**2 + b_sig**2))
        )
    
    if zeta < 1:
        score = 1 
    elif 1 < zeta <3:
        score = 0.5
    else:
        score = 0
    return score


#%% 
def unc_test(a,a_sig,b,b_sig, method = 'square', how = 'sum'):
    import numpy as np
    res = np.array(a - b)
    if method =='square':
        if how =='sum':
            res_unc = np.sqrt( a_sig**2 + b_sig**2)  
        elif how =='subtract':
            res_unc= np.sqrt(a_sig**2 - b_sig**2)
        else:
            raise('Method not recongized')
    elif method =='linear':
        if how =='sum':
            res_unc = np.array(a_sig + b_sig) 
        elif how =='subtract':
            res_unc= np.array(a_sig - b_sig) 
        else:
            raise('Method not recongized')   
    else:
        raise ('Method not recognized')
        
    if np.abs(res) < res_unc:
        agree = 1
    else:
        agree = 0
        
    return agree
#%%
def get_dectime(time):
    from datetime import datetime
    import numpy as np
    t = [datetime.utcfromtimestamp((t- np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')) 
         for t in time]
    t = [t.timetuple().tm_year + (t.timetuple().tm_yday/365) for t in t]
    return np.array(t)
#%%
def sel_best_NM(IC, field, field2):
    import numpy as np
    """Given a field that has dimensions of len(nm),len(lat),len(lon),
    and an information critera that has the same dimensions, select the best noise model for this field.
    Return the selected field and the scoring of the noise models, both with dimensions (len(lat),len(lon))
    """
    dimnm, dimlat, dimlon = field.shape
    field = field.reshape(dimnm, dimlat * dimlon)
    field2 = field2.reshape(dimnm, dimlat * dimlon)

    mask = np.array(field[0, :])
    IC = IC.reshape(dimnm, dimlat * dimlon)

    best_field = np.zeros((dimlat * dimlon))
    best_field.fill(np.nan)
    score = np.full_like(best_field, np.nan)
    best_field2 = np.full_like(best_field, np.nan)

    for icoord in range(dimlat * dimlon):
        if np.isfinite(mask[icoord]):
            target = IC[:, icoord]
            ic = np.zeros((dimnm))
            logic = np.zeros((dimnm))

            for inm in range(dimnm):
                logic[inm] = np.exp((np.nanmin(target) - target[inm]) / 2)
                if logic[inm] > 0.5:
                    ic[inm] = 1
            score[icoord] = int(np.where(ic == np.nanmax(ic))[0][0])

            best_field[icoord] = field[int(score[icoord]), icoord]
            best_field2[icoord] = field2[int(score[icoord]), icoord]

    return (
        score.reshape(dimlat, dimlon),
        best_field.reshape(dimlat, dimlon),
        best_field2.reshape(dimlat, dimlon),
    )


#%%
def add_attrs(
    ds, variables=["lat", "lon"], latname="lat", lonname="lon", depthname="depth"
):
    if "lat" in variables:
        ds[latname].attrs = {
            "axis": "Y",
            "long_name": "latitude",
            "standard_name": "latitude",
            "unit_long": "degrees north",
            "units": "degrees_north",
        }
    if "lon" in variables:
        ds[lonname].attrs = {
            "axis": "X",
            "long_name": "longitude",
            "standard_name": "longitude",
            "step": 0.25,
            "unit_long": "degrees east",
            "units": "degrees_east",
        }
    if "depth" in variables:
        ds[depthname].attrs = {
            "axis": "Z",
            "long_name": "depth",
            "positive": "down",
            "standard_name": "depth",
            "unit_long": "meter",
            "units": "m",
        }

    return


#%%
def make_cmapnm():
    from matplotlib.colors import ListedColormap

    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
    col_dict = {
        1: "black",  # WN
        2: "palegoldenrod",  # PL
        3: "lightpink",  # PLWN
        4: "orange",  # AR1
        5: "teal",  # Ar5
        6: "darkmagenta",  # AR9
        7: "skyblue",  # ARf
        8: "crimson",  # GGM
    }
    
    # We create a colormar from our list of colors
    cmapnm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    return cmapnm

cmapnm=make_cmapnm()
#%%
def sum_linear(datasets, how = 'sum'):
    '''Sum or Subtract a list od datasets linearly.
    **Note**: Subtract mode works with only 2 datasets in the list. 
              Additional elements will be ignored!!
        '''
    import numpy as np
    field = np.zeros((datasets[0].shape))
    if how=='sum':
        for data in datasets:
            field = np.array(field + data)
    elif how=='subtract':
        field = np.array(datasets[0] - datasets[1])
    else:
        raise ('method not recognized')
    return field

def sum_square(datasets, how='sum'):
    '''Sum or Subtract a list od datasets squared. 
    That is, elements are squared, summed, and then removed the square root of the sum.  
    **Note**: Subtract mode works with only 2 datasets in the list. 
              Additional elements will be ignored!!
        '''
    import numpy as np
    field = np.zeros((datasets[0].shape))
    if how=='sum':
        for data in datasets:
            field = np.array(field + (data**2))
    elif how=='subtract':
        field = np.array( (datasets[0]**2) - (datasets[1]**2)  )
    else:
        raise ('method not recognized')
        
    return np.sqrt(np.abs(field))

#%%
import numpy as np
def plot_map2( data,
             plot_type = 'pcolor',
             lon=np.arange(0,360),lat=np.arange(-90,90),
             cmap='tab10',
             cmin=0,cmax=9,
             fsize=(15,10),
             proj='robin',
             land=True,
             grid=False,
             title='',
             clabel='',
             lon0=0,
             landcolor='papayawhip',
             extent = False,
             interval=0.1,
             sig=False,
             unc=0.3,
             hatch ='higher'
             ) :
    '''
    plot 1 variables
    '''
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    lon[-1]=360
    plt.figure(figsize=(15,10),dpi=100)
    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=lon0)
    else:
        proj=ccrs.PlateCarree()
    ax = plt.subplot(111, projection=proj
                     #Mercator()
                     )
    #ax.background_img(name='pop', resolution='high')
    if extent:    
        ax.set_extent(extent,ccrs.PlateCarree())
    else:
        ax.set_global()
    ##             min_lon,,max_lon,minlat,maxlat
    if plot_type=='pcolor':
        mm = ax.pcolormesh(lon,\
                           lat,\
                           data,
                           vmin=cmin, vmax=cmax, 
                           transform=ccrs.PlateCarree(),
                           #cmap='Spectral_r'
                           cmap=cmap
                          )
    if plot_type =='contour':
        lv=np.arange(cmin,cmax+interval,interval)
        mm=plt.contourf(lon,lat,data,levels=lv,
                  transform = ccrs.PlateCarree(),cmap=cmap)

        plt.pcolormesh(lon,lat,data,
                vmin=cmin,vmax=cmax,
                zorder=0,
                transform = ccrs.PlateCarree(),cmap=cmap)
    if sig:
        Z_insg = np.array(data)
        # Z_insg[np.isnan(data)]=-1
        if hatch =='higher':
            Z_insg[np.abs(data)>unc]=0
            Z_insg[np.abs(data)<unc]=1
        elif hatch =='smaler':
            Z_insg[np.abs(data)>unc]=1
            Z_insg[np.abs(data)<unc]=0
            
        ax.contourf(lon,lat, Z_insg, 
                    # levels=[ -1, 0, 0.5, 1],
                    levels=[-1, 0, 1],
                    
                    # levels = np.unique(Z_insg),
                    colors='none', 
                    # hatches=[None, None,'.'], 
                    hatches=[None, '...'], 
                    # 
                    transform = ccrs.PlateCarree(),
                    zorder=10)

            
    if land:
        
        
        # ax.add_feature(cfeature.LAND, facecolor=landcolor)
      
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='gray', facecolor=landcolor))
       
        # resol = '50m'  # use data at this scale
       
        # land = cfeature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        # ax.add_feature(land, facecolor='beige')

        
    # d01 box
    if grid:
        gl=ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # gl.xformatter = LONGITUDE_FORMATTER
        # gl.yformatter = LATITUDE_FORMATTER
    plt.colorbar(mm,label=clabel,orientation='horizontal',
                     shrink=0.9)
    plt.title(title,fontsize=20)

#    plt.show()
    # plt.close()    
    return

def cluster_mean(data,mask,
                 time=[0],
                 lat=[0],
                 lon=[0],
                 norm=False,
                 stats='mean',
                 method='linear'):
    ''' 
    Function to compute mean, min max time series of a dataset, given a mask
    If norm=True, then data is normalized by range before computing mean,min,max
    returns a time series which has mean on axis 0, min on axis 1 and max on axis 2
    '''
    import numpy as np
    import xarray as xr
    dims=data.shape
    if len(dims)==3:
        if len(lat)!=dims[1]:
            lat = np.arange(0,dims[1])
        if len(lon)!=dims[2]:
            lon=np.arange(0,dims[2])
        if len(time)!=dims[0]:
            time = np.arange(0,dims[0])
        da = xr.Dataset(data_vars={'data':(('time','lat','lon'),data*mask),                          
                               },
                    coords={'lat':lat,
                            'lon':lon,
                            'time':time,
                            }
                    )
        if norm:
            da['data']=da.data - da.data.mean()
    if len(dims)==2:
        if len(lat)!=dims[0]:
            lat = np.arange(0,dims[0])
        if len(lon)!=dims[1]:
            lon=np.arange(0,dims[1])

        da = xr.Dataset(data_vars={'data':(('lat','lon'),data*mask),                          
                               },
                    coords={'lat':lat,
                            'lon':lon,
                            }
                    )        
    
    if stats =='all':
        time_series=np.zeros((3,dims[0]))
        time_series[0,:] = da.data.mean(dim=('lat','lon')).data 
        time_series[1,:] = da.data.min(dim=('lat','lon')).data
        time_series[2,:] = da.data.max(dim=('lat','lon')).data
    elif stats=='mean':
        if method =='linear':
            time_series = da.data.mean(dim=('lat','lon')).data 
        elif method =='square':
            da['data'] = da.data **2 # square
            time_series = np.sqrt(da.data.sum(dim=('lat','lon')).data) / len(da.data.data[np.isfinite(da.data.data)])

    
    return time_series
#%% plot map subplot
import numpy as np
def plot_map_subplots( datasets,
             plot_type = 'pcolor',
             lon=np.arange(0,360),lat=np.arange(-90,90),
             cmap='tab10',
             cmin=0,cmax=9,
             fsize=(15,10),
             proj='robin',
             land=True,
             grid=False,
             titles='',
             clabel='',
             lon0=210,
             landcolor='papayawhip',
             extent = False,
             interval=0.1,
             sig=False,
             unc=0.3,
              fontsize=20,
              offset_y=0,
              nrow=False,ncol=False,
             ) :
    '''
    plot different variabls (ublots). Datasets is a list of data to be ploted
    '''
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    lon[-1]=360
    ndata = len(datasets)
    if not nrow:
        if ndata>2:
            if ndata%2==1:
                ncol = round(ndata/2)
                nrow = ndata - ncol +1
            else:
                nrow = ndata/2
                ncol = nrow
        else:
            nrow=1;ncol=2
    fig = plt.figure(figsize=fsize,dpi=100)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if proj=='robin':
        proj=ccrs.Robinson(central_longitude=lon0)
    else:
        proj=ccrs.PlateCarree()
    for idata,data in enumerate(datasets):
        ax = plt.subplot(nrow,ncol,idata+1, projection=proj
                         #Mercator()
                         )
        #ax.background_img(name='pop', resolution='high')
        if extent:    
            ax.set_extent(extent,ccrs.PlateCarree())
        else:
            ax.set_global()
        ##             min_lon,,max_lon,minlat,maxlat
        if plot_type=='pcolor':
            mm = ax.pcolormesh(lon,\
                               lat,\
                               data,
                               vmin=cmin, vmax=cmax, 
                               transform=ccrs.PlateCarree(),
                               #cmap='Spectral_r'
                               cmap=cmap
                              )
        if plot_type =='contour':
            lv=np.arange(cmin,cmax+interval,interval)
            mm=plt.contourf(lon,lat,data,levels=lv,
                      transform = ccrs.PlateCarree(),cmap=cmap)
    
            plt.pcolormesh(lon,lat,data,
                    vmin=cmin,vmax=cmax,
                    zorder=0,
                    transform = ccrs.PlateCarree(),cmap=cmap)
        if sig:
                Z_insg = np.array(data)
                Z_insg[np.abs(data)>(np.array(unc))]=1
                Z_insg[np.abs(data)<(np.array(unc))]=0
                plt.contourf(lon,lat, Z_insg, levels=[ -1,0, 1],
                            colors='none', hatches=[None,'...'], 
                            transform = ccrs.PlateCarree(),zorder=10)
    
                
        if land:
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='gray', facecolor=landcolor))
           
        # d01 box
        if grid:
            gl=ax.gridlines(draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
            # gl.xformatter = LONGITUDE_FORMATTER
            # gl.yformatter = LATITUDE_FORMATTER
        plt.title(titles[idata],fontsize=20)

    plt.tight_layout()
    # # fig.subplots_adjust(right=0.8)
    cbar_ax2 = fig.add_axes([0.25, 0.1+offset_y, 0.5, 0.04])
    cbar2=plt.colorbar(mm, cax=cbar_ax2,orientation='horizontal')
    cbar2.set_label(label=clabel,size=fontsize-5, family='serif')    
    cbar2.ax.tick_params(labelsize=fontsize-5) 

#    plt.show()
    # plt.close()    
    return
#%% plot map
def plot_map(lat, lon, data, seeds, title, cmap = 'viridis', alpha=1.,
             show_colorbar=True, show_grid=False, outpath=None, 
             labels=False, extent=None, pos_dict=None, draw_box=False,
             ax = None):
    """
    Plots a contourplot in a map with a title. If an output-path is specified,
    the plot is saved as <title>.png in the output directory. If this directory
    does not exist already, it will be created first.

    Parameters
    ----------
    lat : TYPE
        Latitude coordinates of the data-array.
    lon : TYPE
        Longitude coordinates of the data-array.
    data : array
        Array containing the data that will be plotted.
    seeds : array or None
        Array containing the locations of the seeds (cells without seed=0, 
        cells with seed=1) or None. If None, no seeds will be plotted.
    title : string
        Title of the plot [and output filename if outpath is specified].
    cmap : string, optional
        Colormap of the plot. The default is 'viridis'. 
    alpha : float, optional
        Alpha (opacity) of the domains.
    show_colorbar :  boolean, optional
        Whether to draw the colorbar or not. Default is True.
    show_grid :  boolean, optional
        Whether to draw gridlines and labels or not. Default is False.        
    outpath : string, optional
        Path where the plot will be saved. The default is None.
    labels : boolean, optional
        If true, labels will be drawn at each domain (mean of the position of 
        all non-nan values in data). The default is False.        
    extent : list, optional
        The extent of the map. The list must have the following structure: 
        [lon_min, lon_max, lat_min, lat_max]. If None is given, the entire 
        earth will be shown. The default is None.             
    pos_dict : dict, optinal
        Points on the map that will be highlighted with a cross (+) and a label
        indicating the locations latitude and longitude, if draw_box=False. 
        Must be in format {"lat": pos_lat, "lon": pos_lon} where pos_lat and 
        pos_lon are lists of coordinates in WGS84.
    draw_box : boolean, optional
        If True, the positions in pos_dict will be interpreted as outer points
        of an area that will be filled with a color. Default is False.
        

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    from  cartopy import crs as ccrs, feature as cfeature
    import os
    import numpy as np
    import cmocean
    
    if extent is None:
        crs = ccrs.PlateCarree(central_longitude=180)
    else:
        crs = ccrs.PlateCarree()
        lon_min, lon_max,  lat_min, lat_max = extent
        # convert longitude coordinates to 0-360 scale
        if lon_min < 0: lon_min = lon_min + 360
        if lon_max < 0: lon_max = lon_max + 360
    
    if ax is None:
        fig, ax =  plt.subplots(1,1,figsize=(12,8), dpi=300,
                                subplot_kw=dict(projection=crs))   
    else:
        ax=ax
    
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(1,1,1, 
    #                      projection = ccrs.PlateCarree(central_longitude=180))
    
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # ax.coastlines('110m', alpha=0.1)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), 
                   facecolor='xkcd:grey', zorder=0)
    
    # Alternative to contourf: plot the "real" raster using pcolormesh
    # filled_c = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
    #                          cmap='gist_ncar')

    filled_c = ax.contourf(lon, lat, data, transform = ccrs.PlateCarree(), 
                           levels = 100, cmap = cmap, alpha=alpha)#, vmin = 0, vmax=100)
    
    if type(seeds) == np.ndarray:
        # Get index of all seed locations and get their lat/lon coordinates
        y, x = np.where(seeds==1)
        y_lat = lat[y]
        x_lon = lon[x]
        # Plot each seed location
        for i in range(len(x_lon)):
            ax.plot(x_lon[i], y_lat[i], marker='.', c='r', markersize=2, 
                    transform=ccrs.PlateCarree())
            
    if labels == True:
        for i in np.unique(data[~np.isnan(data)]):
            y, x = np.where(data==i)
            
            # if domain crosses LON=0, assign the label to one 1° or -1°
            # (otherwise it will be somehwere on the other side of the earth)
            if 0 in x and 179 in x:
                x = int(np.round(np.mean(x)))
                if x < 90:
                    x = 0
                else:
                    x = 179
            else:
                x = int(np.round(np.mean(x)))
            y = int(np.round(np.mean(y)))
            
            if extent is not None:
                # plot label only if it's inside the extent of the plot
                if lon[x] > lon_min and lon[x] < lon_max and \
                   lat[y] > lat_min and lat[y] < lat_max:
                    ax.text(lon[x],lat[y], int(i-1), c='k', transform=ccrs.PlateCarree())

    # plot positions and their labels
    if pos_dict and draw_box==False:
        for i in range(len(pos_dict['lat'])):
            ax.plot(pos_dict['lon'][i], pos_dict['lat'][i], marker='+', 
                    color='k', markersize=12, markeredgewidth = 2,
                    transform=ccrs.Geodetic())
            
            ax.text(pos_dict['lon'][i], pos_dict['lat'][i]+3, 
                    "lat = {lat}\nlon = {lon}".format(lat=pos_dict['lat'][i],
                                                      lon=pos_dict['lon'][i]), 
                    verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'),
                    color='k', transform=ccrs.Geodetic())
    
    # Plot box
    if pos_dict and draw_box==True:
        if type(pos_dict)==list:
            # cols = cmocean.cm.haline(len(pos_dict))
            for i in range(len(pos_dict)):
                temp = pos_dict[i]
                # ax.fill(temp["lon"], temp["lat"], 
                #         color=cmocean.cm.haline(i/len(pos_dict)*256), 
                #         transform=ccrs.Geodetic(), alpha=0.8)
                ax.plot(temp["lon"], temp["lat"], marker='o', 
                        transform=ccrs.Geodetic())
                if len(pos_dict)>1:
                    region_label = "Region {}".format(i)
                else:
                    region_label = "Region"
                ax.text((temp['lon'][0]+temp['lon'][2])/2, 
                        (temp['lat'][0]+temp['lat'][1])/2, 
                        region_label,
                        verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'),
                    color='k', transform=ccrs.Geodetic())
        else:
            ax.fill(pos_dict["lon"], pos_dict["lat"], 
                    color=cmocean.cm.haline(128), 
                    transform=ccrs.Geodetic(), alpha=0.8)           
            ax.plot(pos_dict["lon"], pos_dict["lat"], marker='o', 
                        transform=ccrs.Geodetic())
            
            
    if show_grid==True:
        g1 = ax.gridlines(draw_labels=True)
        g1.top_labels = False
        g1.right_labels = False

        
    if show_colorbar==True:
        fig.colorbar(filled_c, orientation='horizontal')
    ax.set_title(title)

    if outpath==None and ax is None:
        #return ax
        plt.show()
    elif outpath is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(outpath + title + '.png', bbox_inches = 'tight')
        plt.close()    

#%% import netcdf

def importNetcdf(path,variable_name):
    """
    Imports a variable of a netCDF file as a masked array.

    Parameters
    ----------
    path : string
        Path to nc-file.
    variable_name : str
        Name of variable in nc-file.

    Returns
    -------
    field : masked array
        Imported data of the variable of the nc-file.
    """
    from netCDF4 import Dataset
    nc_fid = Dataset(path, 'r')
    var = [var for var in nc_fid.variables.keys()]

    if variable_name=='lon' and 'lon' is not var:
        longs=['lon','long','longitude','LON','x','longitudes']
        variable_name=list(set(var).intersection(longs))[0]

    if variable_name=='lat' and 'lat' is not var:
        lats=['lat','lati','latitude','LAT','y','latitudes']
        variable_name=list(set(var).intersection(lats))[0]
    
    if variable_name=='time' and 'time' is not var:
        times = ['time','TIME','t']
        variable_name=list(set(var).intersection(times))[0]

    if variable_name == 'time':
        from netCDF4 import num2date
        import numpy as np
        time_var = nc_fid.variables[variable_name]
        field = num2date(time_var[:],time_var.units, 
                         only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True).filled(np.nan).reshape(len(time_var),1)
    else:
        field = nc_fid.variables[variable_name][:]     
    return field 
        
#%% plot dMAPS output
def plot_dMaps_output(geofile, 
                      fpath, 
                      output = 'domain', 
                      outpath=None, 
                      title=None,
                      cmap=None,
                      show_seeds=False,
                      extent = None,
                      alpha=1.):
    """
    Function to plot the output of deltaMaps. By default, it plots a map of all
    domains, but it can also visualize the local homogeneity and the location 
    of the seeds as overlay. If no output path (outpath) is specified, the 
    plots will not be saved. If an output path is specified that does not 
    exist, it will be created by plot_map()-function.

    Parameters
    ----------
    geofile : string
        Path to the dataset (nc-file) that has been used for the clustering. 
        (required to get the lat/lon grid.)
    fpath : string
        Path to the directory where deltaMaps saved its results. Must contain
        the subdirectories "domain_identification" and "seed_identification".
    output : string, optional
        Desired extent of output (maps that will be produced). Can take the 
        following values:
            'all' -> plots local homogeneity map and domain map
            'domain' -> plots domain map only
            'homogeneity' -> plots homogeneity map only
        The default is 'domain'.
    outpath : string or None, optional
        Path to the directory where the plots will be stored. If an output path
        is specified that does not exist, it will be created by plot_map()-
        function. If None is given, the plots will not be saved. The default 
        is None.
    show_seeds : string or None, optional
        Specifies whether the seeds locations will be plotted onto the maps. 
        Can take the following values:
            False -> seeds locations will not be plotted
            True -> seeds locations will be plotted on all maps
            'homogeneity' -> seeds locations will be plotted only on the 
                             homogeneity map
        The default is False.
    extent : list, optional
        The extent of the map. The list must have the following structure: 
        [lon_min, lon_max, lat_min, lat_max]. If None is given, the entire 
        earth will be shown. The default is None.  
    alpha : float, optional
        Alpha (opacity) of the domains in the domain map. Default is 1.

    Returns
    -------
    None.

    Usage
    -------
    plot_dMaps_output(geofile = "data/AVISO_MSLA_1993-2020_prep_2_deg_gaus.nc",
                      fpath = "playground/output/res_2_k_5/", 
                      output = 'all', 
                      outpath = None,
                      show_seeds = 'homogeneity')

    """
    
    import numpy as np
    # import lat/lon vectors
    lon = importNetcdf(geofile,'lon')
    lat = importNetcdf(geofile,'lat')
    
    if show_seeds == False:
        seeds = None
    else:
        seeds = np.load(fpath + '/seed_identification/seed_positions.npy')  
    
    if output == 'all' or output == 'homogeneity':
        # Import homogeneity field
        homogeneity_field = np.load(fpath + 
                        '/seed_identification/local_homogeneity_field.npy')
        if not title:
            title = 'local homogeneity field'
            
        plot_map(lat = lat, 
                 lon = lon, 
                 data = homogeneity_field, 
                 seeds = seeds,
                 title = title,
                 cmap = 'viridis',
                 outpath = outpath,
                 extent = extent,
                 alpha = alpha)     
        
    if output == 'all' or output == 'domain':
        if show_seeds=='homogeneity':
            seeds = None
        # Import domain maps
        d_maps = np.load(fpath + '/domain_identification/domain_maps.npy')
        # Create array containing the number of each domain
        domain_map = get_domain_map(d_maps)
        if not title:
            title = 'Domain map'
        if not cmap:
            cmap='prism'
        plot_map(lat = lat, 
                  lon = lon, 
                  data = domain_map,
                  seeds = seeds,
                  title = title,
                  cmap = cmap,
                  outpath = outpath,
                  labels = True,
                  extent = extent,
                  alpha = alpha)             
        
    if output == 'all' or output == 'domain strength':
        seeds = None
        # Import domain maps
        strength_map = np.load(fpath + '/network_inference/strength_map.npy')
        strength_map[strength_map==0] = np.nan
        if not title:
            title = 'Strength map'
        plot_map(lat = lat, 
                 lon = lon, 
                 data = strength_map,
                 seeds = seeds,
                 title = title,
                 cmap = 'viridis',
                 outpath = outpath,
                 extent = extent,
                 alpha = alpha)          
 
def calc_nmi_matrix(path, res, k, 
                        path_end = '/domain_identification/domain_maps.npy'):

        """
        Calculates a matrix of Normalized Mutual Information between the results
        of deltaMaps for different Neighborhood-sizes (K) based on scikit-learns
        NMI-metric.

        Parameters
        ----------
        path : string
            Path to the directory with the different dMaps outputs for the
            different values of k.
        res : int
            Resolution that shall be assessed.
        k : range
            Range of k for which the output of dMaps is available unter the 
            specified filepath and for which the NMI matrix shall be created. 
        path_end : string, optional
            Path from root directory of dMaps run to the numpy file containing the
            domain maps. The default is '/domain_identification/domain_maps.npy'.

        Returns
        -------
        nmi_matrix : numpy array
            Array containing the NMI for different combinations of K-values.

        """
        import numpy as np   
        from sklearn.metrics.cluster import normalized_mutual_info_score
        nmi_matrix = np.zeros((len(k), len(k)))
        
        for row,row_k in enumerate(k):
            # pname = 'res_' + str(res) + '_k_' + str(row_k)
            pname = 'k' + str(row_k)
            row_d_vec = get_domain_vec(path+pname+path_end)
            
            for col, col_k in enumerate(k):
                # pname = 'res_' + str(res) + '_k_' + str(col_k)
                pname = 'k' + str(col_k)
                col_d_vec = get_domain_vec(path+pname+path_end)   
                
                nmi_matrix[row-1, col-1] = normalized_mutual_info_score(row_d_vec,
                                                                        col_d_vec)


        return nmi_matrix

def get_domain_map(d_maps):
    """
    Helper function that returns an array with the grid values for the 
    corresponding domain.

    Parameters
    ----------
    d_maps : np.array
        Three dimensional umpy array from 
        .../domain_identification/domain_maps.npy.

    Returns
    -------
    domain_map : np.array
        Two dimensional numpy array with the domain number as grid cell values.
        If no domain is present at a grid cell, a np.nan will be inserted.

    """
    import numpy as np
    # Create array containing the number of each domain
    domain_map = np.zeros((d_maps.shape[1], d_maps.shape[2]))
    i = 1
    for d in range(len(d_maps)):
        domain_map[d_maps[d] == 1] = i
        i += 1
    domain_map[domain_map==0] = np.nan
    return domain_map



def get_domain_vec(path_domain_maps):
    """
    Imports the deltaMaps domain map file and produces a numpy vector with the
    assignment of each grid cell to a domain. All grid cells which are in no 
    domain have the value 0. All overlaps between domains are assigned as new 
    domains (i.e. values > len(d_maps)+1).

    Parameters
    ----------
    path_domain_maps : string
        Path to domain maps numpy file (i.e. something like 
                                ".../domain_identification/domain_maps.npy".

    Returns
    -------
    domain_vec : 1D-numpy array of float64
        Numpy vector containing the assignment of each grid cell to a domain.
        All grid cells which are in no domain have the value 0. All overlaps 
        between domains are assigned as new domains.

    """
    import numpy as np
    # Import domain maps
    d_maps = np.load(path_domain_maps)
           
    domain_map = np.zeros((d_maps.shape[1], d_maps.shape[2]))
    i = 1
    k = len(d_maps)+2
    for d in range(len(d_maps)):
        # Account for possible overlaps between two domains: If a domain
        # is assigned to a grid cell which already is in a domain, assign
        # it to a new domain. Overlaps will start at len(d_maps)+1!
        domain_map[np.logical_and(d_maps[d] == 1, 
                                  domain_map != 0)] = k
        k += 1
        # If the grid cell is not assigned to a domain already, copy the
        # original domain number
        domain_map[np.logical_and(d_maps[d] == 1, 
                                          domain_map == 0)] = i
        i += 1
            
    domain_vec = np.concatenate(domain_map)
    return domain_vec

def plot_nmi_matrix(nmi_matrix, k, fname=None):
        """
        Produces a contourf-plot of the NMI-matrix and saves the output in a 
        specified filepath.

        Parameters
        ----------
        nmi_matrix : numpy array
            The NMI matrix from calc_nmi_matrix().
        k : range
            Range of the k-values to be plotted (i.e. x and y values of the NMI 
            matrix). Must be ascending (otherwise the image itself is flipped)!
        fname : string, optional
            Desired output filepath and name. If none is specified, the plot will 
            be shown and not saved. The default is None.

        Returns
        -------
        None.

        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(dpi=300)
        dat = ax.contourf(k, k, nmi_matrix, 
                          cmap='jet', 
                          levels = np.linspace(0.2, 1.0, 100))
        plt.colorbar(dat, 
                     ax=ax, 
                     extend='both', 
                     ticks=np.arange(0.2, 1.2, 0.2))
        ax.set_ylabel("K")
        ax.set_xlabel("K")
        ax.set_xlim([min(k),max(k)])
        ax.set_ylim([min(k),max(k)])
        ax.set_yticks(np.arange(min(k),max(k)+1,2))
        ax.set_xticks(np.arange(min(k),max(k)+1,2))
        ax.set_aspect('equal', 'box')   
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname, bbox_inches = 'tight')
            