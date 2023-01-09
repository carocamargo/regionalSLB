#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:14:58 2022

@author: ccamargo
"""
path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
# open
def open_dataset():
    import xarray as xr
    path = '/Volumes/LaCie_NIOZ/data/budget/'
    path = path_to_data
    fname = 'budget_components'    
    da = xr.open_dataset(path+fname+'.nc')
    
    return da

# regrid
def regrid():
    import os
    path = '/Volumes/LaCie_NIOZ/data/budget/'
    path=path_to_data
    file = 'budget_components'

    for res in [2,5]:
        res_lat = 180/res 
        res_lon = 360/res
        
        os.system(
            "cdo -L remapbil,r{}x{} ".format(int(res_lon),int(res_lat))
            + str(path + file+'.nc')
            + " "
            + str(path + file+"_{}deg.nc".format(res))      
            )
        
regrid()
