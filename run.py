#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:15:03 2022

@author: ccamargo
"""

import os
root = '/Users/ccamargo/Documents/github/SLB/'

def run(lst):
    for pwd in lst:
        if '/' in pwd:
            if len(pwd.split('/'))>2:
                s = pwd.split('/')
                script = s[len(pwd.split('/'))-1]
                path = pwd.split(script)[0]
                path = path[0:-2]
            else:
                path,script = pwd.split('/')
        else:
            path = ''
            script = pwd
        fmt = script.split('.')[-1]
        print('Running {}'.format(script))     
        if fmt =='py':
            os.system('python {}/{}'.format(root+path,script))
        else:
            os.system('jupyter nbconvert --to notebook --execute {}/{}'.format(path,script))

    
lst = [
       # 'components/select_dataset_rmse.py',
       # 'components/correct_alt_GIA.py',
        # 'budget_analysis/budget_combinations.py',
        # 'components/SLB-components_dict.py',
        # 'components/SLB-components_nc.py',
        # 'components/regrid.py',
        # 'budget_analysis/budget.py',
        # 'budget_analysis/SLB_stats.py',
        # 'budget_analysis/budget_2x2_5x5.py',
        # 'manuscript_figures/script_per_figures/all.py',
        'organize_data.py',
        'regions/prep-nc-dmaps_som.py', # for  interactive maps
        # # manuscript_figures/geoviews-interactive_map.ipynb        
           ]
run(lst)

'''
1) Get budget components to folders:
    data/budget/trends/
    data/budget/ts/

1b) find dataset with closest RMSE to ENS:
    components/select_dataset_rmse.py
        input data: time series at '/Volumes/LaCie_NIOZ/data/budget/ts/'
        output:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = "best_rmse.pkl"

1c) Correct altimetry trend with GIA:
    components/correct_alt_GIA.py
        input data: trends at /Volumes/LaCie_NIOZ/data/budget/trends/'
        output:

2) make budget combinations
    budget_analysis/budget_combinations.py
        input data: trends at '/Volumes/LaCie_NIOZ/data/budget/trends/'
        output:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = 'combinations.nc'

3) Make dictionary with components:
    components/SLB-components_dict.py
        input data:
        - trends at '/Volumes/LaCie_NIOZ/data/budget/trends/'
        - time series at '/Volumes/LaCie_NIOZ/data/budget/ts/'
        path = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = budget_components.pkl

4) make nc with components"
    components/SLB-components_nc.py
        input data:
        - SLB dictionary: budget_components.pkl at '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/' (step 3)
        output:
        path = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = budget_components.nc

5) regrid SLB components to 2x2 and 5x5:
    components/regrid.py
        input data:
        - SLB netcdf: budget_components.nc at '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/' (step 4)
        output:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = 'budget_components_2deg.nc'
        file = 'budget_components_5deg.nc'

6) make budget dic:
    budget_analysis/budget.py
        input data:
        - trends at '/Volumes/LaCie_NIOZ/data/budget/trends/'
        - time series at '/Volumes/LaCie_NIOZ/data/budget/ts/'
        output:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = 'budget.pkl'

7) make budget stats
    budget_analysis/SLB_stats.py
        input data:
        - SLB dictionary (step6): budget.plk at '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = 'budget-stats.p'
        file = 'budget-stats.pkl'

8) Run stats with 2x2 and 5x5:
    budget_analysis/budget_2x2_5x5.py
        input:
        - budget stats (step 7)
        - regrided trends (step 5)
        output:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        file = 'budget-stats_complete.p'

9) Re-run figures:
    manuscript_figures/script_per_figures/all.py
    -input data:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        budget.pkl
        budget-stats_complete.p
    path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/figures/'

10) Organize data to publish:
    organize_data.py
    -input data:
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
        budget.pkl
    -output:
    path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/pub/'
    - masks.nc
    - SOM_trends.xlsx
    - SOM_trends.pkl
    - dmaps_trends.xlsx
    - dmaps_trends.pkl
    - budget_components_ENS.nc

11) Prepare data for interactive maps
    regions/prep-nc-dmaps_som.py
        path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/pub/'
        Input: - dmaps_trends.pkl
            - som_trends.pkl

path_to_data = '/Users/ccamargo/Desktop/manuscript_SLB/data/revisions/'
path = path_to_data+'pub/'
# path_save = path_to_data+'pub/'
file = 'budget.pkl'
dic = pd.read_pickle(path_to_data+file)
df = pd.read_pickle(path+'dmaps_trends.pkl')

12) Make interactive maps
    manuscript_figures/geoviews-interactive_map.ipynb
          

'''
