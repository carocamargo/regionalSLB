#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:12:40 2022

@author: ccamargo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
#%%
# make_figure(save=False)
path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'

def make_figure(save=True,
                path_to_figures = path_figures,
                figname = 'budget_stats',
                figfmt='png'):
    settings = set_settings()
    path = '/Volumes/LaCie_NIOZ/data/budget/'
    path = '/Users/ccamargo/Desktop/manuscript_SLB/data/'
    #path = '/Volumes/LaCie_NIOZ/data/budget/'
    df = pd.read_pickle(path+"budget-stats.p")
    df = df.drop(df[df.comb == '(alt,sum)'].index)
    df['budget_name'] = ['1 degree' if label=='1degree'  
                         else 'SOM' if label=='som'  
                         else '$\delta$-MAPS' 
                         for label in df['budget'] ]
    df['combo'] = [combo.replace('bar','GRD') 
               # if 'bar' in combo 
               # else combo
               for combo in df['comb']]
    # plot grouped boxplot
    varis = ['r','nRMSE']
    ncol=len(varis)
    nrow = 1
    fig = plt.figure(figsize=(10,10))
    for i,var in enumerate(varis):
        plt.subplot(ncol,nrow,i+1)
        sns.boxplot(x = df['combo'],
                    y = df[var],
                    hue = df['budget_name'],
                    showfliers = False,
                    palette = 'Set2')
        
        if var=='R2':
            plt.ylim(-2,2)
        if var =='r':
            plt.ylim(-1,1)
        if var=='nRMSE':
            plt.ylim(0,0.3)
        plt.ylabel(var,fontsize=20)
        plt.xlabel('')
        if i==0:
            plt.legend(loc='lower left')
            plt.text(6.25,-0.9,'({})'.format(settings['letters'][i]),
                     fontsize=settings['textsize'],fontweight="bold")
        else: 
            plt.legend(loc='upper left')
            plt.text(6.25,0.28,'({})'.format(settings['letters'][i]),
                     fontsize=settings['textsize'],fontweight="bold")   
        
    plt.tight_layout()
    plt.show()

    plt.show()
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return

def set_settings():
    global settings
    settings = {}
    
    # Global plotting settings
    settings['fontsize']=25
    settings['lon0']=201;
    settings['fsize']=(15,10)
    settings['proj']='robin'
    settings['land']=True
    settings['grid']=False
    settings['landcolor']='papayawhip'
    settings['extent'] = False
    settings['plot_type'] = 'contour'
    
    
    settings['textsize'] = 13
    settings['labelsize']=18    
    settings['colors_dic'] = {
        'Altimetry':'mediumseagreen',
        'Sum':'mediumpurple',
        'Steric':'goldenrod',
        'Dynamic':'indianred',
        'Barystatic':'steelblue',
        'Sterodynamic':'palevioletred',
        'Residual':'gray'
        }
    settings['acronym_dic'] = {
        'alt':'Altimetry',
        'sum':'Sum',
        'steric':'Steric',
        'res':'Residual',
        'dynamic':'Dynamic',
        'barystatic':'Barystatic'
        }
    settings['titles_dic'] = {
            'alt':r"$\eta_{obs}$",
             'steric': r"$\eta_{SSL}$", 
             'sum': r"$\sum(\eta_{SSL}+\eta_{GRD}+\eta_{DSL})$", 
              'barystatic':r"$\eta_{GRD}$",
             'res': r"$\eta_{obs} - \eta_{\sum(drivers)}$", 
              'dynamic':r"$\eta_{DSL}$",
                 }
    settings['letters'] = list(string.ascii_lowercase)
    
    return settings

