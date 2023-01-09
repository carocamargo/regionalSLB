#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:10:34 2022

@author: ccamargo
"""
path_figures = '/Users/ccamargo/Desktop/manuscript_SLB/overleaf/figures/'

def plot_stats(save=True,
                path_to_figures = path_figures,
                figname = 'stats_SI',
                figfmt='png'
                ):
    #% %
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import string
    letters = list(string.ascii_lowercase)
    path = '/Volumes/LaCie_NIOZ/data/budget/'
    df =  pd.read_pickle(path+"budget-stats_complete.p")
    # plot grouped boxplot
    # df = df.sort_values(by=['budget_name'])
    varis = ['r','nRMSE']
    ncol=len(varis)
    nrow = 1
    textsize = 13
    fig = plt.figure(figsize=(10,10))
    for i,var in enumerate(varis):
        plt.subplot(ncol,nrow,i+1)
        sns.boxplot(x = df['comb'],
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
            plt.text(6.25,-0.9,'({})'.format(letters[i]),fontsize=textsize,fontweight="bold")
        else: 
            plt.legend(loc='upper left')
            plt.text(6.25,0.28,'({})'.format(letters[i]),fontsize=textsize,fontweight="bold")   
        
    plt.tight_layout()
    plt.show()
    
    if save:
        kurs=path_to_figures+figname+'.'+figfmt
            
        fig.savefig(kurs,format=figfmt,dpi=300,bbox_inches='tight')

    return


