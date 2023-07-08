
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import sys
import spectral.io.envi as envi
import time
import os
os.chdir('D:/_RESEARCH/_JARS2023/')
from sklearn import metrics
from spectral import ace

datasets        = ['gudalur', 'cookecity', 'avon', 'ritcampus']
#methods         = ['refHSI', 'CNMF', 'HYSURE', 'GDD']
methods         = ['SSRNET', 'ResTFNet', 'NNDIFFUSE_II']

#method_colors   = ['b', 'tab:orange', 'c', 'm', 'tab:brown', 'gray', 'r']

fig, axs = plt.subplots(len(methods), len(datasets))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

for m in range(len(methods)):
    method = methods[m]
    if method=='refHSI':       method_color='b'
    if method=='CNMF':         method_color='tab:orange'
    if method=='HYSURE':       method_color='c'
    if method=='GDD':          method_color='m'
    if method=='SSRNET':       method_color='tab:brown'
    if method=='ResTFNet':     method_color='seagreen'
    if method=='NNDIFFUSE_II': method_color='r'
    
    
    for d in range(len(datasets)):
        dataset = datasets[d]
        #Directory of the ACE map
        acemap_dir = 'ACE_maps/'+dataset+'/'
        
        #Directory of the target map
        targetmask_dir = 'truthmasks_targets/'+dataset+'/'
        
        if dataset=='gudalur':
            #Load ACE map
            target_name = 'green'
            path_acemap = acemap_dir + target_name + '_' + method
            acemap      = np.array(envi.open(path_acemap+'.hdr', path_acemap+'.img').load())
            acemap      = acemap.ravel()
            numpixels   = acemap.shape[0]
            
            #Load targetmask
            path_targetmask = targetmask_dir + 'GUDALURSTD_subset_wv3_targetmask'
            targetmask      = np.array(envi.open(path_targetmask+'.hdr', path_targetmask+'.img').load())
            targetmask      = targetmask.ravel()
            #The blue cotton-containing pixels are valued = 1
            targetmask[targetmask!=1] = 0
            
            #Plot histogram of current sharpened image
            axs[m, d].hist(acemap, 64, density=True, facecolor='k', alpha=0.50)
            
            for p in range(numpixels):
                if targetmask[p]!=0: axs[m, d].axvline(x=acemap[p], ymin=0, ymax=0.2, color=method_color, alpha=0.5)
                
            #axs[m, d].set_xlim((-0.01,0.8))
            axs[m, d].get_yaxis().set_visible(False)
            #axs[m, d].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
        
        if dataset=='cookecity':
            #Load ACE map
            target_name = 'blue_cotton'
            path_acemap = acemap_dir + target_name + '_' + method
            acemap      = np.array(envi.open(path_acemap+'.hdr', path_acemap+'.img').load())
            acemap      = acemap.ravel()
            numpixels   = acemap.shape[0]
            
            #Load targetmask
            path_targetmask = targetmask_dir + 'cookecity_multipixel_targetmask'
            targetmask      = np.array(envi.open(path_targetmask+'.hdr', path_targetmask+'.img').load())
            targetmask      = targetmask.ravel()
            #The blue cotton-containing pixels are valued = 3
            targetmask[targetmask!=3] = 0
            
            #Plot histogram of current sharpened image
            axs[m, d].hist(acemap, 64, density=True, facecolor='k', alpha=0.50)
            
            for p in range(numpixels):
                if targetmask[p]!=0: axs[m, d].axvline(x=acemap[p], ymin=0, ymax=0.2, color=method_color, alpha=0.5)
            #axs[m, d].set_xlim((-0.01,0.8))
            axs[m, d].get_yaxis().set_visible(False)
            #axs[m, d].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
        
        if dataset=='avon':
            #Load ACE map
            target_name = 'blue_felt'
            path_acemap = acemap_dir + target_name + '_' + method
            acemap      = np.array(envi.open(path_acemap+'.hdr', path_acemap+'.img').load())
            acemap      = acemap.ravel()
            numpixels   = acemap.shape[0]
            
            #Load targetmask
            path_targetmask = targetmask_dir + 'avon_multipixel_targetmask'
            targetmask      = np.array(envi.open(path_targetmask+'.hdr', path_targetmask+'.img').load())
            targetmask      = targetmask.ravel()
            #The blue target-containing pixels are valued = 2
            targetmask[targetmask!=2] = 0
            
            #Plot histogram of current sharpened image
            axs[m, d].hist(acemap, 64, density=True, facecolor='k', alpha=0.50)
            
            for p in range(numpixels):
                if targetmask[p]!=0: axs[m, d].axvline(x=acemap[p], ymin=0, ymax=0.2, color=method_color, alpha=0.5)
            #axs[m, d].set_xlim((-0.01,0.8))
            axs[m, d].get_yaxis().set_visible(False)
            #axs[m, d].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
        
        if dataset=='ritcampus':
            #Load ACE map
            target_name = 'red_felt'
            path_acemap = acemap_dir + target_name + '_' + method
            acemap      = np.array(envi.open(path_acemap+'.hdr', path_acemap+'.img').load())
            acemap      = acemap.ravel()
            numpixels   = acemap.shape[0]
            
            #Load targetmask
            path_targetmask = targetmask_dir + 'ritcampus_multipixel_targetmask'
            targetmask      = np.array(envi.open(path_targetmask+'.hdr', path_targetmask+'.img').load())
            targetmask      = targetmask.ravel()
            #The red target-containing pixels are valued = 1
            targetmask[targetmask!=1] = 0
            
            #Plot histogram of current sharpened image
            axs[m, d].hist(acemap, 64, density=True, facecolor='k', alpha=0.50)
            
            for p in range(numpixels):
                if targetmask[p]!=0: axs[m, d].axvline(x=acemap[p], ymin=0, ymax=0.2, color=method_color, alpha=0.5)
            #axs[m, d].set_xlim((-0.01,0.8))
            axs[m, d].get_yaxis().set_visible(False)
            #axs[m, d].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
            
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.show()
            