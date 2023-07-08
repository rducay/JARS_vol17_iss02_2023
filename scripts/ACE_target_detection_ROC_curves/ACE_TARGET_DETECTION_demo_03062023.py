import numpy as np
import matplotlib.pyplot as plt
import sys
import spectral.io.envi as envi
import time
import os
os.chdir('D:/_RESEARCH/_JARS2023/')
from sklearn import metrics
from spectral import ace
from scripts.my_ACE_calculator import calc_aceMap
#Specify dataset
dataset = 'avon'

#Specify target
target  = 'redfelt'

#Specify fusion algorithm
path_refHSI        = 'data/TEST/'+dataset+'/'+dataset+'_TEST_refHSI'
path_NNDIFFUSE_II  = 'fused_images/NNDIFFUSE_II/'+dataset+'/'+dataset+'_NNDIFFUSE_II'
path_CNMF          = 'fused_images/CNMF/'+dataset+'/'+dataset+'_CNMF_ver0'


refHSI             = np.array(envi.open(path_refHSI+'.hdr', path_refHSI+'.img').load())
ref_lambdas        = np.array([float(i) for i in envi.open(path_refHSI+'.hdr').metadata['wavelength']])
fused_NNDIFFUSE_II = np.array(envi.open(path_NNDIFFUSE_II+'.hdr', path_NNDIFFUSE_II+'.img').load())
fused_CNMF         = np.array(envi.open(path_CNMF+'.hdr', path_CNMF+'.img').load())

NR, NC, NB         = refHSI.shape
NPIX               = NR*NC

if dataset == 'avon':
    #tgt_T1_red  = np.loadtxt('targetspectra/redfelt_extracted_from_avonAM_0920_1701_located_on_uniform_field.txt')
    #tgt_T2_blue = np.loadtxt('targetspectra/bluefelt_extracted_from_avonAM_0920_1701_located_on_uniform_field.txt')
    tgt_T1_red  = np.loadtxt('targetspectra/redfelt_from_share2012_asdmeasurements.txt')
    tgt_T2_blue = np.loadtxt('targetspectra/bluefelt_from_share2012_asdmeasurements.txt')
tgt_T1_red  = np.interp(ref_lambdas, tgt_T1_red[:,0], tgt_T1_red[:, 1])/10000
tgt_T2_blue = np.interp(ref_lambdas, tgt_T2_blue[:,0], tgt_T2_blue[:, 1])/10000

plt.figure()
plt.scatter(ref_lambdas, tgt_T1_red,  label='red felt',  c='r')
plt.scatter(ref_lambdas, tgt_T2_blue, label='blue felt', c='b')
plt.legend()
plt.show()

#Target masks and ROC curves
tgtmask_fnm  = 'truthmasks_targets/'+dataset+'/'+dataset+'_multipixel_targetmask'
tgtmask      = envi.open(tgtmask_fnm+'.hdr', tgtmask_fnm+'.img').load()
tgtmask      = (np.array(tgtmask)).ravel()

tgtmask_red  = np.copy(tgtmask)
tgtmask_blue = np.copy(tgtmask)

tgtmask_red[tgtmask_red!=1]   = 0
tgtmask_blue[tgtmask_blue!=2] = 0

tgtmask_red  = tgtmask_red.astype(int)
tgtmask_blue = tgtmask_blue.astype(int)

del(tgtmask)

#Plot ROC curve for one target: red felt
#ACE CALCULATION METHOD: LOCAL
if target=='redfelt':
    tgtmask = np.copy(tgtmask_red)
    pos_label = 1
    acemap_refHSI              = calc_aceMap(refHSI,             tgt_T1_red, domain=10000, bkg_stats='local', bkg_size=11)
    acemap_NNDIFFUSE_II_local  = calc_aceMap(fused_NNDIFFUSE_II, tgt_T1_red, domain=10000, bkg_stats='local', bkg_size=11)
    acemap_NNDIFFUSE_II_global = calc_aceMap(fused_NNDIFFUSE_II, tgt_T1_red, domain=10000, bkg_stats='global')
    acemap_CNMF_local          = calc_aceMap(fused_CNMF,         tgt_T1_red, domain=10000, bkg_stats='local', bkg_size=11)
    
    plt.figure()
    plt.title(dataset+': red felt: local vs global ACE calculation method')
    
    fpr_refHSI,              tpr_refHSI,              _ = metrics.roc_curve(tgtmask, acemap_refHSI.ravel(), pos_label=1)
    fpr_NNDIFFUSE_II_local,  tpr_NNDIFFUSE_II_local,  _ = metrics.roc_curve(tgtmask, acemap_NNDIFFUSE_II_local.ravel(), pos_label=1)
    fpr_NNDIFFUSE_II_global, tpr_NNDIFFUSE_II_global, _ = metrics.roc_curve(tgtmask, acemap_NNDIFFUSE_II_global.ravel(), pos_label=1)
    fpr_CNMF_local,          tpr_CNMF_local,          _ = metrics.roc_curve(tgtmask, acemap_CNMF_local.ravel(), pos_label=1)
    
    plt.plot(fpr_refHSI,              tpr_refHSI,              label='refHSI',           color='k')
    plt.plot(fpr_NNDIFFUSE_II_local,  tpr_NNDIFFUSE_II_local,  label='NNDiffuse-local',  color='r')
    plt.plot(fpr_NNDIFFUSE_II_global, tpr_NNDIFFUSE_II_global, label='NNDiffuse-global', color='b')
    plt.plot(fpr_CNMF_local,          tpr_CNMF_local,          label='CNMF-local',       color='g')
    
    plt.legend()
    plt.show()
    
