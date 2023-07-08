import os
os.chdir('D:/_RESEARCH/_JARS2023/')
import numpy as np
import spectral.io.envi as envi
from models.NNDIFFUSE.hrmsi_lrhsi_fusion import fusion_NNDIFFUSE_I, fusion_NNDIFFUSE_II
import time
script_time0 = time.time()

#------------------------------------------------------------------------------
#datasets = ['avon', 'chikusei', 'cookecity', 'cupriteng', 'gudalur', 'ritcampus']
datasets = ['avon']
#EDGE_SENSITIVITY_PCTILE = 80

#SRF info
srf_wv2_fnm        = 'srf/srf_wv2.csv'
srf_idxBands_HRMSI = (2,3,4,5,6,7,8,9)
srf_WLedges_HRMSI  = [(0.300,0.452), (0.452,0.510), (0.510,0.585), (0.585,0.628),\
                      (0.628,0.698), (0.698,0.755), (0.755,0.865), (0.865,1.100)]

for dataset in datasets:
    #Open refHSI and put in reflectance mode
    path_refHSI = 'data/TEST/'+dataset+'/'+dataset+'_TEST_refHSI'
    refHSI      = np.array(envi.open(path_refHSI+'.hdr', path_refHSI+'.img').load())
    if np.mean(refHSI) >= 1:
        refHSI  = refHSI/10000
    
    #Open reference wavelengths and keep in microns
    ref_lambdas = np.array([float(i) for i in envi.open(path_refHSI+'.hdr').metadata['wavelength']])
    if ref_lambdas.mean()>=10:
        ref_lambdas = ref_lambdas/1000
    
    #Open LRHSI and put in reflectance mode
    path_LRHSI  = 'data/TEST/'+dataset+'/'+dataset+'_TEST_LRHSI'
    LRHSI       = np.array(envi.open(path_LRHSI+'.hdr', path_LRHSI+'.img').load())
    if np.mean(LRHSI) >= 1:
        LRHSI   = LRHSI/10000
    
    #Open HRMSI and put in reflectance mode
    path_HRMSI  = 'data/TEST/'+dataset+'/'+dataset+'_TEST_HRMSI'
    HRMSI       = np.array(envi.open(path_HRMSI+'.hdr', path_HRMSI+'.img').load())
    if np.mean(HRMSI) >= 1:
        HRMSI   = HRMSI/10000
    
    print('\nData exploration: '+dataset+': refHSI:................................')
    print('refHSI.mean(): ', refHSI.mean())
    print('refHSI.max(): ',  refHSI.max())
    print('refHSI.min(): ',  refHSI.min())
    
    print('\nData exploration: '+dataset+': LRHSI:................................')
    print('LRHSI.mean(): ', LRHSI.mean())
    print('LRHSI.max(): ',  LRHSI.max())
    print('LRHSI.min(): ',  LRHSI.min())
    
    print('\nData exploration: '+dataset+': HRMSI:................................')
    print('HRMSI.mean(): ', HRMSI.mean())
    print('HRMSI.max(): ',  HRMSI.max())
    print('HRMSI.min(): ',  HRMSI.min())
    
    #Specify path for fused images
    path_fused_NNDIFFUSE_I  = 'fused_images/NNDIFFUSE_I/'+dataset+'/'+dataset+'_NNDIFFUSE_I'
    path_fused_NNDIFFUSE_II = 'fused_images/NNDIFFUSE_II/'+dataset+'/'+dataset+'_NNDIFFUSE_II'
    
    #Perform fusion
    fused_NNDIFFUSE_I  = fusion_NNDIFFUSE_I(  LRHSI, HRMSI, ref_lambdas, srf_WLedges_HRMSI)
    fused_NNDIFFUSE_II = fusion_NNDIFFUSE_II(LRHSI, HRMSI, ref_lambdas, srf_WLedges_HRMSI)

    print('\nData exploration: fused_NNDIFFUSE_I:................................')
    print('fused_NNDIFFUSE_I.mean(): ', fused_NNDIFFUSE_I.mean())
    print('fused_NNDIFFUSE_I.max(): ',  fused_NNDIFFUSE_I.max())
    print('fused_NNDIFFUSE_I.min(): ',  fused_NNDIFFUSE_I.min())
    
    print('\nData exploration: fused_NNDIFFUSE_II:................................')
    print('fused_NNDIFFUSE_II.mean(): ', fused_NNDIFFUSE_II.mean())
    print('fused_NNDIFFUSE_II.max(): ',  fused_NNDIFFUSE_II.max())
    print('fused_NNDIFFUSE_II.min(): ',  fused_NNDIFFUSE_II.min())
    
    #Save in reflx10000 format (0,10000)
    envi.save_image(path_fused_NNDIFFUSE_I+'.hdr',  10000*fused_NNDIFFUSE_I, force=True)
    envi.save_image(path_fused_NNDIFFUSE_II+'.hdr', 10000*fused_NNDIFFUSE_II, force=True)

script_time1 = time.time()
print('\nTotal script time: ', script_time1-script_time0)

