import os
os.chdir('D:/_RESEARCH/_JARS2023/')
import numpy as np
import spectral.io.envi as envi
import sys
import scripts.my_walds_protocol as wp
from models.CNMF.CNMF import *
import matplotlib.pyplot as plt
import time
script_time0 = time.time()

#------------------------------------------------------------------------------
datasets = ['avon', 'chikusei', 'cookecity', 'cupriteng', 'gudalur', 'ritcampus']

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
    
    print('\nData exploration: refHSI: '+dataset+'...........................')
    print('refHSI.mean(): ', refHSI.mean())
    print('refHSI.max(): ',  refHSI.max())
    print('refHSI.min(): ',  refHSI.min())
    
    print('\nData exploration: LRHSI: '+dataset+'............................')
    print('LRHSI.mean(): ', LRHSI.mean())
    print('LRHSI.max(): ',  LRHSI.max())
    print('LRHSI.min(): ',  LRHSI.min())
    
    print('\nData exploration: HRMSI: '+dataset+'............................')
    print('HRMSI.mean(): ', HRMSI.mean())
    print('HRMSI.max(): ',  HRMSI.max())
    print('HRMSI.min(): ',  HRMSI.min())
    
    #Specify path for fused images
    path_fused_CNMF = 'fused_images/CNMF/'+dataset+'/'+dataset+'_CNMF_ver'
    
    #Perform CNMF fusion 5 times
    for vernum in range(5):
        fused_CNMF = CNMF_fusion(LRHSI,HRMSI,verbose='on')
        envi.save_image(path_fused_CNMF+str(vernum)+'.hdr', 10000*fused_CNMF, force=True)
    
    print('\nData exploration: fused_CNMF: '+dataset+'.......................')
    print('fused_CNMF.mean(): ', fused_CNMF.mean())
    print('fused_CNMF.max(): ',  fused_CNMF.max())
    print('fused_CNMF.min(): ',  fused_CNMF.min())

script_time1 = time.time()
print('\nTotal script time: ', script_time1-script_time0)

