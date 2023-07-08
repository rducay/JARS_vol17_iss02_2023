import os
os.chdir('D:/_RESEARCH/_JARS2023/')
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import scripts.my_walds_protocol as wp

datasets = ['avon','chikusei','cookecity','cupriteng','gudalur','ritcampus']

#Synthesize LR and HR images
def simulate_LRHSI_HRMSI(path_refHSI, path_ref_lambdas, path_LRHSI, path_HRMSI, path_srf):
    #Open reference image and put in reflectance space (0,1)
    refHSI   = np.array(envi.open(path_refHSI+'.hdr', path_refHSI+'.img').load())
    if refHSI.mean()>=1:
        refHSI = refHSI/10000
    print('\nrefHSI.shape: ', refHSI.shape)
    print('refHSI.max(): ', refHSI.max())
    print('refHSI.mean(): ', refHSI.mean())
    print('refHSI.min(): ', refHSI.min())
    NR,NC,NB = refHSI.shape
    NPIX     = NR*NC
    
    #Open reference wavelengths and put in microns
    ref_lambdas = np.array([float(i) for i in envi.open(path_ref_lambdas+'.hdr').metadata['wavelength']])
    if ref_lambdas.mean()>10:
        ref_lambdas = ref_lambdas/1000
    print('\nref_lambdas.shape: ', ref_lambdas.shape)
    print('ref_lambdas.max(): ', ref_lambdas.max())
    print('ref_lambdas.min(): ', ref_lambdas.min())

    #Simulate LRHSI
    LRHSI    = wp.simLRHSI(np.copy(refHSI), 'gaussian', 4)
    nr,nc,nb = LRHSI.shape
    npix     = nr*nc
    print('\nLRHSI.shape: ', LRHSI.shape)
    print('LRHSI.max(): ', LRHSI.max())
    print('LRHSI.mean(): ', LRHSI.mean())
    print('LRHSI.min(): ', LRHSI.min())
    envi.save_image(path_LRHSI+'.hdr', 10000*LRHSI, force=True)
    
    #Simulate HRMSI via band integration with a given HRMSI srf curve (WorldView-2)
    refimg      = np.copy(np.reshape(refHSI, (NPIX, NB)))
    srf_wv2     = np.loadtxt(path_srf, delimiter=',')
    srfdatacols = srf_wv2.shape[1]

    #Construct R matrix (reflectance curve to band-integrate the reference)
    plt.figure()
    R               = np.copy(ref_lambdas).reshape((-1,1))
    for col in range(2, srfdatacols):
        srfcurve = np.interp(R[:,0], srf_wv2[:,0], srf_wv2[:,col]).reshape((-1,1))
        if srfcurve.sum() != 0: srfcurve = srfcurve/srfcurve.sum()
        R = np.concatenate((R, srfcurve), axis=1)
        plt.plot(ref_lambdas, srfcurve)
    R = R[:,1:].T #shape: (8,num_of_lrhsi_bands)
    
    #Construct HRMSI
    HRMSI = np.dot(refimg, R.T)
    HRMSI = np.reshape(HRMSI, (NR, NC, HRMSI.shape[1]))
    envi.save_image(path_HRMSI+'.hdr', 10000*HRMSI, force=True)
    print('\nHRMSI.shape: ', HRMSI.shape)
    print('HRMSI.max(): ', HRMSI.max())
    print('HRMSI.mean(): ', HRMSI.mean())
    print('HRMSI.min(): ', HRMSI.min())
    return([refHSI, LRHSI, HRMSI, ref_lambdas])

#TEST images
for dataset in datasets:
    path_refHSI = 'data/TEST/'+dataset+'/'+dataset+'_TEST_refHSI'
    path_LRHSI  = 'data/TEST/'+dataset+'/'+dataset+'_TEST_LRHSI'
    path_HRMSI  = 'data/TEST/'+dataset+'/'+dataset+'_TEST_HRMSI'
    path_srf    = 'srf/srf_wv2.csv'
    path_refWL  = path_refHSI
    
    _, LRHSI, HRMSI, _ = simulate_LRHSI_HRMSI(path_refHSI, path_refWL, path_LRHSI, path_HRMSI, path_srf)

#TRAIN images
for dataset in datasets:
    path_refHSI = 'data/TRAIN/'+dataset+'/'+dataset+'_TRAIN_refHSI'
    path_LRHSI  = 'data/TRAIN/'+dataset+'/'+dataset+'_TRAIN_LRHSI'
    path_HRMSI  = 'data/TRAIN/'+dataset+'/'+dataset+'_TRAIN_HRMSI'
    path_srf    = 'srf/srf_wv2.csv'
    path_refWL  = path_refHSI
    
    _, LRHSI, HRMSI, _ = simulate_LRHSI_HRMSI(path_refHSI, path_refWL, path_LRHSI, path_HRMSI, path_srf)
