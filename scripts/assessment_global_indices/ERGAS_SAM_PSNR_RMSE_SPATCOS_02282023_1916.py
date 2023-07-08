import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
os.chdir('D:/_RESEARCH/_JARS2023/')
from scripts.my_fusion_metrics import calc_ergas, calc_sam, calc_psnr, calc_rmse, calc_crosscorr

#datasets      = ['avon', 'chikusei', 'cookecity', 'cupriteng', 'gudalur', 'ritcampus']
datasets      = ['cupriteng']
methods       = ['NNDIFFUSE_I', 'NNDIFFUSE_II', 'CNMF', 'HYSURE', 'GDD', 'SSRNET', 'ResTFNet']
hrmsi_systems = ['wv2']

for hrmsi_system in hrmsi_systems:
    #print('\n'+'-------------------'+hrmsi_system+' results:'+'-------------------')
    for d in range(len(datasets)):
        dataset     = datasets[d]
        path_refHSI = 'data/TEST/'+dataset+'/'+dataset+'_TEST_refHSI'
        refHSI      = np.array(envi.open(path_refHSI+'.hdr', path_refHSI+'.img').load())
        if dataset in ['chikusei', 'gudalur']:
            refHSI  = 10000*refHSI
        
        method      = 'CNMF'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method+'_ver0'
        fused_CNMF  = 10000*np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        #ver0: 
        #ver1: 1.4649
        
        
        method      = 'NNDIFFUSE_I'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_NNDIFFUSE_I = np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        
        method      = 'NNDIFFUSE_II'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_NNDIFFUSE_II = np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        
        
        method      = 'HYSURE'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_HYSURE= np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        
        method      = 'GDD'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_GDD   = np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        
        method      = 'SSRNET'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_SSRNET= np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        
        method      = 'ResTFNet'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_ResTFNet= np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
                
# =============================================================================
#         #ERGAS
# =============================================================================
        ergas_NND_I = np.around(calc_ergas(refHSI, fused_NNDIFFUSE_I),  5)
        ergas_NND_II= np.around(calc_ergas(refHSI, fused_NNDIFFUSE_II), 5)
        ergas_CNMF  = np.around(calc_ergas(refHSI, fused_CNMF),         5)
        ergas_HYSURE= np.around(calc_ergas(refHSI, fused_HYSURE),       5)
        ergas_GDD   = np.around(calc_ergas(refHSI, fused_GDD),          5)
        ergas_SSRNET= np.around(calc_ergas(refHSI, fused_SSRNET),       5)
        ergas_ResTFNet= np.around(calc_ergas(refHSI, fused_ResTFNet),   5)
        print('\n'+dataset+' ERGAS results...................................')
        print('NNDIFFUSE_I: ', ergas_NND_I)
        print('NNDIFFUSE_II: ',ergas_NND_II)
        print('CNMF: ',        ergas_CNMF)
        print('HYSURE: ',      ergas_HYSURE)
        print('GDD: ',         ergas_GDD)
        print('SSRNET: ',      ergas_SSRNET)
        print('ResTFNet: ',ergas_ResTFNet)
        
# =============================================================================
#         #SAM
# =============================================================================
        sam_NND_I = np.around(calc_sam(refHSI, fused_NNDIFFUSE_I), 5)
        sam_NND_II= np.around(calc_sam(refHSI, fused_NNDIFFUSE_II),5)
        sam_CNMF  = np.around(calc_sam(refHSI, fused_CNMF),        5)
        sam_HYSURE= np.around(calc_sam(refHSI, fused_HYSURE),      5)
        sam_GDD   = np.around(calc_sam(refHSI, fused_GDD),         5)
        sam_SSRNET= np.around(calc_sam(refHSI, fused_SSRNET),      5)
        sam_ResTFNet = np.around(calc_sam(refHSI, fused_ResTFNet), 5)
            
        print('\n'+dataset+' SAM results...................................')
        print('NNDIFFUSE_I: ', sam_NND_I)
        print('NNDIFFUSE_II: ',sam_NND_II)
        print('CNMF: ',        sam_CNMF)
        print('HYSURE: ',      sam_HYSURE)
        print('GDD: ',         sam_GDD)
        print('SSRNET: ',      sam_SSRNET)
        print('ResTFNet: ',sam_ResTFNet)
        
# =============================================================================
#         #PSNR
# =============================================================================
        psnr_NND_I = np.around(calc_psnr(refHSI, fused_NNDIFFUSE_I), 5)
        psnr_NND_II= np.around(calc_psnr(refHSI, fused_NNDIFFUSE_II),5)
        psnr_CNMF  = np.around(calc_psnr(refHSI, fused_CNMF),        5)
        psnr_HYSURE= np.around(calc_psnr(refHSI, fused_HYSURE),      5)
        psnr_GDD   = np.around(calc_psnr(refHSI, fused_GDD),         5)
        psnr_SSRNET= np.around(calc_psnr(refHSI, fused_SSRNET),      5)
        psnr_ResTFNet=np.around(calc_psnr(refHSI, fused_ResTFNet),   5)
            
        print('\n'+dataset+' PSNR results...................................')
        print('NNDIFFUSE_I: ', psnr_NND_I)
        print('NNDIFFUSE_II: ',psnr_NND_II)
        print('CNMF: ',        psnr_CNMF)
        print('HYSURE: ',      psnr_HYSURE)
        print('GDD: ',         psnr_GDD)
        print('SSRNET: ',      psnr_SSRNET)
        print('ResTFNet: ', psnr_ResTFNet)
            
# =============================================================================
#         #RMSE
# =============================================================================
        rmse_NND_I = np.around(calc_rmse(refHSI, fused_NNDIFFUSE_I), 5)
        rmse_NND_II= np.around(calc_rmse(refHSI, fused_NNDIFFUSE_II),5)
        rmse_CNMF  = np.around(calc_rmse(refHSI, fused_CNMF),        5)
        rmse_HYSURE= np.around(calc_rmse(refHSI, fused_HYSURE),      5)
        rmse_GDD   = np.around(calc_rmse(refHSI, fused_GDD),         5)
        rmse_SSRNET= np.around(calc_rmse(refHSI, fused_SSRNET),      5)
        rmse_ResTFNet = np.around(calc_rmse(refHSI, fused_ResTFNet), 5)
            
        print('\n'+dataset+' RMSE results...................................')
        print('NNDIFFUSE_I: ', rmse_NND_I)
        print('NNDIFFUSE_II: ',rmse_NND_II)
        print('CNMF: ',        rmse_CNMF)
        print('HYSURE: ',      rmse_HYSURE)
        print('GDD: ',         rmse_GDD)
        print('SSRNET: ',      rmse_SSRNET)
        print('ResTFNet: ', rmse_ResTFNet)
        
# =============================================================================
#         #CC (Cross-correlation)
# =============================================================================
        crosscorr_NND_I = np.around(calc_crosscorr(refHSI, fused_NNDIFFUSE_I), 5)
        crosscorr_NND_II= np.around(calc_crosscorr(refHSI, fused_NNDIFFUSE_II),5)
        crosscorr_CNMF  = np.around(calc_crosscorr(refHSI, fused_CNMF),        5)
        crosscorr_HYSURE= np.around(calc_crosscorr(refHSI, fused_HYSURE),      5)
        crosscorr_GDD   = np.around(calc_crosscorr(refHSI, fused_GDD),         5)
        crosscorr_SSRNET= np.around(calc_crosscorr(refHSI, fused_SSRNET),      5)
        crosscorr_ResTFNet = np.around(calc_crosscorr(refHSI, fused_ResTFNet), 5)
            
        print('\n'+dataset+' CROSS_CORR results...................................')
        print('NNDIFFUSE_I: ', crosscorr_NND_I)
        print('NNDIFFUSE_II: ',crosscorr_NND_II)
        print('CNMF: ',        crosscorr_CNMF)
        print('HYSURE: ',      crosscorr_HYSURE)
        print('GDD: ',         crosscorr_GDD)
        print('SSRNET: ',      crosscorr_SSRNET)
        print('ResTFNet: ', crosscorr_ResTFNet)
            
        
        
        
        
        