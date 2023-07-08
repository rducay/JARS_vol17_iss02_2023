import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
os.chdir('D:/_RESEARCH/_JARS2023/')
from scripts.my_fusion_metrics import calc_ergas, calc_sam, calc_psnr, calc_rmse, calc_crosscorr

#datasets      = ['avon', 'chikusei', 'cookecity', 'cupriteng', 'gudalur', 'ritcampus']
datasets      = ['cupriteng']
hrmsi_systems = ['wv2']

for hrmsi_system in hrmsi_systems:
    for d in range(len(datasets)):
        dataset     = datasets[d]
        path_refHSI = 'data/TEST/'+dataset+'/'+dataset+'_TEST_refHSI'
        refHSI      = np.array(envi.open(path_refHSI+'.hdr', path_refHSI+'.img').load())
        if dataset in ['chikusei', 'gudalur']:
            refHSI  = 10000*refHSI
        print('\nrefHSI.mean(): ', refHSI.mean())
        
        method      = 'CNMF'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method+'_ver0'
        fused_CNMF  = 10000*np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        #ver0: 
        #ver1: 1.4649
        print('fused_CNMF.mean(): ', fused_CNMF.mean())
        
        method      = 'NNDIFFUSE_I'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_NNDIFFUSE_I = np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        print('fused_NNDIFFUSE_I.mean(): ', fused_NNDIFFUSE_I.mean())
                
        method      = 'NNDIFFUSE_II'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_NNDIFFUSE_II = np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        print('fused_NNDIFFUSE_II.mean(): ', fused_NNDIFFUSE_II.mean())
        
        method      = 'HYSURE'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_HYSURE= np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        print('fused_HYSURE.mean(): ', fused_HYSURE.mean())
                
        method      = 'GDD'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_GDD   = np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        print('fused_GDD.mean(): ', fused_GDD.mean())
        
        method      = 'SSRNET'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_SSRNET= np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        print('fused_SSRNET.mean(): ', fused_SSRNET.mean())
                
        method      = 'ResTFNet'
        path_method = 'fused_images/'+method+'/'+dataset+'/'+dataset+'_'+method
        fused_ResTFNet= np.array(envi.open(path_method+'.hdr', path_method+'.img').load())
        print('fused_ResTFNet.mean(): ', fused_ResTFNet.mean())
        
        #Calculate spectral angle map
        sam_NNDIFFUSE_I      = calc_sam(truth_img=refHSI, test_img=fused_NNDIFFUSE_I, output_SAMmap=True)
        path_sam_NNDIFFUSE_I = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_NNDIFFUSE_I'
        envi.save_image(path_sam_NNDIFFUSE_I+'.hdr', sam_NNDIFFUSE_I, force=True)
        
        sam_NNDIFFUSE_II      = calc_sam(truth_img=refHSI, test_img=fused_NNDIFFUSE_II, output_SAMmap=True)
        path_sam_NNDIFFUSE_II = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_NNDIFFUSE_II'
        envi.save_image(path_sam_NNDIFFUSE_II+'.hdr', sam_NNDIFFUSE_II, force=True)
                
        sam_CNMF      = calc_sam(truth_img=refHSI, test_img=fused_CNMF, output_SAMmap=True)
        path_sam_CNMF = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_CNMF'
        envi.save_image(path_sam_CNMF+'.hdr', sam_CNMF, force=True)
                
        sam_HYSURE      = calc_sam(truth_img=refHSI, test_img=fused_HYSURE, output_SAMmap=True)
        path_sam_HYSURE = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_HYSURE'
        envi.save_image(path_sam_HYSURE+'.hdr', sam_HYSURE, force=True)
        
        sam_GDD      = calc_sam(truth_img=refHSI, test_img=fused_GDD, output_SAMmap=True)
        path_sam_GDD = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_GDD'
        envi.save_image(path_sam_GDD+'.hdr', sam_GDD, force=True)
        
        sam_SSRNET      = calc_sam(truth_img=refHSI, test_img=fused_SSRNET, output_SAMmap=True)
        path_sam_SSRNET = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_SSRNET'
        envi.save_image(path_sam_SSRNET+'.hdr', sam_SSRNET, force=True)
                
        sam_ResTFNet      = calc_sam(truth_img=refHSI, test_img=fused_ResTFNet, output_SAMmap=True)
        path_sam_ResTFNet = 'spectral_angle_maps/'+dataset+'/sam_'+dataset+'_ResTFNet'
        envi.save_image(path_sam_ResTFNet+'.hdr', sam_ResTFNet, force=True)
        
        
        
        
        