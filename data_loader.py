import os
os.chdir('D:/_RESEARCH/_JARS2023/')
from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py
import spectral.io.envi as envi
import scripts.my_walds_protocol as wp

def pad_hsi(input_hsi, pad_size_row, pad_size_col):
    hsi_padded = np.pad(input_hsi, ((pad_size_row,pad_size_row), (pad_size_col,pad_size_col), (0,0)), mode='reflect')
    return(hsi_padded)

def build_datasets(root, dataset, size, n_select_bands, scale_ratio):
    #Image preprocessing, normalization for the pretrained resnet
    
    #Specify 'TRAIN' and 'TEST' paths
    #TRAIN paths
    train_path       = 'data/TRAIN/'+dataset+'/'
    train_refHSI_fnm = train_path+dataset+'_TRAIN_refHSI'
    train_HRMSI_fnm  = train_path+dataset+'_TRAIN_HRMSI'
    
    #TEST paths
    test_path        = 'data/TEST/'+dataset+'/'
    test_refHSI_fnm  = test_path+dataset+'_TEST_refHSI'
    test_HRMSI_fnm   = test_path+dataset+'_TEST_HRMSI'
    
    #Load 'TRAIN' images
    train_refHSI = np.array(envi.open(train_refHSI_fnm+'.hdr', train_refHSI_fnm+'.img').load())
    train_HRMSI  = np.array(envi.open(train_HRMSI_fnm+'.hdr', train_HRMSI_fnm+'.img').load())
    
    #Load 'TEST' images
    test_refHSI  = np.array(envi.open(test_refHSI_fnm+'.hdr', test_refHSI_fnm+'.img').load())
    test_HRMSI   = np.array(envi.open(test_HRMSI_fnm+'.hdr', test_HRMSI_fnm+'.img').load())
    
    #pad the high-res 'TEST' images
    NRtest, NCtest, NBtest    = test_refHSI.shape
    padded_test_refHSI        = pad_hsi(test_refHSI, 4, 4)
    padded_test_HRMSI         = pad_hsi(test_HRMSI,  4, 4)
    
    #--------------------------------------------------------------------------
    test_ref               = padded_test_refHSI.copy()
    test_lr                = wp.simLRHSI(test_ref,   'gaussian', GSDratio=4, kernelSize=6)
    test_hr                = np.copy(padded_test_HRMSI)
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    train_ref               = np.copy(train_refHSI)
    train_lr                = wp.simLRHSI(train_ref, 'gaussian', GSDratio=4, kernelSize=6)
    train_hr                = np.copy(train_HRMSI)
    #--------------------------------------------------------------------------
    
    #Transform to 8-bit: TRAIN
    if train_ref.mean()>=1: train_ref = 255*train_ref/10000
    if train_ref.mean()<1:  train_ref = 255*train_ref
    
    if train_hr.mean()>=1:  train_hr  = 255*train_hr/10000
    if train_hr.mean()<1:   train_hr  = 255*train_hr
    
    if train_lr.mean()>=1:  train_lr  = 255*train_lr/10000
    if train_lr.mean()<1:   train_lr  = 255*train_lr
    
    #Transform to 8-bit: TEST
    if test_ref.mean()>=1: test_ref   = 255*test_ref/10000
    if test_ref.mean()<1:  test_ref   = 255*test_ref
    
    if test_hr.mean()>=1:  test_hr    = 255*test_hr/10000
    if test_hr.mean()<1:   test_hr    = 255*test_hr
    
    if test_lr.mean()>=1:  test_lr    = 255*test_lr/10000
    if test_lr.mean()<1:   test_lr    = 255*test_lr
    
    print('\ntrain_ref.shape: ', train_ref.shape)
    print('train_lr.shape: ',    train_lr.shape)
    print('train_hr.shape: ',    train_hr.shape)
    
    print('\ntrain_ref.mean(): ', train_ref.mean())
    print('train_lr.mean(): ',    train_lr.mean())
    print('train_hr.mean(): ',    train_hr.mean())
    
    print('\ntest_ref.shape: ',  test_ref.shape)
    print('test_lr.shape: ',     test_lr.shape)
    print('test_hr.shape: ',     test_hr.shape)
    
    print('\ntest_ref.mean(): ', test_ref.mean())
    print('test_lr.mean(): ',    test_lr.mean())
    print('test_hr.mean(): ',    test_hr.mean())
    
    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr  = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0) 
    train_hr  = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0) 
    test_ref  = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0) 
    test_lr   = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0) 
    test_hr   = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0) 

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]