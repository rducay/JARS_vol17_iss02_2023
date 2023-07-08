import numpy as np
import os
os.chdir('D:/_RESEARCH/_JARS2023/')
from models.NNDIFFUSE import pansharpen as ps
from scripts.my_SAMbased_edgeDetector import edgeDetector_SA
import scripts.my_walds_protocol as wp
# 'major' superpixel: the superpixel surrounding but NOT containing the pixel of interest (x,y)
# 'minor' superpixel: the superpixel containing the pixel of interest (x,y)

def fusion_NNDIFFUSE_I(LRHSI, HRMSI, lambdas_LRHSI, wlength_bounds_HRMSI):
    if np.mean(LRHSI)>=1: raise Exception('LRHSI input has to be in reflectance units (0,1)')
    if np.mean(HRMSI)>=1: raise Exception('HRMSI input has to be in reflectance units (0,1)')
    #Data cube dimensions
    NR, NC, NB_HRMSI   = HRMSI.shape
    nr, nc, nb         = LRHSI.shape
    npix               = nr*nc
    
    #Construct LRHSI_bandcorrMap for band-band correlations: use the LRHSI
    LRHSI_bandcorrMap = np.copy(LRHSI)
    
    #Working on 'inside bands'
    inside_lrhsi_bands_idxs = []
    lrhsi_hrmsi_idx_pairs   = []
    for m in range(NB_HRMSI):
        WL0_m, WL1_m = wlength_bounds_HRMSI[m]
        for n in range(nb):
            if (lambdas_LRHSI[n]>=WL0_m) and (lambdas_LRHSI[n]<WL1_m):
                lrhsi_hrmsi_idx_pairs.append((n, m))
                inside_lrhsi_bands_idxs.append(n)
    lrhsi_hrmsi_idx_pairs = np.array(lrhsi_hrmsi_idx_pairs, dtype=object)
    
    #Working on 'outside bands'
    for i in range(nb):
        if i in inside_lrhsi_bands_idxs:
            continue
        else:
            coefvals = []
            for j in inside_lrhsi_bands_idxs:
                vec_outside_edgemap = LRHSI_bandcorrMap[:, :, i].reshape((nr*nc, -1)).T
                vec_inside_edgemap  = LRHSI_bandcorrMap[:, :, j].reshape((nr*nc, -1)).T
                coefmatrix          = np.corrcoef(vec_outside_edgemap, vec_inside_edgemap) #shape: (2,2)
                coefvals.append(np.absolute(coefmatrix[0,1])) #Treat negative correlations as positive
                
            #Choose inside band with highest correlation, then add to index pairs
            best_inside_lrhsi_band_idx = inside_lrhsi_bands_idxs[np.argmax(coefvals)]
            hrmsi_idx = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 0]==best_inside_lrhsi_band_idx][:, 1]
            newpair   = np.array([i, hrmsi_idx], dtype=object).reshape((1,-1))
            lrhsi_hrmsi_idx_pairs = np.concatenate((lrhsi_hrmsi_idx_pairs, newpair), axis=0)
    
    #Sort by HRMSI band idx number
    lrhsi_hrmsi_idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1].argsort()]
    
    HRHSI_HRHSI_NNDIFFUSE_I = np.zeros((NR, NC, 1))
    for i in range(NB_HRMSI):
        idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1]==i]
        if len(idx_pairs)==0:
            continue
        else:
            print('\nNNDIFF_I: Fusing group '+str(i)+'.....')
            msi      = np.copy(LRHSI[:,:,tuple(idx_pairs[:, 0])])
            PAN      = np.expand_dims(np.copy(HRMSI[:, :, i]), axis=2)
            fused_grp = ps.pansharpen_NND0(PAN, msi)
            HRHSI_HRHSI_NNDIFFUSE_I = np.concatenate((HRHSI_HRHSI_NNDIFFUSE_I, fused_grp), axis=2)
            
    HRHSI_HRHSI_NNDIFFUSE_I = HRHSI_HRHSI_NNDIFFUSE_I[:,:,1:]
    
    #Re-sort by lrhsi_hrmsi_idx_pairs by HRMSI band idx number
    correct_idx_order     = np.arange(len(lrhsi_hrmsi_idx_pairs)).reshape((-1,1))
    lrhsi_hrmsi_idx_pairs = np.concatenate((correct_idx_order, lrhsi_hrmsi_idx_pairs), axis=1)
    lrhsi_hrmsi_idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1].argsort()]
    correct_idx_order     = tuple(lrhsi_hrmsi_idx_pairs[:, 0])
    
    HRHSI_HRHSI_NNDIFFUSE_I     = HRHSI_HRHSI_NNDIFFUSE_I[:, :, correct_idx_order]
    
    #Return fused image in REFLECTANCE DOMAIN (0,1)
    return(HRHSI_HRHSI_NNDIFFUSE_I)

def fusion_NNDIFFUSE_II(LRHSI, HRMSI, lambdas_LRHSI, wlength_bounds_HRMSI):
    if np.mean(LRHSI)>=1: raise Exception('LRHSI input has to be in reflectance units (0,1)')
    if np.mean(HRMSI)>=1: raise Exception('HRMSI input has to be in reflectance units (0,1)')
    #Data cube dimensions
    NR, NC, NB_HRMSI   = HRMSI.shape
    nr, nc, nb         = LRHSI.shape
    npix               = nr*nc
    
    #Construct LRHSI_edgeMap using ScharrEdge detector (or Laplacian, etc.)
    #This band-by-band edgeMaps will be used for band correlations
    LRHSI_edgeMap = np.zeros((nr, nc, 1))
    for b in range(nb):
        edgemap          = np.expand_dims(wp.ScharrEdge(LRHSI[:,:,b]), axis=2)
        #edgemap          = np.expand_dims(wp.LaplacianEdge(LRHSI[:,:,b]), axis=2)
        LRHSI_edgeMap    = np.concatenate((LRHSI_edgeMap, edgemap), axis=2)
    LRHSI_edgeMap        = LRHSI_edgeMap[:,:,1:]

    #Working on 'inside bands'
    inside_lrhsi_bands_idxs = []
    lrhsi_hrmsi_idx_pairs   = []
    for m in range(NB_HRMSI):
        WL0_m, WL1_m = wlength_bounds_HRMSI[m]
        for n in range(nb):
            if (lambdas_LRHSI[n]>=WL0_m) and (lambdas_LRHSI[n]<WL1_m):
                lrhsi_hrmsi_idx_pairs.append((n, m))
                inside_lrhsi_bands_idxs.append(n)
    lrhsi_hrmsi_idx_pairs = np.array(lrhsi_hrmsi_idx_pairs, dtype=object)
    
    #Working on 'outside bands'
    for i in range(nb):
        if i in inside_lrhsi_bands_idxs:
            continue
        else:
            coefvals = []
            for j in inside_lrhsi_bands_idxs:
                vec_outside_edgemap = LRHSI_edgeMap[:, :, i].reshape((nr*nc, -1)).T
                vec_inside_edgemap  = LRHSI_edgeMap[:, :, j].reshape((nr*nc, -1)).T
                coefmatrix          = np.corrcoef(vec_outside_edgemap, vec_inside_edgemap) #shape: (2,2)
                coefvals.append(np.absolute(coefmatrix[0,1])) #Treat negative correlations as positive
                
            #Choose inside band with highest correlation, then add to index pairs
            best_inside_lrhsi_band_idx = inside_lrhsi_bands_idxs[np.argmax(coefvals)]
            hrmsi_idx = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 0]==best_inside_lrhsi_band_idx][:, 1]
            newpair   = np.array([i, hrmsi_idx], dtype=object).reshape((1,-1))
            lrhsi_hrmsi_idx_pairs = np.concatenate((lrhsi_hrmsi_idx_pairs, newpair), axis=0)
    
    #Sort by HRMSI band idx number
    lrhsi_hrmsi_idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1].argsort()]
    
    HRHSI_NNDIFFUSE_II = np.zeros((NR, NC, 1))
    for i in range(NB_HRMSI):
        idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1]==i]
        if len(idx_pairs)==0:
            continue
        else:
            print('\nNNDIFF_II: Fusing group '+str(i)+'.....')
            msi      = np.copy(LRHSI[:,:,tuple(idx_pairs[:, 0])])
            PAN      = np.expand_dims(np.copy(HRMSI[:, :, i]), axis=2)
            fused_grp = ps.pansharpen_NND0(PAN, msi)
            HRHSI_NNDIFFUSE_II = np.concatenate((HRHSI_NNDIFFUSE_II, fused_grp), axis=2)
            
    HRHSI_NNDIFFUSE_II = HRHSI_NNDIFFUSE_II[:,:,1:]
    
    #Re-sort by lrhsi_hrmsi_idx_pairs by HRMSI band idx number
    correct_idx_order     = np.arange(len(lrhsi_hrmsi_idx_pairs)).reshape((-1,1))
    lrhsi_hrmsi_idx_pairs = np.concatenate((correct_idx_order, lrhsi_hrmsi_idx_pairs), axis=1)
    lrhsi_hrmsi_idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1].argsort()]
    correct_idx_order     = tuple(lrhsi_hrmsi_idx_pairs[:, 0])
    
    HRHSI_NNDIFFUSE_II     = HRHSI_NNDIFFUSE_II[:, :, correct_idx_order]
    
    #Return fused image in REFLECTANCE DOMAIN (0,1)
    return(HRHSI_NNDIFFUSE_II)







