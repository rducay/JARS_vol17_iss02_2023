import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
os.chdir('D:/_RESEARCH/_JARS2023/')

#datasets      = ['AVONPM', 'CHIKUSEI', 'COOKECITY', 'CUPRITENG', 'GUDALURSTD', 'RITCAMPUS']
datasets      = ['AVONPM']
methods       = ['CNMF', 'GDD', 'HYSURE', 'NNDIFFUSE_I', 'NNDIFFUSE_II']
hrmsi_systems = ['wv3']

for hrmsi_system in hrmsi_systems:
    print('\n'+'-------------------'+hrmsi_system+' results:'+'-------------------')
    for d in range(len(datasets)):
        dataset        = datasets[d]
        os.chdir(portDir+'NNDIFFUSE_FUSION_subset/'+dataset+'/')
        refHSI_fnm     = '_refHSI/'+dataset+'_subset_refHSI'
        refHSI         = np.array(envi.open(refHSI_fnm+'.hdr', refHSI_fnm+'.img').load())
        nrows,ncols,_  = refHSI.shape
        P, B           = refHSI.shape[0]*refHSI.shape[1], refHSI.shape[2]
        refHSI         = refHSI.reshape((P, B))
        
        #Constrain within (0, 10000)
        if np.mean(refHSI) < 1:  refHSI = refHSI*10000
        #These two steps below were imposed when HRMSI and LRHSI were simulated
        if refHSI.max() > 10000: refHSI[refHSI>10000] = 10000
        if refHSI.min() < 0:     refHSI[refHSI<0] = 0
        
        for m in range(len(methods)):
            method    = methods[m]
            HRHSI_fnm = '_'+method+'/'+dataset+'_subset_'+method+'_'+hrmsi_system
            HRHSI     = envi.open(HRHSI_fnm+'.hdr', HRHSI_fnm+'.img').load()
            HRHSI     = HRHSI.reshape((P, B))
            
            #Looping over bands (spatial quality metrics)
            ERGAS     = []
            PSNR      = []
            SPATCOS   = []
            for b in range(B):
                vecRef = np.copy(refHSI[:,b])
                vecEst = np.copy(HRHSI[:,b])
                mse    = np.mean((vecEst-vecRef)**2)
                maxI   = (refHSI[:,b]).max()
                #RMSE_b = np.sqrt(mse) #MODIFY THIS LINE; CHECK LITERATURE (YOKOYA, SSRNET)
                Mean_b = np.mean(vecRef)
                ERGAS.append((RMSE_b/Mean_b)**2)
                PSNR.append(10*np.log10((maxI**2)/mse))
                
                vecRef    = refHSI[:,b] - np.mean(refHSI[:,b])
                vecEst    = HRHSI[:,b]  - np.mean(HRHSI[:,b])
                vecRefMag = np.sqrt(np.sum(vecRef**2))
                vecEstMag = np.sqrt(np.sum(vecEst**2))
                SPATCOS.append(np.sum(vecRef*vecEst)/(vecRefMag*vecEstMag))
            
            #Looping over pixels (spectral quality metrics)
            SAM      = []
            RMSE      = []
            for p in range(P):
                vecRef = refHSI[p,:]
                vecEst = HRHSI[p,:]
                vecRefMag = np.sqrt(np.sum(vecRef**2))
                vecEstMag = np.sqrt(np.sum(vecEst**2))
                SAM.append(np.arccos(np.sum(vecRef*vecEst)/(vecRefMag*vecEstMag)))
                #SAM.append(np.sum(vecRef*vecEst)/(vecRefMag*vecEstMag))
                RMSE.append(np.sqrt(np.mean((vecEst-vecRef)**2)))
            
            #print('ERGAS.size: ', len(ERGAS))
            #print('SAM.size: ', len(SAM))
            #print('PSNR.size: ', len(PSNR))
            #print('RMSE.size: ', len(RMSE))
            #print('SPATCOS.size: ', len(SPATCOS))
            
            #Save spatial maps of SAM and RMSE
            qualMap_SAM     = np.array(SAM).reshape((nrows,ncols))
            qualMap_SAM_fnm = 'SAM/SAM_'+method+'_'+dataset+'_'+hrmsi_system
            envi.save_image(qualMap_SAM_fnm+'.hdr', qualMap_SAM, force=True)
            
            qualMap_RMSE     = np.array(RMSE).reshape((nrows,ncols))
            qualMap_RMSE_fnm = 'RMSE/RMSE_'+method+'_'+dataset+'_'+hrmsi_system
            envi.save_image(qualMap_RMSE_fnm+'.hdr', qualMap_RMSE, force=True)
            
            #Calculate global metric
            ERGAS   = format(np.round(100*0.25*np.sqrt(np.mean(ERGAS)), 5), '.5f')
            SAM     = format(np.round(np.mean(SAM), 5), '.5f')
            PSNR    = format(np.round(np.mean(PSNR), 5), '.5f')
            RMSE    = format(np.round(np.mean(RMSE), 5), '.5f')
            SPATCOS = format(np.round(np.mean(SPATCOS), 5), '.5f')
            #format(value, '.6f')
            #print(dataset+'-'+method+' [ERGAS|SAM|PSNR|RMSE|SPATCOS]: [ '+str(ERGAS)+' | '+str(SAM)+' | '+str(PSNR)+' | '+str(RMSE)+' | '+str(SPATCOS)+' ]')
            print(dataset+'-'+method+' [ERGAS|SAM|PSNR|RMSE|SPATCOS]: [ '+ERGAS+' | '+SAM+' | '+PSNR+' | '+RMSE+' | '+SPATCOS+' ]')
            
            