import numpy as np
import time

def calc_aceMap(input_hsi, ref_target_sig, bkg_stats, domain=10000, bkg_size=0):
    time_start = time.time()
    
    #Check if input HSI has the right shape
    if len(input_hsi.shape)!=3:
        raise Exception('Shape if HSI input should be (NR, NC, NB)')
    else:
        NR,NC,NB       = input_hsi.shape
        ref_target_sig = ref_target_sig.ravel().reshape((NB,1))
    
    #Specify domain
    if domain==10000:
        input_hsi      = 10000*input_hsi/input_hsi.max()
        ref_target_sig = 10000*ref_target_sig/ref_target_sig.max()
    elif domain==1:
        input_hsi      = input_hsi/input_hsi.max()
        ref_target_sig = ref_target_sig/ref_target_sig.max()
    else:
        raise Exception('Specify HSI values: reflectance (domain=1) or 100xrefl (domain=10000)')
    
    #Calculate inverse of covariance matrix
    Sinv      = np.linalg.inv(np.cov(input_hsi.reshape((NR*NC, NB)), rowvar=False))
    mu_global = np.mean(input_hsi.reshape((NR*NC, NB)), axis=0).reshape((NB, 1)) 
    
    #Construct ACE map
    acemap    = np.empty((NR, NC))
    pixcount  = 0
    if bkg_stats == 'local':
        roimask  = np.ones((int(2*bkg_size+1), int(2*bkg_size+1)))
        roimask[bkg_size, bkg_size] = 0
        roimask  = roimask.ravel()
        padded_input_hsi = np.pad(input_hsi, ((bkg_size,bkg_size), (bkg_size,bkg_size), (0,0)), mode='reflect')
        for i in range(NR):
            for j in range(NC):
                
                #Monitor progress
                pixcount += 1
                print('ACE calculations progress: ', pixcount/(NR*NC))
                
                roi         = padded_input_hsi[i:i+(2*bkg_size+1), j:j+(2*bkg_size+1), :]
                nr,nc,_     = roi.shape
                roi         = roi.reshape((nr*nc, NB))
                roi         = roi[roimask!=0]
                #Calculate local mean
                mu_local    = (np.mean(roi, axis=0)).reshape((-1,1))
                
                #Calculate ACE score
                x           = input_hsi[i,j,:].reshape((NB, 1)) - mu_local
                T           = ref_target_sig - mu_local
                numer       = float(np.dot(T.T, np.dot(Sinv, x)))**2
                denom1      = float(np.dot(T.T, np.dot(Sinv, T)))
                denom2      = float(np.dot(x.T, np.dot(Sinv, x)))
                acemap[i,j] = numer/(denom1*denom2)
                del(mu_local)
        return(acemap)
    
    if bkg_stats == 'global':
        for i in range(NR):
            for j in range(NC):
                
                #Monitor progress
                pixcount += 1
                print('ACE calculations progress: ', pixcount/(NR*NC))
                
                x           = input_hsi[i,j,:].reshape((NB, 1)) - mu_global
                T           = ref_target_sig - mu_global
                numer       = float(np.dot(T.T, np.dot(Sinv, x)))**2
                denom1      = float(np.dot(T.T, np.dot(Sinv, T)))
                denom2      = float(np.dot(x.T, np.dot(Sinv, x)))
                acemap[i,j] = numer/(denom1*denom2)
        return(acemap)        
    
    time_stop = time.time()
    print('Total acemap calculation time: ', time_stop-time_start)            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                