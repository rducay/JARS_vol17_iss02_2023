from torch import nn
from utils import *
import cv2
import pdb
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
import spectral.io.envi as envi
import numpy as np

def validate(test_list, arch, model, epoch, n_epochs, dataset, testimsize):
    test_ref, test_lr, test_hr = test_list
    model.eval()

    psnr = 0
    with torch.no_grad():
        # Set mini-batch dataset
        ref = to_var(test_ref).detach()
        lr = to_var(test_lr).detach()
        hr = to_var(test_hr).detach()
        if arch == 'SSRNet':
            out, _, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpat':
            _, out, _, _, _, _ = model(lr, hr)
        elif arch == 'SSRSpec':
            _, _, out, _, _, _ = model(lr, hr)
        else:
            out, _, _, _, _, _ = model(lr, hr)

        ref = ref.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

        rmse = calc_rmse(ref, out)
        psnr = calc_psnr(ref, out)
        ergas = calc_ergas(ref, out)
        sam = calc_sam(ref, out)
        
        #MY MODIFICATION
        if epoch==n_epochs-1:
            HRHSI_fused       = np.squeeze(out, axis=0)
            HRHSI_fused       = np.swapaxes(HRHSI_fused, 0, 1)
            HRHSI_fused       = np.swapaxes(HRHSI_fused, 1, 2)
            
            if dataset=='avon':      NRtest, NCtest, NBtest = 400, 320, 327
            if dataset=='chikusei':  NRtest, NCtest, NBtest = 400, 400, 128
            if dataset=='cookecity': NRtest, NCtest, NBtest = 280, 400, 126
            if dataset=='cupriteng': NRtest, NCtest, NBtest = 400, 400, 372
            if dataset=='gudalur':   NRtest, NCtest, NBtest = 400, 400, 370
            if dataset=='ritcampus': NRtest, NCtest, NBtest = 400, 320, 327
            
            #Padding size for the test image
            #padsize_row = int((testimsize-NRtest+8)/2)
            #padsize_col = int((testimsize-NCtest+8)/2)
            
            padsize_row = 4
            padsize_col = 4
            
            #Restore un-padded test image dimensions
            HRHSI_fused = HRHSI_fused[padsize_row : NRtest+padsize_row, padsize_col : NCtest+padsize_col, :]
            
            print('\n'+arch+' fused image: HRHSI_fused.shape: ', HRHSI_fused.shape)
            print(arch+' fused image: HRHSI_fused.mean(): ',     HRHSI_fused.mean())
            print(arch+' fused image: HRHSI_fused.max(): ',      HRHSI_fused.max())
            print(arch+' fused image: HRHSI_fused.min(): ',      HRHSI_fused.min())
            
            if arch=='SSRNET':
                path_fused_HRHSI = 'fused_images/SSRNET/'+dataset+'/'
                fnm_fused_HRHSI  = path_fused_HRHSI + dataset + '_SSRNET'
                envi.save_image(fnm_fused_HRHSI+'.hdr', 10000*HRHSI_fused/255, force=True)
            
            elif arch=='ResTFNet':
                path_fused_HRHSI = 'fused_images/ResTFNet/'+dataset+'/'
                fnm_fused_HRHSI  = path_fused_HRHSI + dataset + '_ResTFNet'
                envi.save_image(fnm_fused_HRHSI+'.hdr', 10000*HRHSI_fused/255, force=True)
                
            elif arch=='ConSSFCNN':
                path_fused_HRHSI = 'fused_images/ResTFNet/'+dataset+'/'
                fnm_fused_HRHSI  = path_fused_HRHSI + dataset + '_ConSSFCNN'
                envi.save_image(fnm_fused_HRHSI+'.hdr', 10000*HRHSI_fused/255, force=True)
                
        with open(dataset+'_'+arch+'.txt', 'a') as f:
            f.write(str(epoch) + ',' + str(rmse) + ',' + str(psnr) + ',' + str(ergas) + ',' + str(sam) + ',' + '\n')

    return psnr