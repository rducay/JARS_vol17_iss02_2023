import numpy as np
import math

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)
    return psnr

def calc_ergas(img_tgt, img_fus):
    if img_tgt.shape!=img_fus.shape:
        raise Exception('Reference and fused images do NOT match!')
    else:
        img_tgt = np.squeeze(img_tgt)
        img_fus = np.squeeze(img_fus)
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
        img_fus = img_fus.reshape(img_fus.shape[0], -1)
        rmse = np.mean((img_tgt-img_fus)**2, axis=1)
        rmse = rmse**0.5
        mean = np.mean(img_tgt, axis=1)
        ergas = np.mean((rmse/mean)**2)
        ergas = 100/4*ergas**0.5
        return(ergas)
    
def calc_rmse(img_tgt, img_fus):
    if img_tgt.shape!=img_fus.shape:
        raise Exception('Reference and fused images do NOT match!')
    else:
        return(np.sqrt(np.mean((img_tgt-img_fus)**2)))

def calc_sam(truth_img, test_img, output_SAMmap=False):
    #Given a ground truth image and fused image, generate a spectral angle map
    #Inputs:
    #truth_img (ground truth image): (NR, NC, NB)
    #test_img (fused image):         (NR, NC, NB)
    #Returns
    #thetamap: a spectral angle map over the entire image: (NR, NC)
    NR, NC, NB   = truth_img.shape
    thetamap     = np.empty((NR, NC))
    pixcount     = 0
    #Check if dimensions match
    if truth_img.shape!=test_img.shape:
        raise Exception('Test and truth images have to be of equal dimensions.')
    else:
        for i in range(NR):
            for j in range(NC):
                pixcount += 1
                #print('SAM calculation progress (%): ', 100*pixcount/(512*512))
                vec_truth      = truth_img[i,j].reshape((-1,1))
                vec_test       = test_img[i,j].reshape((-1,1))
                vec_truth_norm = np.sqrt(np.sum(vec_truth**2))
                vec_test_norm  = np.sqrt(np.sum(vec_test**2))
                thetamap[i,j]  = np.arccos(np.dot(vec_test.T, vec_truth)/(vec_truth_norm*vec_test_norm))
    #print('Image-wide SAM index ('+testImgName+'): ', np.mean(thetamap))
    if output_SAMmap==False:
        return(np.mean(thetamap))
    elif output_SAMmap==True:
        return(thetamap)

def calc_crosscorr(truth_img, test_img):
    if truth_img.shape!=test_img.shape:
        raise Exception('Test and truth images have to be of equal shape.')
    else:
        truth_img  = np.reshape(truth_img, (truth_img.shape[0]*truth_img.shape[1], -1))
        test_img   = np.reshape(test_img,  (test_img.shape[0]*test_img.shape[1],   -1))
        spatialcos = []
        for bandnum in range(truth_img.shape[1]):
            vecTruth    = np.copy(truth_img[:,bandnum])
            vecTruth    = vecTruth - np.mean(vecTruth)
            vecTruth    = vecTruth.reshape((-1,1))
            
            vecTest     = np.copy(test_img[:, bandnum])
            vecTest     = vecTest - np.mean(vecTest)
            vecTest     = vecTest.reshape((-1,1))
            
            dotprod     = float(np.dot(vecTruth.T, vecTest))
            vecTruthMag = np.sqrt(float(np.dot(vecTruth.T, vecTruth)))
            vecTestMag  = np.sqrt(float(np.dot(vecTest.T, vecTest)))
            
            if (vecTruthMag==0) or (vecTestMag==0):
                costheta = 0
                spatialcos.append(costheta)
            if (vecTruthMag!=0) and (vecTestMag!=0):
                costheta    = dotprod/(vecTruthMag*vecTestMag)
                spatialcos.append(costheta)
        return(np.mean(spatialcos))

def spectral_angle_mapper(truth_img, test_img, testImgName='Test image'):
    #Given a ground truth image and fused image, generate a spectral angle map
    #Inputs:
    #GTI (ground truth image): (NR, NC, NB)
    #HRMSI (fused image):      (NR, NC, NB)
    NR, NC, NB   = truth_img.shape
    thetamap     = np.empty((NR, NC))
    pixcount     = 0
    #Check if dimensions match
    if truth_img.shape!=test_img.shape:
        raise Exception('Test and truth images have to be of equal dimensions.')
    else:
        for i in range(NR):
            for j in range(NC):
                pixcount += 1
                #print('SAM calculation progress (%): ', 100*pixcount/(512*512))
                vec_truth     = truth_img[i,j].reshape((-1,1))
                vec_test      = test_img[i,j].reshape((-1,1))
                thetamap[i,j] = np.arccos(np.dot(vec_test.T, vec_truth)/(np.linalg.norm(vec_test)*np.linalg.norm(vec_truth)))
    print('Image-wide SAM index ('+testImgName+'): ', np.mean(thetamap))
    return(thetamap)
    