import numpy as np
#import sys
#sys.path.insert(1, 'D:/_RESEARCH/_functions/')
import time
# 'major' superpixel: the superpixel surrounding but NOT containing the pixel of interest (x,y)
# 'minor' superpixel: the superpixel containing the pixel of interest (x,y)

def calc_Tvector(PAN, msi):
    '''
    This function calculates the photometric vector (whose components denote
       spectral contributions from the MSI bands to the PAN)
    The inputs should have the following array shapes
    PAN: (NR, NC, 1)
    msi: (nr, nc, nb)
    '''
    nr, nc, nb, npx = msi.shape[0], msi.shape[1], msi.shape[2], msi.shape[0]*msi.shape[1]
    NR              = PAN.shape[0]
    gsd_ratio       = NR/nr
    #The GSD ratio between high and low resolution images should be integral
    if gsd_ratio-np.round(gsd_ratio)!=0:
        raise Exception('GSD ratio should be an integer!')
    else:
        gsd_ratio=int(gsd_ratio)
    
    #Construct a downgraded PAN version
    pan_lr = np.zeros((nr, nc, 1))
    for ii in range(gsd_ratio):
        for jj in range(gsd_ratio):
            pan_lr += np.copy(PAN[ii::gsd_ratio, jj::gsd_ratio, :])
    pan_lr = (pan_lr/(4*4)).ravel()
    
    #Calculate Moore-Penrose pseudoinverse
    X = np.empty((npx, nb))
    for b in range(nb):
        X[:, b] = np.copy(msi[:, :, b]).ravel()
    y = np.copy(pan_lr).reshape((-1, 1))
    w = np.matmul(np.linalg.pinv(X), y) #equivalent to above but in one line of code
    return(w.ravel())

def calc_spectral_angle(vec1, vec2, mode=None):
    vec1 = vec1.ravel().reshape((-1, 1))
    vec2 = vec2.ravel().reshape((-1, 1))
    vec1mag = float(np.sqrt(np.dot(vec1.T, vec1)))
    vec2mag = float(np.sqrt(np.dot(vec2.T, vec2)))
    if vec1mag==0 or vec2mag==0:
        return(0)
    else:
        if mode=='cosine':
            return(float(np.dot(vec1.T, vec2)/(vec1mag*vec2mag)))
        elif mode=='radians':
            costheta = np.dot(vec1.T, vec2)/(vec1mag*vec2mag)
            if costheta>1:
                costheta=1
            elif costheta<-1:
                costheta=-1
            return(np.arccos(costheta))
        else:
            raise Exception('Specify mode as "cosine" or "radians"')

def pansharpen_NND0(PAN, msi, sigma_spat=2.5, sigma_int=1):
    sigma2_int_floor = 0.1
    
    #Image dimensions
    #Hi-res: UPPERCASE; low-res: lowercase
    NR, NC, NB, NPX = PAN.shape[0], PAN.shape[1], PAN.shape[2], PAN.shape[0]*PAN.shape[1]
    nr, nc, nb, npx = msi.shape[0], msi.shape[1], msi.shape[2], msi.shape[0]*msi.shape[1]
    gsd             = NR/nr #gsd ratio 
    
    #Check if PAN/msi GSD ratio makes sense
    if gsd-np.round(gsd) != 0:
        raise Exception('PAN and MSI inputs need to have integer GSD ratio')
    elif gsd<=1:
        raise Exception('PAN/MSI GSD ratio needs to be greater than 1')
    elif gsd==0:
        raise Exception('GSD ratio cannot be zero')
    else:
        gsd = int(gsd)
    
    #Downsample the PAN and generate T vector
    Tvector = calc_Tvector(PAN=PAN, msi=msi)
    
    #Loop through each pixel on the high-res PAN
    loop_timer0 = time.time()
    HRMSI       = np.empty((NR, NC, nb))
    for y in range(NR):
        for x in range(NC):
            mody, modx = y%gsd, x%gsd
            
            #Prevent zero division
            if PAN[y,x,:]==0: Pyx = PAN[y,x,:]+1e-4
            if PAN[y,x,:]!=0: Pyx = PAN[y,x,:]
            
            #REGION 5: Current superpixel (pixel spectrum: M5)
            N5      = np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx:x-modx+gsd, 0]))
            M5      = msi[int(y//gsd), int(x//gsd)]
            yuv,xuv = 0.5*(y-mody+y-mody+gsd), 0.5*(x-modx+x-modx+gsd)
            yuv,xuv = y-mody+0.5*gsd, x-modx+0.5*gsd
            c_spat5 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT5     = np.sum(M5*Tvector)
            
            #REGION 1
            if y in range(4) or x in range(4):
                N1      = np.sum(np.absolute(Pyx - PAN[y-mody:y, x-modx:x, 0]))
                M1      = np.zeros((nb,))
                c_spat1 = 1
            else:
                N1      = np.sum(np.absolute(Pyx-PAN[y-mody-gsd:y-mody, x-modx-gsd:x-modx, 0])) + np.sum(np.absolute(Pyx-PAN[y-mody:y, x-modx:x, 0]))
                M1      = msi[int(y//gsd)-1, int(x//gsd)-1]
                yuv,xuv = 0.5*(y-mody-gsd+y-mody), 0.5*(x-modx-gsd+x-modx)
                yuv,xuv = y-mody-0.5*gsd, x-modx-0.5*gsd
                c_spat1 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT1 = np.sum(M1*Tvector)
            
            
            #REGION 2
            if x in range(4):
                N2      = np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx:x+1, 0]))
                M2      = np.zeros((nb,))
                c_spat2 = 1
            else:
                N2      = np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx-gsd:x-modx, 0])) + np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx:x+1, 0]))
                M2      = msi[int(y//gsd), int(x//gsd)-1]
                yuv,xuv = y-mody+0.5*gsd, x-modx-0.5*gsd
                c_spat2 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT2 = np.sum(M2*Tvector)
            
            
            #REGION 3
            if y in range(NR-4,NR) or x in range(4):
                N3      = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x-modx:x+1, 0]))
                M3      = np.zeros((nb,))
                c_spat3 = 1
            else:
                N3      = np.sum(np.absolute(Pyx-PAN[y-mody+gsd:y-mody+(2*gsd), x-modx-gsd:x-modx, 0])) + np.sum(np.absolute(Pyx-PAN[y:y-mody+gsd, x-modx:x+1, 0]))
                M3      = msi[int(y//gsd)+1, int(x//gsd)-1]
                yuv,xuv = y-mody+1.5*gsd, x-modx-0.5*gsd
                c_spat3 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT3 = np.sum(M3*Tvector)
            
            
            #REGION 4
            if y in range(4):
                N4      = np.sum(np.absolute(Pyx - PAN[y-mody:y+1, x-modx:x-modx+gsd, 0]))
                M4      = np.zeros((nb,))
                c_spat4 = 1
                
            else:
                N4      = np.sum(np.absolute(Pyx-PAN[y-mody-gsd:y-mody, x-modx:x-modx+gsd, 0])) + np.sum(np.absolute(Pyx-PAN[y-mody:y+1, x-modx:x-modx+gsd, 0]))
                M4      = msi[int(y//gsd)-1, int(x//gsd)]
                yuv,xuv = y-mody-0.5*gsd, x-modx+0.5*gsd
                c_spat4   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT4 = np.sum(M4*Tvector)
            
            #REGION 6
            if y in range(NR-4, NR):
                N6      = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x-modx:x-modx+gsd, 0]))
                M6      = np.zeros((nb,))
                c_spat6 = 1
            else:
                N6      = np.sum(np.absolute(Pyx-PAN[y-mody+gsd:y-mody+(2*gsd), x-modx:x-modx+gsd, 0])) + np.sum(np.absolute(Pyx-PAN[y:y-mody+gsd, x-modx:x-modx+gsd, 0]))
                M6      = msi[int(y//gsd)+1, int(x//gsd)]
                yuv,xuv = y-mody+1.5*gsd, x-modx+0.5*gsd
                c_spat6 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT6 = np.sum(M6*Tvector)
            
            
            #REGION 7
            if y in range(4) or x in range(NC-4, NC):
                N7      = np.sum(np.absolute(Pyx - PAN[y-mody:y+1, x:x-modx+gsd, 0]))
                M7      = np.zeros((nb,))
                c_spat7 = 1
            else:
                N7      = np.sum(np.absolute(Pyx-PAN[y-mody-gsd:y-mody, x-modx+gsd:x-modx+(2*gsd), 0])) + np.sum(np.absolute(Pyx-PAN[y-mody:y+1, x:x-modx+gsd, 0]))
                M7      = msi[int(y//gsd)-1, int(x//gsd)+1]
                yuv,xuv = y-mody-0.5*gsd, x-modx+1.5*gsd
                c_spat7 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT7 = np.sum(M7*Tvector)
            
            
            #REGION 8
            if x in range(NC-4, NC):
                N8      = np.sum(np.absolute(Pyx - PAN[y-mody:y-mody+gsd, x:x-modx+gsd, 0]))
                M8      = np.zeros((nb,))
                c_spat8 = 1
                
            else:
                N8      = np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx+gsd:x-modx+(2*gsd), 0])) + np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x:x-modx+gsd, 0]))
                M8      = msi[int(y//gsd), int(x//gsd)+1]
                yuv,xuv = y-mody+0.5*gsd, x-modx+1.5*gsd
                c_spat8 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT8 = np.sum(M8*Tvector)
            
            
            #REGION 9
            if y in range(NR-4,NR) or x in range(NC-4, NC):
                N9      = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x:x-modx+gsd, 0]))
                M9      = np.zeros((nb,))
                c_spat9 = 1
            else:
                N9      = np.sum(np.absolute(Pyx-PAN[y-mody+gsd:y-mody+(2*gsd), x-modx+gsd:x-modx+(2*gsd), 0])) + np.sum(np.absolute(Pyx-PAN[y:y-mody+gsd, x:x-modx+gsd, 0]))
                M9      = msi[int(y//gsd)+1, int(x//gsd)]
                yuv,xuv = y-mody+1.5*gsd, x-modx+1.5*gsd
                c_spat9 = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            MT9 = np.sum(M9*Tvector)
            
            #------------------------------------------------------------------
            N_min       = np.min([N1, N2, N3, N4, N5, N6, N7, N8, N9])
            sigma2_int  = np.max([sigma2_int_floor, N_min])
            
            #Calculate intensity factors
            c_int1 = np.exp(-N1/sigma2_int)
            c_int2 = np.exp(-N2/sigma2_int)
            c_int3 = np.exp(-N3/sigma2_int)
            c_int4 = np.exp(-N4/sigma2_int)
            c_int5 = np.exp(-N5/sigma2_int)
            c_int6 = np.exp(-N6/sigma2_int)
            c_int7 = np.exp(-N7/sigma2_int)
            c_int8 = np.exp(-N8/sigma2_int)
            c_int9 = np.exp(-N9/sigma2_int)
            
            k_xy         = float((1/PAN[y,x, :])*((c_int1*c_spat1*MT1) + (c_int2*c_spat2*MT2) + (c_int3*c_spat3*MT3) + (c_int4*c_spat4*MT4) + (c_int5*c_spat5*MT5) + (c_int6*c_spat6*MT6) + (c_int7*c_spat7*MT7) + (c_int8*c_spat8*MT8) + (c_int9*c_spat9*MT9)))
            HRMSI[y,x,:] = (1/k_xy)*((c_int1*c_spat1*M1) + (c_int2*c_spat2*M2) + (c_int3*c_spat3*M3) + (c_int4*c_spat4*M4) + (c_int5*c_spat5*M5) + (c_int6*c_spat6*M6) + (c_int7*c_spat7*M7) + (c_int8*c_spat8*M8) + (c_int9*c_spat9*M9))

    loop_timer1     = time.time()
    loop_total_time = loop_timer1-loop_timer0
    print('loop_total_time: \n', loop_total_time)
    return(HRMSI)



