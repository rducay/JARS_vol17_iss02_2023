import numpy as np
import sys
sys.path.insert(1, 'D:/_RESEARCH/_functions/')
import time
# 'major' superpixel: the superpixel surrounding but NOT containing the pixel of interest (x,y)
# 'minor' superpixel: the superpixel containing the pixel of interest (x,y)

def calc_Tvector(PAN, msi):
    '''
    The inputs should have the following array shapes
    PAN: (NR, NC, 1, NPX)
    msi: (nr, nc, nb, npxx)
    '''
    nr, nc, nb, npx = msi.shape[0], msi.shape[1], msi.shape[2], msi.shape[0]*msi.shape[1]
    NR              = PAN.shape[0]
    gsd_ratio       = int(NR/nr)
    
    #Reduce resolution (followed Wayne's Matlab code)
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
        #costheta = np.dot(vec1.T, vec2)/(vec1mag*vec2mag)
        costheta = float(np.dot(vec1.T, vec2))/(vec1mag*vec2mag)
        if mode=='cosine':
            return(costheta)
        elif mode=='radians':
            if costheta<-1:
                return(np.arccos(-1))
            elif costheta>1:
                return(np.arccos(1))
            else:
                return(np.arccos(costheta))
        else:
            raise Exception('Specify mode as "cosine" or "radians"')

def pansharpen_multiversion(PAN, msi, LRHSI_EDGEMAP, sigma_spat=2.5, sigma_int=1):
    #For the 'old' NNDiffuse version, both the intensity and spatial smoothness parameters are set to a constant
    #version 0: the old, non-adaptive Wayne version
    #version 1: same as old version but the binary exp_edge (0,1)
    #version 2: include spectral angle term 'exp_sa' AND adaptive intensity term ('sa' terms are squared inside 'exp_sa' terms)
    #version 3: same as version 2, BUT WITH with binary exp_edge (0,1) term included
    #version 4: same as version 2, EXCEPT the 'sa' terms inside 'exp_sa' are NOT squared
    #version 5: same as version 4 BUT with binary exp_edge (0,1)
        
    sigma_int2_floor = 0.1
    sigma_sa_floor   = 0.001
    
    #Image dimensions
    #Hi-res: UPPERCASE; low-res: lowercase
    NR, NC, NB, NPX = PAN.shape[0], PAN.shape[1], PAN.shape[2], PAN.shape[0]*PAN.shape[1]
    nr, nc, nb, npx = msi.shape[0], msi.shape[1], msi.shape[2], msi.shape[0]*msi.shape[1]
    gsd             = NR/nr #gsd ratio 
    
    #Check if PAN/msi GSD ratio makes sense
    if gsd-np.round(gsd) != 0:
        raise Exception('PAN and MSI inputs has to have integer GSD ratio')
    elif gsd<=1:
        raise Exception('PAN/MSI GSD ratio has to be greater than 1')
    elif gsd==0:
        raise Exception('GSD ratio cannot be zero')
    else:
        gsd = int(gsd)
    
    #Downsample the PAN and generate T vector
    Tvector = calc_Tvector(PAN=PAN, msi=msi)
    
    #Loop through each pixel on the high-res PAN
    HRMSI_NNDIFFUSE_V0 = np.empty((NR, NC, nb))
    HRMSI_NNDIFFUSE_V1 = np.copy(HRMSI_NNDIFFUSE_V0)
    HRMSI_NNDIFFUSE_V2 = np.copy(HRMSI_NNDIFFUSE_V0)
    HRMSI_NNDIFFUSE_V3 = np.copy(HRMSI_NNDIFFUSE_V0)
    HRMSI_NNDIFFUSE_V4 = np.copy(HRMSI_NNDIFFUSE_V0)
    HRMSI_NNDIFFUSE_V5 = np.copy(HRMSI_NNDIFFUSE_V0)
    pixcount = 0
    loop_timer0 = time.time()
    print('NNDiffuse fusion...four versions')
    for y in range(NR):
        for x in range(NC):
            pixcount += 1
            mody, modx = y%gsd, x%gsd
            #Prevent zero division
            if PAN[y,x,:]==0: Pyx = PAN[y,x,:]+1e-4
            if PAN[y,x,:]!=0: Pyx = PAN[y,x,:]+0
            
            #REGION 5: Current superpixel (pixel spectrum: M5)
            rmin0,rmin1 = y-mody, y-mody+gsd
            cmin0,cmin1 = x-modx, x-modx+gsd
            N5          = np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
            M5          = msi[int(y//gsd), int(x//gsd)]
            yuv,xuv     = 0.5*(rmin0+rmin1), 0.5*(cmin0+cmin1)
            exp_spat5   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
            #Deal with edges: edge-ness score of interest
            esoi5       = np.mean(LRHSI_EDGEMAP[rmin0:rmin1, cmin0:cmin1])
            exp_edge5   = 1
            del(rmin0,rmin1,cmin0,cmin1,yuv,xuv)
            MT5 = np.sum(M5*Tvector)
            
            #REGION 1
            if y in range(4) or x in range(4):
                N1         = np.sum(np.absolute(Pyx - PAN[y-mody:y, x-modx:x, 0]))
                M1         = np.zeros((nb,))
                exp_spat1  = 1
                exp_edge1  = 1
            else:
                rmaj0,rmaj1 = y-mody-gsd, y-mody
                cmaj0,cmaj1 = x-modx-gsd, x-modx
                rmin0,rmin1 = y-mody, y
                cmin0,cmin1 = x-modx, x
                N1          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M1          = msi[int(y//gsd)-1, int(x//gsd)-1]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat1   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi1       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi1==esoi5: exp_edge1=1
                if esoi1!=esoi5: exp_edge1=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi1)
            MT1 = np.sum(M1*Tvector)
            
            #REGION 2
            if x in range(4):
                N2          = np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx:x+1, 0]))
                M2          = np.zeros((nb,))
                exp_spat2   = 1
                exp_edge2   = 1
            else:
                rmaj0,rmaj1 = y-mody, y-mody+gsd
                cmaj0,cmaj1 = x-modx-gsd, x-modx
                rmin0,rmin1 = y-mody, y-mody+gsd
                cmin0,cmin1 = x-modx, x+1
                N2          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M2          = msi[int(y//gsd), int(x//gsd)-1]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat2   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi2       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi2==esoi5: exp_edge2=1
                if esoi2!=esoi5: exp_edge2=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi2)
            MT2 = np.sum(M2*Tvector)
            
            #REGION 3
            if y in range(NR-4,NR) or x in range(4):
                N3          = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x-modx:x+1, 0]))
                M3          = np.zeros((nb,))
                exp_spat3   = 1
                exp_edge3   = 1
            else:
                rmaj0,rmaj1 = y-mody+gsd, y-mody+(2*gsd)
                cmaj0,cmaj1 = x-modx-gsd, x-modx
                rmin0,rmin1 = y, y-mody+gsd
                cmin0,cmin1 = x-modx, x+1
                N3          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M3          = msi[int(y//gsd)+1, int(x//gsd)-1]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat3   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi3       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi3==esoi5: exp_edge3=1
                if esoi3!=esoi5: exp_edge3=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi3)
            MT3 = np.sum(M3*Tvector)
            
            #REGION 4
            if y in range(4):
                N4         = np.sum(np.absolute(Pyx - PAN[y-mody:y+1, x-modx:x-modx+gsd, 0]))
                M4         = np.zeros((nb,))
                exp_spat4  = 1
                exp_edge4  = 1
            else:
                rmaj0,rmaj1 = y-mody-gsd, y-mody
                cmaj0,cmaj1 = x-modx, x-modx+gsd
                rmin0,rmin1 = y-mody, y+1
                cmin0,cmin1 = x-modx, x-modx+gsd
                N4          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M4          = msi[int(y//gsd)-1, int(x//gsd)]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat4   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi4       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi4==esoi5: exp_edge4=1
                if esoi4!=esoi5: exp_edge4=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi4)
            MT4 = np.sum(M4*Tvector)
            
            #REGION 6
            if y in range(NR-4, NR):
                N6          = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x-modx:x-modx+gsd, 0]))
                M6          = np.zeros((nb,))
                exp_spat6   = 1
                exp_edge6   = 1
            else:
                rmaj0,rmaj1 = y-mody+gsd, y-mody+(2*gsd)
                cmaj0,cmaj1 = x-modx, x-modx+gsd
                rmin0,rmin1 = y, y-mody+gsd
                cmin0,cmin1 = x-modx, x-modx+gsd
                N6          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M6          = msi[int(y//gsd)+1, int(x//gsd)]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat6   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi6       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi6==esoi5: exp_edge6=1
                if esoi6!=esoi5: exp_edge6=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi6)
            MT6 = np.sum(M6*Tvector)
            
            #REGION 7
            if y in range(4) or x in range(NC-4, NC):
                N7          = np.sum(np.absolute(Pyx - PAN[y-mody:y+1, x:x-modx+gsd, 0]))
                M7          = np.zeros((nb,))
                exp_spat7   = 1
                exp_edge7   = 1
            else:
                rmaj0,rmaj1 = y-mody-gsd, y-mody
                cmaj0,cmaj1 = x-modx+gsd, x-modx+(2*gsd)
                rmin0,rmin1 = y-mody, y+1
                cmin0,cmin1 = x, x-modx+gsd
                N7          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M7          = msi[int(y//gsd)-1, int(x//gsd)+1]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat7   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi7       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi7==esoi5: exp_edge7=1
                if esoi7!=esoi5: exp_edge7=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi7)
            MT7 = np.sum(M7*Tvector)
            
            #REGION 8
            if x in range(NC-4, NC):
                N8          = np.sum(np.absolute(Pyx - PAN[y-mody:y-mody+gsd, x:x-modx+gsd, 0]))
                M8          = np.zeros((nb,))
                exp_spat8   = 1
                exp_edge8   = 1
            else:
                rmaj0,rmaj1 = y-mody, y-mody+gsd
                cmaj0,cmaj1 = x-modx+gsd, x-modx+(2*gsd)
                rmin0,rmin1 = y-mody, y-mody+gsd
                cmin0,cmin1 = x, x-modx+gsd
                N8          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M8          = msi[int(y//gsd), int(x//gsd)+1]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat8   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi8       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi8==esoi5: exp_edge8=1
                if esoi8!=esoi5: exp_edge8=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi8)
            MT8 = np.sum(M8*Tvector)
            
            #REGION 9
            if y in range(NR-4,NR) or x in range(NC-4, NC):
                N9          = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x:x-modx+gsd, 0]))
                M9          = np.zeros((nb,))
                exp_spat9   = 1
                exp_edge9   = 1
            else:
                rmaj0,rmaj1 = y-mody+gsd, y-mody+(2*gsd)
                cmaj0,cmaj1 = x-modx+gsd, x-modx+(2*gsd)
                rmin0,rmin1 = y,y-mody+gsd
                cmin0,cmin1 = x,x-modx+gsd
                N9          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                M9          = msi[int(y//gsd)+1, int(x//gsd)]
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat9   = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spat**2)
                #Deal with edges: edge-ness score of interest
                esoi9       = np.mean(LRHSI_EDGEMAP[rmaj0:rmaj1, cmaj0:cmaj1])
                if esoi9==esoi5: exp_edge9=1
                if esoi9!=esoi5: exp_edge9=0
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1,yuv,xuv,esoi9)
            MT9 = np.sum(M9*Tvector)
            
            # =================================================================
            #             #VERSION 0
            # =================================================================
            sigma_int2 = 1
            # Calculate intensity gaussians
            exp_int1 = np.exp(-N1/sigma_int2)
            exp_int2 = np.exp(-N2/sigma_int2)
            exp_int3 = np.exp(-N3/sigma_int2)
            exp_int4 = np.exp(-N4/sigma_int2)
            exp_int5 = np.exp(-N5/sigma_int2)
            exp_int6 = np.exp(-N6/sigma_int2)
            exp_int7 = np.exp(-N7/sigma_int2)
            exp_int8 = np.exp(-N8/sigma_int2)
            exp_int9 = np.exp(-N9/sigma_int2)
            
            #Calculate normalization factor k_xy
            p1 = exp_int1*exp_spat1*MT1
            p2 = exp_int2*exp_spat2*MT2
            p3 = exp_int3*exp_spat3*MT3
            p4 = exp_int4*exp_spat4*MT4
            p5 = exp_int5*exp_spat5*MT5
            p6 = exp_int6*exp_spat6*MT6
            p7 = exp_int7*exp_spat7*MT7
            p8 = exp_int8*exp_spat8*MT8
            p9 = exp_int9*exp_spat9*MT9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            # Calculate HM(xy)
            q1 = exp_int1*exp_spat1*M1  # q1.shape:  (8,)
            q2 = exp_int2*exp_spat2*M2
            q3 = exp_int3*exp_spat3*M3
            q4 = exp_int4*exp_spat4*M4
            q5 = exp_int5*exp_spat5*M5
            q6 = exp_int6*exp_spat6*M6
            q7 = exp_int7*exp_spat7*M7
            q8 = exp_int8*exp_spat8*M8
            q9 = exp_int9*exp_spat9*M9
            HRMSI_NNDIFFUSE_V0[y, x, :] = np.ravel((1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9))
            
            # =================================================================
            #             #VERSION 1
            # =================================================================
            #Same as version 0, except there's an added binary exp_edgei factor
            #Calculate normalization factor k_xy
            p1 = exp_edge1*p1
            p2 = exp_edge2*p2
            p3 = exp_edge3*p3
            p4 = exp_edge4*p4
            p5 = exp_edge5*p5
            p6 = exp_edge6*p6
            p7 = exp_edge7*p7
            p8 = exp_edge8*p8
            p9 = exp_edge9*p9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            # Calculate HM(xy)
            q1 = exp_edge1*q1  # q1.shape:  (8,)
            q2 = exp_edge2*q2
            q3 = exp_edge3*q3
            q4 = exp_edge4*q4
            q5 = exp_edge5*q5
            q6 = exp_edge6*q6
            q7 = exp_edge7*q7
            q8 = exp_edge8*q8
            q9 = exp_edge9*q9
            HRMSI_NNDIFFUSE_V1[y, x, :] = np.ravel((1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9))
            #In version 2, the intensity parameter is adaptive, so delete all p, q, exp_int values
            del(p1, p2, p3, p4, p5, p6, p7, p8, p9, q1, q2, q3, q4, q5, q6, q7, q8, q9, k_xy)
            del(sigma_int2,exp_int1,exp_int2,exp_int3,exp_int4,exp_int5,exp_int6,exp_int7,exp_int8,exp_int9)
            
            # =================================================================
            #             #VERSION 2
            # =================================================================
            #version 2: include spectral angle term 'exp_sa' AND adaptive intensity term
            
            #Adaptive intensity calculations
            N_min      = np.min([N1, N2, N3, N4, N5, N6, N7, N8, N9])
            sigma_int2 = np.max([sigma_int2_floor, N_min])
            exp_int1   = np.exp(-N1/sigma_int2)
            exp_int2   = np.exp(-N2/sigma_int2)
            exp_int3   = np.exp(-N3/sigma_int2)
            exp_int4   = np.exp(-N4/sigma_int2)
            exp_int5   = np.exp(-N5/sigma_int2)
            exp_int6   = np.exp(-N6/sigma_int2)
            exp_int7   = np.exp(-N7/sigma_int2)
            exp_int8   = np.exp(-N8/sigma_int2)
            exp_int9   = np.exp(-N9/sigma_int2)
            
            #Adaptive spectral angle calculations (put spectral angle in radians)
            sa1        = np.absolute(calc_spectral_angle(M5, M1, mode='radians'))
            sa2        = np.absolute(calc_spectral_angle(M5, M2, mode='radians'))
            sa3        = np.absolute(calc_spectral_angle(M5, M3, mode='radians'))
            sa4        = np.absolute(calc_spectral_angle(M5, M4, mode='radians'))
            sa6        = np.absolute(calc_spectral_angle(M5, M6, mode='radians'))
            sa7        = np.absolute(calc_spectral_angle(M5, M7, mode='radians'))
            sa8        = np.absolute(calc_spectral_angle(M5, M8, mode='radians'))
            sa9        = np.absolute(calc_spectral_angle(M5, M9, mode='radians'))
            
            sa_min     = np.min([sa1,sa2,sa3,sa4,sa6,sa7,sa8,sa9])
            sigma_sa   = np.max([sigma_sa_floor, sa_min])
            exp_sa1    = np.exp(-(sa1**2)/(sigma_sa**2))
            exp_sa2    = np.exp(-(sa2**2)/(sigma_sa**2))
            exp_sa3    = np.exp(-(sa3**2)/(sigma_sa**2))
            exp_sa4    = np.exp(-(sa4**2)/(sigma_sa**2))
            exp_sa5    = 1
            exp_sa6    = np.exp(-(sa6**2)/(sigma_sa**2))
            exp_sa7    = np.exp(-(sa7**2)/(sigma_sa**2))
            exp_sa8    = np.exp(-(sa8**2)/(sigma_sa**2))
            exp_sa9    = np.exp(-(sa9**2)/(sigma_sa**2))
            
            #Calculate normalization factor k_xy
            p1 = exp_int1*exp_spat1*exp_sa1*MT1
            p2 = exp_int2*exp_spat2*exp_sa2*MT2
            p3 = exp_int3*exp_spat3*exp_sa3*MT3
            p4 = exp_int4*exp_spat4*exp_sa4*MT4
            p5 = exp_int5*exp_spat5*exp_sa5*MT5
            p6 = exp_int6*exp_spat6*exp_sa6*MT6
            p7 = exp_int7*exp_spat7*exp_sa7*MT7
            p8 = exp_int8*exp_spat8*exp_sa8*MT8
            p9 = exp_int9*exp_spat9*exp_sa9*MT9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            # Calculate HM(xy)
            q1 = exp_int1*exp_spat1*exp_sa1*M1  # q1.shape:  (8,)
            q2 = exp_int2*exp_spat2*exp_sa2*M2
            q3 = exp_int3*exp_spat3*exp_sa3*M3
            q4 = exp_int4*exp_spat4*exp_sa4*M4
            q5 = exp_int5*exp_spat5*exp_sa5*M5
            q6 = exp_int6*exp_spat6*exp_sa6*M6
            q7 = exp_int7*exp_spat7*exp_sa7*M7
            q8 = exp_int8*exp_spat8*exp_sa8*M8
            q9 = exp_int9*exp_spat9*exp_sa9*M9
            HRMSI_NNDIFFUSE_V2[y, x, :] = np.ravel((1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9))
            
            # =================================================================
            #             #VERSION 3
            # =================================================================
            #version 3: same as version 2, BUT WITH with binary exp_edge (0,1) term
            p1 = exp_edge1*p1
            p2 = exp_edge2*p2
            p3 = exp_edge3*p3
            p4 = exp_edge4*p4
            p5 = exp_edge5*p5
            p6 = exp_edge6*p6
            p7 = exp_edge7*p7
            p8 = exp_edge8*p8
            p9 = exp_edge9*p9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            q1 = exp_edge1*q1
            q2 = exp_edge2*q2
            q3 = exp_edge3*q3
            q4 = exp_edge4*q4
            q5 = exp_edge5*q5
            q6 = exp_edge6*q6
            q7 = exp_edge7*q7
            q8 = exp_edge8*q8
            q9 = exp_edge9*q9
            HRMSI_NNDIFFUSE_V3[y, x, :] = np.ravel((1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9))
            #The next version (version 4) is the same as version 2, 
            #EXCEPT the 'sai' terms inside 'exp_sa' are NOT squared
            #the 'sa' terms are still the same (spectral angle in radians)
            del(p1, p2, p3, p4, p5, p6, p7, p8, p9, q1, q2, q3, q4, q5, q6, q7, q8, q9, k_xy)
            del(exp_sa1,exp_sa2,exp_sa3,exp_sa4,exp_sa5,exp_sa6,exp_sa7,exp_sa8,exp_sa9)
            
            # =================================================================
            #             #VERSION 4
            # =================================================================
            #version 4: same as version 2, EXCEPT the 'sai' terms inside 'exp_sa' are NOT squared
            sigma_sa2  = np.max([sigma_sa_floor, sa_min])
            exp_sa1    = np.exp(-sa1/sigma_sa2)
            exp_sa2    = np.exp(-sa2/sigma_sa2)
            exp_sa3    = np.exp(-sa3/sigma_sa2)
            exp_sa4    = np.exp(-sa4/sigma_sa2)
            exp_sa5    = 1
            exp_sa6    = np.exp(-sa6/sigma_sa2)
            exp_sa7    = np.exp(-sa7/sigma_sa2)
            exp_sa8    = np.exp(-sa8/sigma_sa2)
            exp_sa9    = np.exp(-sa9/sigma_sa2)
            
            #Calculate normalization factor k_xy
            p1 = exp_int1*exp_spat1*exp_sa1*MT1
            p2 = exp_int2*exp_spat2*exp_sa2*MT2
            p3 = exp_int3*exp_spat3*exp_sa3*MT3
            p4 = exp_int4*exp_spat4*exp_sa4*MT4
            p5 = exp_int5*exp_spat5*exp_sa5*MT5
            p6 = exp_int6*exp_spat6*exp_sa6*MT6
            p7 = exp_int7*exp_spat7*exp_sa7*MT7
            p8 = exp_int8*exp_spat8*exp_sa8*MT8
            p9 = exp_int9*exp_spat9*exp_sa9*MT9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            # Calculate HM(xy)
            q1 = exp_int1*exp_spat1*exp_sa1*M1  # q1.shape:  (8,)
            q2 = exp_int2*exp_spat2*exp_sa2*M2
            q3 = exp_int3*exp_spat3*exp_sa3*M3
            q4 = exp_int4*exp_spat4*exp_sa4*M4
            q5 = exp_int5*exp_spat5*exp_sa5*M5
            q6 = exp_int6*exp_spat6*exp_sa6*M6
            q7 = exp_int7*exp_spat7*exp_sa7*M7
            q8 = exp_int8*exp_spat8*exp_sa8*M8
            q9 = exp_int9*exp_spat9*exp_sa9*M9
            HRMSI_NNDIFFUSE_V4[y, x, :] = np.ravel((1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9))
            del(k_xy)
            
            # =================================================================
            #             #VERSION 5
            # =================================================================
            #version 5: same as version 4 BUT with binary exp_edge (0,1)
            
            #Calculate normalization factor k_xy
            p1 = exp_edge1*p1
            p2 = exp_edge2*p2
            p3 = exp_edge3*p3
            p4 = exp_edge4*p4
            p5 = exp_edge5*p5
            p6 = exp_edge6*p6
            p7 = exp_edge7*p7
            p8 = exp_edge8*p8
            p9 = exp_edge9*p9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            # Calculate HM(xy)
            q1 = exp_edge1*q1
            q2 = exp_edge2*q2
            q3 = exp_edge3*q3
            q4 = exp_edge4*q4
            q5 = exp_edge5*q5
            q6 = exp_edge6*q6
            q7 = exp_edge7*q7
            q8 = exp_edge8*q8
            q9 = exp_edge9*q9
            HRMSI_NNDIFFUSE_V5[y, x, :] = np.ravel((1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9))
            del(p1, p2, p3, p4, p5, p6, p7, p8, p9, q1, q2, q3, q4, q5, q6, q7, q8, q9, k_xy)
            #del(exp_int1,exp_int2,exp_int3,exp_int4,exp_int5,exp_int6,exp_int7,exp_int8,exp_int9)
            del(exp_spat1,exp_spat2,exp_spat3,exp_spat4,exp_spat5,exp_spat6,exp_spat7,exp_spat8,exp_spat9)
            del(exp_sa1,exp_sa2,exp_sa3,exp_sa4,exp_sa5,exp_sa6,exp_sa7,exp_sa8,exp_sa9)
            del(exp_edge1,exp_edge2,exp_edge3,exp_edge4,exp_edge5,exp_edge6,exp_edge7,exp_edge8,exp_edge9)
    
    loop_timer1 = time.time()
    loop_total_time = loop_timer1-loop_timer0
    print('loop_total_time: \n', loop_total_time)
    HRMSI_NNDIFFUSE_VERSIONSLIST = [HRMSI_NNDIFFUSE_V0, HRMSI_NNDIFFUSE_V1, HRMSI_NNDIFFUSE_V2, HRMSI_NNDIFFUSE_V3, HRMSI_NNDIFFUSE_V4, HRMSI_NNDIFFUSE_V5]
    return(HRMSI_NNDIFFUSE_VERSIONSLIST)



