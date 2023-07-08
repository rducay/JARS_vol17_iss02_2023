import numpy as np
import sys
sys.path.insert(1, 'D:/_RESEARCH/_functions/')
import my_walds_protocol as wp
import time
# 'major' superpixel: the superpixel surrounding but NOT containing the pixel of interest (x,y)
# 'minor' superpixel: the superpixel containing the pixel of interest (x,y)

def calc_spectral_angle_cosine(vec1, vec2):
    vec1 = vec1.reshape((-1, 1))
    vec2 = vec2.reshape((-1, 1))
    vec1mag = np.sqrt(np.dot(vec1.T, vec1))
    vec2mag = np.sqrt(np.dot(vec2.T, vec2))
    if vec1mag==0 or vec2mag==0:
        return(0)
    else:
        return(np.dot(vec1.T, vec2)/(vec1mag*vec2mag))

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
    #w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    w = np.matmul(np.linalg.pinv(X), y) #equivalent to above but in one line of code
    return(w.ravel())

def generate_majRowRanges(y_poi, gsd_ratio, region_num):
    mod_y_poi = y_poi % gsd_ratio
    majRowRanges = [(y_poi-mod_y_poi-gsd_ratio, y_poi-mod_y_poi),
                    (y_poi-mod_y_poi,           y_poi-mod_y_poi+gsd_ratio),
                    (y_poi-mod_y_poi+gsd_ratio, y_poi-mod_y_poi+(2*gsd_ratio)),
                    (y_poi-mod_y_poi-gsd_ratio, y_poi-mod_y_poi),
                    (False,                     False),
                    (y_poi-mod_y_poi+gsd_ratio, y_poi-mod_y_poi+(2*gsd_ratio)),
                    (y_poi-mod_y_poi-gsd_ratio, y_poi-mod_y_poi),
                    (y_poi-mod_y_poi,           y_poi-mod_y_poi+gsd_ratio),
                    (y_poi-mod_y_poi+gsd_ratio, y_poi-mod_y_poi+(2*gsd_ratio))]
    return(majRowRanges[region_num])

def generate_majColRanges(x_poi, gsd_ratio, region_num):
    mod_x_poi = x_poi % gsd_ratio
    majColRanges = [(x_poi-mod_x_poi-gsd_ratio, x_poi-mod_x_poi),
                    (x_poi-mod_x_poi-gsd_ratio, x_poi-mod_x_poi),
                    (x_poi-mod_x_poi-gsd_ratio, x_poi-mod_x_poi),
                    (x_poi-mod_x_poi,           x_poi-mod_x_poi+gsd_ratio),
                    (False,                     False),
                    (x_poi-mod_x_poi,           x_poi-mod_x_poi+gsd_ratio),
                    (x_poi-mod_x_poi+gsd_ratio, x_poi-mod_x_poi+(2*gsd_ratio)),
                    (x_poi-mod_x_poi+gsd_ratio, x_poi-mod_x_poi+(2*gsd_ratio)),
                    (x_poi-mod_x_poi+gsd_ratio, x_poi-mod_x_poi+(2*gsd_ratio))]
    return(majColRanges[region_num])


def generate_minRowRanges(y_poi, gsd_ratio, region_num):
    mod_y_poi = y_poi % gsd_ratio
    minRowRanges = [(y_poi-mod_y_poi,           y_poi+1),
                    (y_poi-mod_y_poi,           y_poi-mod_y_poi+gsd_ratio),
                    (y_poi,                     y_poi-mod_y_poi+gsd_ratio),
                    (y_poi-mod_y_poi,           y_poi+1),
                    (y_poi-mod_y_poi,           y_poi-mod_y_poi+gsd_ratio),
                    (y_poi,                     y_poi-mod_y_poi+gsd_ratio),
                    (y_poi-mod_y_poi,           y_poi+1),
                    (y_poi-mod_y_poi,           y_poi-mod_y_poi+gsd_ratio),
                    (y_poi,                     y_poi-mod_y_poi+gsd_ratio)]
    return(minRowRanges[region_num])

def generate_minColRanges(x_poi, gsd_ratio, region_num):
    mod_x_poi = x_poi % gsd_ratio
    minColRanges = [(x_poi-mod_x_poi, x_poi+1),
                    (x_poi-mod_x_poi, x_poi+1),
                    (x_poi-mod_x_poi, x_poi+1),
                    (x_poi-mod_x_poi, x_poi-mod_x_poi+gsd_ratio),
                    (x_poi-mod_x_poi, x_poi-mod_x_poi+gsd_ratio),
                    (x_poi-mod_x_poi, x_poi-mod_x_poi+gsd_ratio),
                    (x_poi, x_poi-mod_x_poi+gsd_ratio),
                    (x_poi, x_poi-mod_x_poi+gsd_ratio),
                    (x_poi, x_poi-mod_x_poi+gsd_ratio)]
    return(minColRanges[region_num])

def nndiffuse_pansharpen(PAN, msi, sigma_spatial=2.5, sigma_inten_adaptive=True, sigma_sacos=True):
    #sigma_inten2_floor = 0.001
    sigma_inten2_floor = 0.1
    
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
    #print('T_pseudo: \n', Tvector)
    
    HRMSI_NNDIFFUSE = np.empty((NR, NC, nb))
    #Loop through each pixel on the high-res PAN
    pixcount = 0
    loop_timer0 = time.time()
    
    #Upsample the MSI to match spatial dimensions with the PAN
    MSI_US = np.kron(msi, np.ones((4,4,1)))
    print('NNDiffuse fusion...')
    for y in range(NR):
        for x in range(NC):
            pixcount += 1
            mody, modx = y%gsd, x%gsd
            if PAN[y,x,:]==0:
                Pyx = PAN[y,x,:]+1e-4 #to prevent zero division
            if PAN[y,x,:]!=0:
                Pyx = PAN[y,x,:]
            
            #REGION 1
            if y in range(4) or x in range(4):
                N1      = np.sum(np.absolute(Pyx - PAN[y-mody:y, x-modx:x, 0]))
                M1      = np.zeros((nb,))
                exp_spat1  = 1
                
            else:
                rmaj0,rmaj1 = y-mody-gsd, y-mody
                cmaj0,cmaj1 = x-modx-gsd, x-modx
                rmin0,rmin1 = y-mody, y
                cmin0,cmin1 = x-modx, x
                N1          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi1       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M1          = np.mean(np.reshape(spoi1, (spoi1.shape[0]*spoi1.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat1      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi1,yuv,xuv)
            MT1 = np.sum(M1*Tvector)
            
            #REGION 2
            if x in range(4):
                N2      = np.sum(np.absolute(Pyx-PAN[y-mody:y-mody+gsd, x-modx:x+1, 0]))
                M2      = np.zeros((nb,))
                exp_spat2  = 1
                
            else:
                rmaj0,rmaj1 = y-mody, y-mody+gsd
                cmaj0,cmaj1 = x-modx-gsd, x-modx
                rmin0,rmin1 = y-mody, y-mody+gsd
                cmin0,cmin1 = x-modx, x+1
                N2          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi2       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M2          = np.mean(np.reshape(spoi2, (spoi2.shape[0]*spoi2.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat2      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi2,yuv,xuv)
            MT2 = np.sum(M2*Tvector)
            
            #REGION 3
            if y in range(NR-4,NR) or x in range(4):
                N3      = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x-modx:x+1, 0]))
                M3      = np.zeros((nb,))
                exp_spat3  = 1 
                
            else:
                rmaj0,rmaj1 = y-mody+gsd, y-mody+(2*gsd)
                cmaj0,cmaj1 = x-modx-gsd, x-modx
                rmin0,rmin1 = y, y-mody+gsd
                cmin0,cmin1 = x-modx, x+1
                N3          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi3       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M3          = np.mean(np.reshape(spoi3, (spoi3.shape[0]*spoi3.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat3      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi3,yuv,xuv)
            MT3 = np.sum(M3*Tvector)
            
            #REGION 4
            if y in range(4):
                N4      = np.sum(np.absolute(Pyx - PAN[y-mody:y+1, x-modx:x-modx+gsd, 0]))
                M4      = np.zeros((nb,))
                exp_spat4  = 1 
                
            else:
                rmaj0,rmaj1 = y-mody-gsd, y-mody
                cmaj0,cmaj1 = x-modx, x-modx+gsd
                rmin0,rmin1 = y-mody, y+1
                cmin0,cmin1 = x-modx, x-modx+gsd
                N4          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi4       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M4          = np.mean(np.reshape(spoi4, (spoi4.shape[0]*spoi4.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat4      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi4,yuv,xuv)
            MT4 = np.sum(M4*Tvector)
            
            #REGION 5
            rmin0,rmin1 = y-mody, y-mody+gsd
            cmin0,cmin1 = x-modx, x-modx+gsd
            N5          = np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
            spoi5       = MSI_US[rmin0:rmin1, cmin0:cmin1]
            M5          = np.mean(np.reshape(spoi5, (spoi5.shape[0]*spoi5.shape[1], nb)), axis=0)
            yuv,xuv     = 0.5*(rmin0+rmin1), 0.5*(cmin0+cmin1)
            exp_spat5      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
            del(rmin0,rmin1,cmin0,cmin1, spoi5,yuv,xuv)
            MT5 = np.sum(M5*Tvector)
            
            #REGION 6
            if y in range(NR-4, NR):
                N6      = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x-modx:x-modx+gsd, 0]))
                M6      = np.zeros((nb,))
                exp_spat6  = 1 
                
            else:
                rmaj0,rmaj1 = y-mody+gsd, y-mody+(2*gsd)
                cmaj0,cmaj1 = x-modx, x-modx+gsd
                rmin0,rmin1 = y, y-mody+gsd
                cmin0,cmin1 = x-modx, x-modx+gsd
                N6          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi6       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M6          = np.mean(np.reshape(spoi6, (spoi6.shape[0]*spoi6.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat6      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi6,yuv,xuv)
            MT6 = np.sum(M6*Tvector)
            
            #REGION 7
            if y in range(4) or x in range(NC-4, NC):
                N7      = np.sum(np.absolute(Pyx - PAN[y-mody:y+1, x:x-modx+gsd, 0]))
                M7      = np.zeros((nb,))
                exp_spat7  = 1 
                
            else:
                rmaj0,rmaj1 = y-mody-gsd, y-mody
                cmaj0,cmaj1 = x-modx+gsd, x-modx+(2*gsd)
                rmin0,rmin1 = y-mody, y+1
                cmin0,cmin1 = x, x-modx+gsd
                N7          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi7       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M7          = np.mean(np.reshape(spoi7, (spoi7.shape[0]*spoi7.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat7      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi7,yuv,xuv)
            MT7 = np.sum(M7*Tvector)
            
            #REGION 8
            if x in range(NC-4, NC):
                N8      = np.sum(np.absolute(Pyx - PAN[y-mody:y-mody+gsd, x:x-modx+gsd, 0]))
                M8      = np.zeros((nb,))
                exp_spat8  = 1 
                
            else:
                rmaj0,rmaj1 = y-mody, y-mody+gsd
                cmaj0,cmaj1 = x-modx+gsd, x-modx+(2*gsd)
                rmin0,rmin1 = y-mody, y-mody+gsd
                cmin0,cmin1 = x, x-modx+gsd
                N8          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi8       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M8          = np.mean(np.reshape(spoi8, (spoi8.shape[0]*spoi8.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat8      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi8,yuv,xuv)
            MT8 = np.sum(M8*Tvector)
            
            #REGION 9
            if y in range(NR-4,NR) or x in range(NC-4, NC):
                N9      = np.sum(np.absolute(Pyx - PAN[y:y-mody+gsd, x:x-modx+gsd, 0]))
                M9      = np.zeros((nb,))
                exp_spat9  = 1 
                
            else:
                rmaj0,rmaj1 = y-mody+gsd, y-mody+(2*gsd)
                cmaj0,cmaj1 = x-modx+gsd, x-modx+(2*gsd)
                rmin0,rmin1 = y,y-mody+gsd
                cmin0,cmin1 = x,x-modx+gsd
                N9          = np.sum(np.absolute(Pyx-PAN[rmaj0:rmaj1, cmaj0:cmaj1, 0])) + np.sum(np.absolute(Pyx-PAN[rmin0:rmin1, cmin0:cmin1, 0]))
                spoi9       = MSI_US[rmaj0:rmaj1, cmaj0:cmaj1, :] #superpixel of interest
                M9          = np.mean(np.reshape(spoi9, (spoi9.shape[0]*spoi9.shape[1], nb)), axis=0)
                yuv,xuv     = 0.5*(rmaj0+rmaj1), 0.5*(cmaj0+cmaj1)
                exp_spat9      = np.exp(-((y-yuv)**2 + (x-xuv)**2)/sigma_spatial**2)
                del(rmaj0,rmaj1,cmaj0,cmaj1, rmin0,rmin1,cmin0,cmin1, spoi9,yuv,xuv)
            MT9 = np.sum(M9*Tvector)
            
            # Set intensity param-squared as the minimum of the difference factors
            if sigma_inten_adaptive == True:
                N_min = np.min([N1, N2, N3, N4, N5, N6, N7, N8, N9])
                
                if N_min < sigma_inten2_floor:
                    sigma_inten2 = sigma_inten2_floor
                else:
                    sigma_inten2 = N_min
            else:
                sigma_inten2 = 1

            #Mean spectral angle differences
            sacos1 = calc_spectral_angle_cosine(M5, M1)
            sacos2 = calc_spectral_angle_cosine(M5, M2)
            sacos3 = calc_spectral_angle_cosine(M5, M3)
            sacos4 = calc_spectral_angle_cosine(M5, M4)
            sacos5 = 1
            sacos6 = calc_spectral_angle_cosine(M5, M6)
            sacos7 = calc_spectral_angle_cosine(M5, M7)
            sacos8 = calc_spectral_angle_cosine(M5, M8)
            sacos9 = calc_spectral_angle_cosine(M5, M9)
            
            if sigma_sacos == True:
                sacoslist   = [sacos1, sacos2, sacos3, sacos4, sacos6, sacos7, sacos8, sacos9]
                sigma_sacos = np.max([i for i in sacoslist if i != 0])
                #sigma_sacos = np.min([i for i in sacoslist if i != 0])
            else:
                sigma_sacos,sacos1,sacos2,sacos3,sacos4,sacos5,sacos6,sacos7,sacos8,sacos9 = 1,0,0,0,0,0,0,0,0,0
                
            #Calculate spectral angle gaussians
            exp_spang1 = np.exp(-sacos1**2/sigma_sacos**2)
            exp_spang2 = np.exp(-sacos2**2/sigma_sacos**2)
            exp_spang3 = np.exp(-sacos3**2/sigma_sacos**2)
            exp_spang4 = np.exp(-sacos4**2/sigma_sacos**2)
            exp_spang5 = np.exp(-sacos5**2/sigma_sacos**2)
            exp_spang6 = np.exp(-sacos6**2/sigma_sacos**2)
            exp_spang7 = np.exp(-sacos7**2/sigma_sacos**2)
            exp_spang8 = np.exp(-sacos8**2/sigma_sacos**2)
            exp_spang9 = np.exp(-sacos9**2/sigma_sacos**2)
                    
            # Calculate intensity gaussians
            #if sigma_inten2>1: print('sigma_inten2: ',sigma_inten2)
            exp_inten1 = np.exp(-N1/sigma_inten2)
            exp_inten2 = np.exp(-N2/sigma_inten2)
            exp_inten3 = np.exp(-N3/sigma_inten2)
            exp_inten4 = np.exp(-N4/sigma_inten2)
            exp_inten5 = np.exp(-N5/sigma_inten2)
            exp_inten6 = np.exp(-N6/sigma_inten2)
            exp_inten7 = np.exp(-N7/sigma_inten2)
            exp_inten8 = np.exp(-N8/sigma_inten2)
            exp_inten9 = np.exp(-N9/sigma_inten2)
            
            #Calculate normalization factor k_xy
            p1 = exp_inten1*exp_spat1*exp_spang1*MT1
            p2 = exp_inten2*exp_spat2*exp_spang1*MT2
            p3 = exp_inten3*exp_spat3*exp_spang1*MT3
            p4 = exp_inten4*exp_spat4*exp_spang1*MT4
            p5 = exp_inten5*exp_spat5*exp_spang1*MT5
            p6 = exp_inten6*exp_spat6*exp_spang1*MT6
            p7 = exp_inten7*exp_spat7*exp_spang1*MT7
            p8 = exp_inten8*exp_spat8*exp_spang1*MT8
            p9 = exp_inten9*exp_spat9*exp_spang1*MT9
            k_xy = float((1/PAN[y, x, :])*(p1+p2+p3+p4+p5+p6+p7+p8+p9))
            
            # Calculate HM(xy)
            q1 = exp_inten1*exp_spat1*exp_spang1*M1  # q1.shape:  (8,)
            q2 = exp_inten2*exp_spat2*exp_spang2*M2
            q3 = exp_inten3*exp_spat3*exp_spang3*M3
            q4 = exp_inten4*exp_spat4*exp_spang4*M4
            q5 = exp_inten5*exp_spat5*exp_spang5*M5
            q6 = exp_inten6*exp_spat6*exp_spang6*M6
            q7 = exp_inten7*exp_spat7*exp_spang7*M7
            q8 = exp_inten8*exp_spat8*exp_spang8*M8
            q9 = exp_inten9*exp_spat9*exp_spang9*M9

            sharp_spectrum           = (1/k_xy)*(q1+q2+q3+q4+q5+q6+q7+q8+q9)
            HRMSI_NNDIFFUSE[y, x, :] = sharp_spectrum.ravel()
            del(p1, p2, p3, p4, p5, p6, p7, p8, p9, q1, q2, q3, q4, q5, q6, q7, q8, q9, k_xy)
            
    loop_timer1 = time.time()
    loop_total_time = loop_timer1-loop_timer0
    print('loop_total_time: \n', loop_total_time)
    return(HRMSI_NNDIFFUSE)

def nndiffuse_lrhsi_hrmsi_fusion(LRHSI, HRMSI, lambdas_LRHSI, wlength_bounds_HRMSI, version='current'):
    NR, NC, NB_HRMSI   = HRMSI.shape
    nr, nc, nb         = LRHSI.shape
    npix               = nr*nc
    
    #Construct LRHSI_edgeMap using ScharrEdge detector (or Laplacian, etc.)
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
    lrhsi_hrmsi_idx_pairs = np.array(lrhsi_hrmsi_idx_pairs)
    
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
                
            #Choose inside band with highest correlation
            best_inside_lrhsi_band_idx = inside_lrhsi_bands_idxs[np.argmax(coefvals)]
            #WORKING HERE
            hrmsi_idx = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 0]==best_inside_lrhsi_band_idx][:, 1]
            newpair   = np.array([i, hrmsi_idx], dtype=object).reshape((1,-1))
            #print('newpair.shape: ', newpair.shape)
            lrhsi_hrmsi_idx_pairs = np.concatenate((lrhsi_hrmsi_idx_pairs, newpair), axis=0)
    
    #Sort by HRMSI band idx number
    lrhsi_hrmsi_idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1].argsort()]
    
    HRHSI_NNDIFFUSE = np.zeros((NR, NC, 1))
    for i in range(NB_HRMSI):
        idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1]==i]
        if len(idx_pairs)==0:
            continue
        else:
            print('\nFusing group '+str(i)+'.....')
            msi = np.copy(LRHSI[:,:,tuple(idx_pairs[:, 0])])
            PAN = np.expand_dims(np.copy(HRMSI[:, :, i]), axis=2)
            
            msi = msi/10000
            PAN = PAN/10000
            
            if version=='current':
                fused_group     = nndiffuse_pansharpen(PAN, msi, sigma_inten_adaptive=True, sigma_sacos=True)
            elif version=='old':
                fused_group     = nndiffuse_pansharpen(PAN, msi, sigma_inten_adaptive=False, sigma_sacos=False)
            
            HRHSI_NNDIFFUSE = np.concatenate((HRHSI_NNDIFFUSE, fused_group), axis=2)
    
    HRHSI_NNDIFFUSE = HRHSI_NNDIFFUSE[:, :, 1:]
    
    #Re-sort by lrhsi_hrmsi_idx_pairs by HRMSI band idx number
    correct_idx_order     = np.arange(len(lrhsi_hrmsi_idx_pairs)).reshape((-1,1))
    lrhsi_hrmsi_idx_pairs = np.concatenate((correct_idx_order, lrhsi_hrmsi_idx_pairs), axis=1)
    lrhsi_hrmsi_idx_pairs = lrhsi_hrmsi_idx_pairs[lrhsi_hrmsi_idx_pairs[:, 1].argsort()]
    correct_idx_order     = tuple(lrhsi_hrmsi_idx_pairs[:, 0])
    
    HRHSI_NNDIFFUSE       = HRHSI_NNDIFFUSE[:, :, correct_idx_order]
    
    #Return fused image in REFLECTANCE DOMAIN (0,1)
    return(HRHSI_NNDIFFUSE)
