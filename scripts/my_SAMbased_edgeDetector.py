import numpy as np

def edgeDetector_SA(msi, marginsize=1):
    '''
    For each multispectral/hyperspectral pixel of interest, it calculates an 
    'edgeness' score based on the mean spectral angle it makes with its 
    8 surrounding neighbors.
    Returns a map of edgeness scores
    '''
    nr, nc, nb = msi.shape
    msipad     = np.pad(msi, ((marginsize,marginsize+1), (marginsize,marginsize+1), (0,0)), mode='reflect')
    sam_map    = np.zeros((nr,nc))
    for i in range(nr):
        for j in range(nc):
            y, x          = i+marginsize, j+marginsize
            vecoi         = msipad[y, x, :].reshape((1, -1))     #shape: (1,nb)
            vecmag        = np.sqrt(np.sum(vecoi**2))            #shape: (1,)
            bkg           = msipad[y-marginsize:y+marginsize+1, x-marginsize:x+marginsize+1, :] #shape: (nr+(2*marginsize), nc+(2*marginsize), nb)
            np_bkg        = bkg.shape[0]*bkg.shape[1]
            bkg           = np.reshape(bkg, (np_bkg, nb))        #shape: (npix, nb)
            vecdots       = np.dot(vecoi, bkg.T)                 #shape: (npix,)
            bkgmags       = np.sqrt(np.sum(bkg**2, axis=1))      #shape: (npix,)
            #print('np.arccos(vecdots/(vecmag*bkgmags)): ', np.arccos(vecdots/(vecmag*bkgmags)))
            #sam_map[i,j]  = np.mean(np.arccos(vecdots/(vecmag*bkgmags)))    #shape: (1,)
            sam_map[i,j]  = np.arccos(np.mean(vecdots/(vecmag*bkgmags)))    #shape: (1,)
    return(sam_map)