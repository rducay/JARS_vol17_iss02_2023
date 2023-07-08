import spectral.io.envi as envi

#Gaussian function to construct psf kernel
def gaussian(x, y, u, FWHM):
    s = FWHM/2.355
    A = 1/(2*np.pi*s**2)
    B = ((x-u)/s)**2
    C = ((y-u)/s)**2
    return(A*np.exp(-0.5*(B+C)))

#Open a target spectral signature
import numpy as np
def openTgtSig(tgtName):
    if tgtName=='REDFELT': tgtSpectrum_fnm = 'AVONPM_AllRed_Emmett_micronwavelengths.txt'
    if tgtName=='BLUEFELT': tgtSpectrum_fnm = 'AVONPM_AllBlue_Emmett_micronwavelengths.txt'
    if tgtName=='DARKGREEN': tgtSpectrum_fnm = 'JoeCarrock_DarkGreen_refl_microns.txt'
    if tgtName=='LIGHTGREEN': tgtSpectrum_fnm = 'JoeCarrock_LightGreen_refl_microns.txt'
    if tgtName=='GREEN': tgtSpectrum_fnm = 'JoeCarrock_Green_refl_microns.txt'
    if tgtName=='CHROMITE': tgtSpectrum_fnm = 'chromite.txt'
    if tgtName=='QUARTZ': tgtSpectrum_fnm = 'quartz.txt'
    if tgtName=='TREMOLITE': tgtSpectrum_fnm = 'tremolite.txt'
    tgtSpectrum_fnm = 'F:/TARGET_IMPLANTATION/targetspectra/'+tgtSpectrum_fnm
    return(np.loadtxt(tgtSpectrum_fnm, skiprows=0))

def simHRMSI(refDataCube, refDataCube_lambdas, csv_srfName, HRMSIbands_tuple, HRMSIbands_coWL):
    '''
    Parameters
    ----------
    refDataCube : NumPy array
        Data cube of shape (num_rows, num_cols, num_bands).
        This reference data cube will be band-integrated to simulate the HRMSI.
    refDataCube_lambdas : NumPy array
        Datacube wavelengths in micrometers; shape: (L,)
    csv_srfName : char
        Directory and filename (with .csv extension) of the spectral response file.
        The csv file should have wavelength values on the first column.
        Subsequent columns are band spectral response curves.
        The first row of this file are headings (use skiprows=1).
    HRMSIbands_tuple : tuple
        The band numbers from 'csv_srfName' to use in simulating HRMSI. 
        Python counting starts from 0, so the 0th column on the srf file would be the wavelengths column.
        Thus, the 'HRMSIbands_tuple', should not contain a '0'.
    HRMSIbands_coWL : list
        This is a list of left and right cutoff wavelengths for each HRMSI band

    ***NOTE: To simulate HRPAN, use HRMSIbands_tuple = (1,)
    
    Returns
    -------
    A simulated HRMSI: a numpy array of shape (num_rows, num_cols, num_bands_HRMSI).
    The SRF matrix used to band-integrate the reference data cube and calculate HRMSI.

    '''
    #Number of bands for simulated HRMSI
    M = len(HRMSIbands_tuple)
    
    #Dimensions of reference data cube 'refDataCube'
    R,C,L = refDataCube.shape
    N = R*C #number of pixels
    
    #Extract srf file for srf and wavelength data
    srfFile     = np.loadtxt(csv_srfName, delimiter=',',skiprows=1)
    srf_lambdas = srfFile[:,0]
    srfFile     = srfFile[:,HRMSIbands_tuple]
    
    #Interpolate srf wavelengths to ref data cube wavelengths
    SRF = np.empty((L, M))
    for b in range(M):
        #Specify band wavelength cutoffs
        cutoff_left, cutoff_right = HRMSIbands_coWL[b][0], HRMSIbands_coWL[b][1]
        
        #Interpolate srf over wavelengths of refHSI, then normalize to maxvalue=1
        srfinterp = np.interp(refDataCube_lambdas, srf_lambdas, np.copy(srfFile[:, b]))
        srfinterp = srfinterp/srfinterp.sum()
        
        #Zero-out the srf outside of the cutoff wavelengths
        srfinterp[(refDataCube_lambdas < cutoff_left) & (refDataCube_lambdas > cutoff_right)] = 0
        SRF[:, b] = srfinterp
        
    #Transform refDataCube into a matrix
    refDataCube = refDataCube.reshape((N, L))
    
    #Matrix-multiply to get HRMSI
    HRMSI = np.matmul(refDataCube, SRF) #shape: (N, M)
    HRMSI = HRMSI.reshape((R,C,M))
    return([HRMSI, SRF])
    
def simHRPAN(refDataCube, refDataCube_lambdas, csv_srfName):
    '''
    Parameters
    ----------
    refDataCube : NumPy array
        Reference data cube of shape (num_rows, num_cols, num_bands).
    refDataCube_lambdas : NumPy array
        Datacube wavelengths in micrometers.
    csv_srfName : char
        Directory and filename (with .csv extension) of the spectral response file.
        The csv file should have wavelength values on the first column.
        Subsequent columns are band spectral response curves.

    Returns
    -------
    A simulated HRPAN: a numpy array of shape (num_rows, num_cols).

    '''
    #Extract srf file for srf data
    srfFile    = np.loadtxt(csv_srfName, delimiter=',',skiprows=1)
    srf_lambdas = np.copy(srfFile[:,0])
    srfHRPAN   = np.copy(srfFile[:,1])
    srf        = np.interp(refDataCube_lambdas, srf_lambdas, srfHRPAN)
    
    #Normalize srf to 1 and reshape to row vector
    srf = (srf/np.sum(srf)).reshape((1,-1))
    
    #Simulate HRPAN
    nrows,ncols,nbands = refDataCube.shape
    refDataCube = refDataCube.reshape((nrows*ncols, nbands))
    return((np.sum(srf*refDataCube, axis=1)).reshape((nrows,ncols)))

def simLRHSI(refDataCube, kernelPSF, GSDratio, kernelSize=6):
    #Dimensions: HR and LR
    nrowsHR, ncolsHR, nbands = refDataCube.shape
    nrowsLR, ncolsLR = nrowsHR//GSDratio, ncolsHR//GSDratio
    
    #Construct blurring filter and normalize
    #Fixed at 6 unless a different size is needed
    if kernelPSF == 'gaussian':
        filt  = np.empty((kernelSize,kernelSize,1))
        for m in range(kernelSize):
            for n in range(kernelSize):
                wcenter   = (kernelSize-1)/2
                filt[m,n,:] = gaussian(m,n,wcenter,GSDratio)
        del(m,n)
    if kernelPSF == 'rect4x4':
        filt = np.zeros((kernelSize,kernelSize,1))
        filt[1:-1, 1:-1, :] = 1
    filt = filt/np.sum(filt) #normalization is very important
    
    #Convolve 'filt' with the raw HSI: manual method (using only numpy)
    LRHSI = np.empty((nrowsLR, ncolsLR, nbands))
    pad_refDataCube = np.pad(refDataCube, ((1,1),(1,1),(0,0)), mode='edge')
    for m in range(nrowsLR):
        for n in range(ncolsLR):
            i = 4*m
            j = 4*n
            roi          = np.copy(pad_refDataCube[i:i+kernelSize, j:j+kernelSize, :])
            conv         = roi*np.copy(filt) #shape: (6,6,240)x(6,6,1)==>(6,6,240)
            LRHSI[m,n,:] = np.sum(conv.reshape((kernelSize**2, nbands)), axis=0)
    
    #Reshape and return
    return(LRHSI.reshape((nrowsLR, ncolsLR, nbands)))

#Edge detection using Scharr kernels
def ScharrEdge(img2d):
    from scipy import signal
    #Pad the input image (to avoid edge effects)
    margin_size  =5
    img2d_padded = np.pad(img2d, ((margin_size,margin_size), (margin_size,margin_size)), mode='reflect')
    #Define x and y gradient kernels
    Gx           = np.array([(-3,0,3), (-10,0,10), (-3,0,3)])
    Gy           = np.copy(Gx.T)
    gradx        = signal.convolve2d(img2d_padded, Gx, mode='same', boundary='fill')
    grady        = signal.convolve2d(img2d_padded, Gy, mode='same', boundary='fill')
    grad         = np.sqrt((gradx**2) + (grady**2))
    #Remove pads
    grad         = grad[5:-5, 5:-5]
    return(grad)


def LaplacianEdge(img2d):
    from scipy import signal
    #Pad the input image (to avoid edge effects)
    margin_size      = 5
    img2d_padded     = np.pad(img2d, ((margin_size,margin_size), (margin_size,margin_size)), mode='reflect')
    #Apply Laplacian kernel
    laplacian_kernel = np.array([(0, -1, 0), (-1,4,-1), (0,-1,0)])
    grad             = signal.convolve2d(img2d_padded, laplacian_kernel, mode='same', boundary='fill')
    #Remove pads
    grad             = grad[5:-5, 5:-5]
    return(grad)


def edgeDetectSAM(hsi, patch_size):
    '''
    Create a function called edgeDetectSAM() that takes an HSI input 'hsi' and patch
    size 'patch_size', and returns an edge map of the HSI image
    '''
    hsi         = np.pad(hsi, ((patch_size, patch_size), (patch_size, patch_size), (0,0)), mode='edge')
    nrows,ncols = hsi.shape[0], hsi.shape[1]
    cosMap      = np.zeros((hsi.shape[0], hsi.shape[1]))
    for n in range(patch_size, nrows-patch_size):
        for m in range(patch_size, ncols-patch_size):
            #Define pixel of interest
            pixOI = hsi[n,m,:].reshape((1,-1))
            denom2 = np.sqrt(np.sum(pixOI*pixOI))
            a,b    = int(np.floor(patch_size/2)), int(np.ceil(patch_size/2))
            bkg    = hsi[n-a:n+b, m-a:m+b, :]
            bkg    = bkg.reshape((bkg.shape[0]*bkg.shape[1], -1))
            #Specify the index of the pixel of interest
            idxOI  = int((patch_size*a) + b - 1)
            #Remove pixel of interest
            bkg    = np.delete(bkg, obj=idxOI, axis=0)
            
            numer  = np.sum((pixOI*bkg), axis=1).ravel()
            denom1 = np.sqrt(np.sum((bkg*bkg), axis=1))
            denom  = (denom1*denom2).ravel()
            cosMap[n,m] = np.mean(numer/denom)
    edgeMap = cosMap[patch_size:nrows-patch_size, patch_size:ncols-patch_size]
    return(edgeMap)
    
#Synthesize LR and HR images using WorldView-2 for band integration
def simulate_LRHSI_HRMSI(path_refHSI, path_LRHSI, path_HRMSI, path_srf_wv2, refHSI_lambdas):
    print("INPUT: refl(%)x100 (maximum value is 10000)")
    refHSI   = np.array(envi.open(path_refHSI+'.hdr', path_refHSI+'.img').load())
    refHSI   = refHSI/10000
    print('\nrefHSI.max(): ', refHSI.max())
    print('refHSI.mean(): ', refHSI.mean())
    print('refHSI.min(): ', refHSI.min())
    NR,NC,NB = refHSI.shape
    NPIX     = NR*NC

    #Simulate LRHSI
    LRHSI    = simLRHSI(np.copy(refHSI), 'gaussian', 4)
    nr,nc,nb = LRHSI.shape
    npix     = nr*nc
    print('\nLRHSI.max(): ', LRHSI.max())
    print('LRHSI.mean(): ', LRHSI.mean())
    print('LRHSI.min(): ', LRHSI.min())
    envi.save_image(path_LRHSI+'.hdr', 10000*LRHSI, force=True)
    
    #Simulate HRMSI via band integration with a given HRMSI srf curve (WorldView-2)
    refimg      = np.copy(np.reshape(refHSI, (NPIX, NB)))
    srf_wv2     = np.loadtxt(path_srf_wv2, delimiter=',')
    srfdatacols = srf_wv2.shape[1]

    #Construct R matrix (reflectance curve to band-integrate the reference)
    R               = np.copy(refHSI_lambdas).reshape((-1,1))
    for col in range(2, srfdatacols):
        srfcurve = np.interp(R[:,0], srf_wv2[:,0], srf_wv2[:,col]).reshape((-1,1))
        if srfcurve.sum() != 0: srfcurve = srfcurve/srfcurve.sum()
        R = np.concatenate((R, srfcurve), axis=1)
    R = R[:,1:].T #shape: (8,num_of_lrhsi_bands)
    
    #Construct HRMSI
    HRMSI = np.dot(refimg, R.T)
    HRMSI = np.reshape(HRMSI, (NR, NC, HRMSI.shape[1]))
    envi.save_image(path_HRMSI+'.hdr', 10000*HRMSI, force=True)
    print('\nHRMSI.max(): ', HRMSI.max())
    print('HRMSI.mean(): ', HRMSI.mean())
    print('HRMSI.min(): ', HRMSI.min())
    return([refHSI, LRHSI, HRMSI, refHSI_lambdas])
    
    
    
    
    
    
    
    
    
    
    
    
