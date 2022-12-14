import numpy as np


def padwidth(N, array): #0-padding function
    return ((int((N - array.shape[0]) / 2), int((N - array.shape[0]) / 2)),
                                   (int((N - array.shape[1]) / 2), int((N - array.shape[1]) / 2)))

def ASM_fw(f, cell_spacing, target_plane_dist, res_fac, k):
    ## Forward angular spectrum method
    f = np.kron(f, np.ones((res_fac, res_fac)))
    # Nfft = target_plane_shape[0] # new side length of array
    Nfft = len(f)
    kx = 2*np.pi*(np.arange(-Nfft/2,(Nfft)/2)/((cell_spacing/res_fac)*Nfft)) # kx vector
    kx = np.reshape(kx, (1,len(kx))) # shape correctly
    ky = kx #spacial frequencies
    f_pad = np.pad(f, padwidth(Nfft, np.zeros(f.shape)), # pad to make F the correct size
                          'constant', constant_values = 0)
    F = np.fft.fft2(f_pad) # 2D FT
    F = np.fft.fftshift(F) # Shift to the centre
    ## Propagate forwards; change signs to back-propagate
    H = np.exp(1j*np.lib.scimath.sqrt(k**2 - kx**2 - (ky**2).T)*target_plane_dist) # propagator function
    Gf = F*H # propagating the signal forward in Fourier space
    gf = np.fft.ifft2(np.fft.ifftshift(Gf)) # IFT & shift to return to real space
    return gf

#TODO: f is complex pressure amplitude (A) and phase (phi) in the form A*exp(i*phi)
#cell_spacing is spacing between pixels in this complex pressure field in meters
#target_plane_dist is the distance to the second field we want to propagate to (m)
#keep res_fac at 1 (resolution)
#k is wavenumber (2pi/lambda)

