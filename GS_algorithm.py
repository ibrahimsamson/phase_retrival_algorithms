import numpy as np
from numpy.fft import fft2, ifft2

def gerchberg_saxton(measured_intensity, target_image, n_iterations):
    # Initialize phase with random values
    phase = np.exp(2j * np.pi * np.random.rand(*measured_intensity.shape))
    
    for i in range(n_iterations):
        # Apply Fourier transform to current phase to obtain amplitude and phase components
        amplitude = np.sqrt(measured_intensity) * np.exp(1j * np.angle(fft2(phase)))
        
        # Apply inverse Fourier transform to obtain new image
        new_image = ifft2(amplitude)
        
        # Replace the phase of the new image with that of the target image
        phase = np.exp(1j * np.angle(fft2(target_image))) * np.exp(1j * np.angle(fft2(new_image)))
        
        # Apply Fourier transform to update amplitude and phase
        amplitude = np.sqrt(measured_intensity) * np.exp(1j * np.angle(fft2(phase)))
        
        # Apply inverse Fourier transform to obtain updated image
        updated_image = ifft2(amplitude)
        
    # Return the final updated image
    return updated_image.real
