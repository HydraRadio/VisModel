
import numpy as np
import scipy.signal

def delay(vis, inverse=False, taper=None):
    """
    Perform delay transform on visibility data.
    
    ``vis`` must have shape (Nfreqs, Ntimes).
    """
    # Construct taper function
    if taper is not None:
        w = np.array(taper(vis.shape[0]))
    else:
        w = np.ones(vis.shape[0])
    
    # Perform either forward or inverse FFT
    if inverse:
        # Do the inverse FFT on each time sample
        return np.fft.ifft(vis * w[:,np.newaxis], axis=0)
    else:
        # Do the forward FFT on each time sample
        return np.fft.fft(vis * w[:,np.newaxis], axis=0)


def fringe_rate(vis, inverse=False, taper=None):
    """
    Perform fringe rate transform on visibility data.
    
    ``vis`` must have shape (Nfreqs, Ntimes).
    """
    # Construct taper function
    if taper is not None:
        w = np.array(taper(vis.shape[1]))
    else:
        w = np.ones(vis.shape[1])
    
    # Perform either forward or inverse FFT
    if inverse:
        # Do the inverse FFT on each frequency sample
        return np.fft.ifft(vis * w[np.newaxis,:], axis=1)
    else:
        # Do the forward FFT on each frequency sample
        return np.fft.fft(vis * w[np.newaxis,:], axis=1)
