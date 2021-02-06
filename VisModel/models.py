
import numpy as np

def bandpass_only_gains(freqs, lsts, params):
    """
    Complex gain model that returns a constant-in-time bandpass factor.
    """
    # Default behaviour: identity
    if params is None:
        return 1. + 0.j
    
    # One parameter per freq. channel
    bandpass_re = params[:freqs.size]
    bandpass_im = params[freqs.size:]
    
    # Return (Nfreqs, 1) array
    return np.atleast_2d(bandpass_re + 1.j*bandpass_im)
