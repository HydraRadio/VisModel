
import numpy as np
from pyuvdata import UVData
from .vismodel import VisModel


class VisLike(object):
    """
    Manage likelihood calculations for a set of visibility data.
    """
    
    def __init__(self, data, model, var):
        """
        Object to handle likelihood evaluation.
        
        Parameters
        ----------
        data : UVData object
            Object containing data.
        
        model : VisModel object
            Object containing model spec.
        
        var : float
            Noise variance. For now, this is just a constant float.
        """
        assert isinstance(data, UVData), "data must be an instance of UVData"
        assert isinstance(model, VisModel), "model must be an instance of VisModel"
        self.data = data
        self.model = model
        self.var = var
        
        # Validate shape of data and model
        self.validate()
    
    
    def prior_cube(self, cube, init_vals):
        """
        Return prior in the form of a cube for ultranest.
        """
        trans_params = np.empty_like(cube)
        
        pnames = self.model.param_names()
        for i, pn in enumerate(pnames):
            
            # Antenna positions
            if 'antpos_dx' in pn or 'antpos_dy' in pn or 'antpos_dz' in pn:
                vmin = init_vals[i] - 0.02
                vmax = init_vals[i] + 0.02
                trans_params[i] = vmin + (vmax - vmin) * cube[i] # Uniform distribution
            
            # Beam parameters
            if 'mainlobe_scale' in pn:
                vmin = 0.95
                vmax = 1.05
                trans_params[i] = vmin + (vmax - vmin) * cube[i] # Uniform distribution
            
        return trans_params
    
    
    def validate(self):
        """
        Check that data and model have the same shape.
        """
        assert self.model.uvd.data_array.shape == self.data.data_array.shape, \
            "Model and data have different shape"
    
    
    def logl(self, params, apply_gains=False):
        """
        Calculate the log-likelihood for a set of parameters.
        
        Parameters
        ----------
        params : array_like
            Array of parameters to calculate the log-likelihood for. The array 
            must be in the order specified by `self.model.param_names()`.
        
        apply_gains : bool, optional
            Whether the gain model should be applied to the model visibilities 
            or not. Set to False if the data are calibrated and you are not 
            sampling gain parameters. Default: False.
        """
        # Calculate model visibilities for current set of parameters
        model = self.model.model(params)
        
        # Pre-fetch freqs and lsts arrays
        freqs = np.unique(self.data.freq_array)
        lsts = np.unique(self.data.lst_array)
        
        # Loop over available data
        logl = 0
        for key, d in self.data.antpairpol_iter():
            
            # Get flags and model
            f = ~self.data.get_flags(key)
            mdl = model.get_data(key)
            
            # Apply gains
            g1, g2 = 1., 1.
            if apply_gains:
                ant1, ant2, _ = key
                g1 = self.model.gains_for_antenna(ant1, freqs=freqs, lsts=lsts)
                g2 = self.model.gains_for_antenna(ant2, freqs=freqs, lsts=lsts)
            
            # Calculate diff. between model and data, and no. degrees of freedom
            delta = d - (g1 * np.conj(g2) * mdl)
            Ndof = 2. * np.sum(~f) # factor of 2 for real + imag
            
            # Calculate log-likelihood
            _logl = -0.5 * f*(delta.real**2. + delta.imag**2.) \
                    / self.var \
                    - 0.5 * np.log(2.*np.pi*self.var) * (Ndof)
            logl += np.sum(_logl)
        return logl
    
    
    def logl_direct(self, params):
        """
        Calculate the log-likelihood by directly comparing the `data_array` for 
        the model and data UVData objects.
        """
        model = self.model.model(params)
        flags = ~self.data.flag_array
        logl = -0.5 * flags*np.abs(self.data.data_array - model.data_array)**2. \
             / self.var \
             - 0.5 * np.log(2.*np.pi*self.var) * self.data.data_array.size
        return np.sum(logl)
        

