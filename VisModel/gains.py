
import numpy as np
import copy

class BaseGainModel(object):
    
    def __init__(self, uvd):
        """
        Per-antenna complex gain model.
        
        Parameters
        ----------
        uvd : UVData object
            UVData object used to define metadata etc. 
        """
        # Collect information about the shape of the data
        # (the order of these lists is used to determine parameter order)
        self.freqs = np.unique(uvd.freq_array)
        self.times = np.unique(uvd.time_array)
        self.antpairs = sorted(uvd.get_antpairs())
        self.ants = sorted(uvd.ants())
        self.pols = sorted(uvd.get_pols())
        
        # Set-up empty parameter dictionary
        self.params = {}
        for pol in self.pols:
            self.params[pol] = {}
            for ant in self.ants:
                self.params[pol][ant] = None
    
    
    def set_params(self, params):
        """
        Set gain model parameter values per polarization and antenna.
        
        Parameters
        ----------
        params : dict
            Dict of parameters to assign, of the form `params[pol][ant]`.
        """
        for pol in params.keys():
            assert pol in self.pols, \
                "Polarization '%s' not found." % pol
            for ant in params[pol].keys():
                assert ant in self.ants, \
                    "Antenna '%s' not found for polarization '%s'." % (ant, pol)
                self.params[pol][ant] = params[pol][ant]
    
    
    def model(self, freqs, times, params=None):
        """
        Complex gain model, as a function of frequency, time, and a set of 
        parameters.
        
        The model function should have a well-defined behaviour if `params=None`.
        
        Parameters
        ----------
        freqs : array_like
            1D array of frequency values, in Hz.
        
        times : array_like
            1D array of time values.
        """
        # Basic complex gain model (g = 1 = const.)
        g = np.ones((freqs.size, times.size)) + 0.j
        return g
    
    
    def apply_gains(self, uvd_in, params=None, mode='multiply', inplace=False, 
                    check_order=True):
        """
        Apply gains to a UVData object.
        
        Parameters
        ----------
        uvd_in : UVData
            UVData object to apply the gains to.
        
        params : dict, optional
            Dict containing gain model parameters for each polarization 
            and antenna. Structure: `params[pol][ant]`.
        
        mode : str, optional
            How to apply the gains. 'multiply' means that the 
            data will be multiplied by the gains. 'calibrate' 
            means that the data will be divided by the gains.
        
        inplace : bool, optional
            Whether to apply the gains to the input data in-place, or to 
            return a new copy of the object. Default: False.
        
        check_order : bool, optional
            Whether to explicitly check that the ordering of the times in the 
            UVData object's `data_array` matches what is expected.
        """
        # Check inputs
        assert mode in ['multiply', 'calibrate'], \
            "mode must be 'multiply' or 'calibrate'"
        
        # Check whether to apply operation in-place
        if inplace:
            uvd = uvd_in
        else:
            uvd = copy.deepcopy(uvd_in)
        
        # Get frequencies and times
        freqs = np.unique(uvd.freq_array)
        times = np.unique(uvd.time_array)
        
        # Loop over known antennas and polarisations and compute the gain model
        gains = {}
        for pol in self.pols:
            for ant in self.ants:
                gains[(ant, pol)] = self.model(freqs, times, self.params[pol][ant])
        
        # Loop over antenna pairs and polarisations in the UVData object
        for ant1, ant2, pol in uvd.get_antpair_pols():
            assert ant1 in self.ants, "Unexpected antenna %d in uvd_in" % ant1
            assert ant2 in self.ants, "Unexpected antenna %d in uvd_in" % ant2
            assert pol in self.pols, "Unexpected polarization %d in uvd_in" % pol
            
            # Get gains
            g1 = gains[(ant1, pol)]
            g2 = gains[(ant2, pol)]
            
            # Find correct elements of the data array
            # uvd.data_array shape: (Nblts, 1, Nfreqs, Npols)
            idxs = uvd.antpair2ind(ant1, ant2)  # blts index
            ipol = np.where(self.pols == pol)[0]  # polarization index
            
            # Explicitly check the order of the times
            if check_order:
                assert np.almost_equal(uvd.time_array[idxs], times), \
                    "Times in the UVData object do not match expected ordering"
            
            # Apply gains
            if mode == 'multiply':
                uvd.data_array[idxs, 0, :, ipol] *= g1 * g2.conj()
            else:
                uvd.data_array[idxs, 0, :, ipol] /= (g1 * g2.conj())
        
        return uvd



class FactorizedFourierGainModel(BaseGainModel):
    
    def __init__(self, uvd, freq_range, time_range, freq_modes, time_modes):
        """
        Per-antenna complex gain model with factorisable complex Fourier 
        series in frequency and time.
        
        Parameters
        ----------
        uvd : UVData object
            UVData object used to define metadata etc.
        
        freq_range, time_range : tuple of float
            The frequencies and times to consider as the minimum and maximum 
            of the range, e.g. `freq_range = (freq_min, freq_max)`. The model 
            is allowed to extend outside this range
        
        freq_modes, time_modes : int or array_like
            If specified as an integer ``n``, the first ``n`` complex Fourier 
            modes will be used to define the model, starting with the zero mode.
            
            If specified as an array, each element gives the order of a Fourier 
            mode to include in the model, e.g. ``freq_modes = [0, 1, 5]`` would 
            include the n=0, 1, and 5 Fourier modes in the model only.
        """
        # Initialise superclass
        super().__init__(uvd)
        
        # Check inputs
        assert len(freq_range) == 2, "freq_range must be a tuple: (freq_min, freq_max)"
        assert len(time_range) == 2, "time_range must be a tuple: (time_min, time_max)"
        assert freq_range[1] > freq_range[0]
        assert time_range[1] > time_range[0]
        
        # Calculate period etc.
        self.freq_min, self.freq_max = freq_range
        self.time_min, self.time_max = time_range
        self.freq_period = self.freq_max - self.freq_min
        self.time_period = self.time_max - self.time_min
        
        # Specify which Fourier modes to include in the model
        if isinstance(freq_modes, (int, np.integer)):
            self.freq_modes = np.arange(freq_modes, dtype=np.integer)
        else:
            self.freq_modes = np.array(freq_modes)
        
        if isinstance(time_modes, (int, np.integer)):
            self.time_modes = np.arange(time_modes, dtype=np.integer)
        else:
            self.time_modes = np.array(time_modes)
        
        
    def model(self, freqs, times, params=None):
        """
        Complex gain model, as a function of frequency, time, and a set of 
        parameters.
        
        Parameters
        ----------
        freqs : array_like
            1D array of frequency values, in Hz.
        
        times : array_like
            1D array of time values.
            
        params : dict, optional
            Dictionary of model parameters per polarisation and per antenna, 
            with structure `params[pol][ant]`. The parameters here are the 
            complex coefficients of the Fourier series, corresponding to each 
            mode in the lists `self.freq_modes` and `self.time_modes`.
        """
        # Factors for 
        freq_fac = 1.j * 2.*np.pi * (freqs - self.freq_min) / self.freq_period
        time_fac = 1.j * 2.*np.pi * (times - self.time_min) / self.time_period
        
        # Frequency modes
        gf = 0
        for i, n in enumerate(self.freq_modes):
            gf += cn * np.exp(n * freq_fac) # FIXME: cn not defined
        
        # Time modes
        gf = 0
        for i, n in enumerate(self.time_modes):
            gt += cn * np.exp(n * time_fac) # FIXME: cn not defined
        
        # Return total gain model
        return gf[np.newaxis,:] * gt[:,np.newaxis] # (Ntimes, Nfreqs)
