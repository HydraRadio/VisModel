
import numpy as np
import copy

class BaseGainModel(object):
    
    def __init__(self, uvd, default_params=None):
        """
        Per-antenna complex gain model.
        
        Parameters
        ----------
        uvd : UVData object
            UVData object used to define metadata etc.
        
        default_params : array_like, optional
            Default parameters for the gains. These will be used for all 
            antennas that do not have gain model parameters explicitly set.
        """
        # Collect information about the shape of the data
        # (the order of these lists is used to determine parameter order)
        self.freqs = np.unique(uvd.freq_array)
        self.times = np.unique(uvd.time_array)
        self.antpairs = sorted(uvd.get_antpairs())
        self.ants = sorted(uvd.get_ants())
        self.pols = sorted(uvd.get_pols())
        
        # Set-up empty parameter dictionary
        self.params = {}
        for pol in self.pols:
            self.params[pol] = {}
            for ant in self.ants:
                self.params[pol][ant] = None
        
        # Store default params
        self.default_params = default_params
        
        # Empty list of parameter names
        self.paramnames = []
    
    
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
                self.params[pol][ant] = params[pol][ant].copy()
    
    
    def model(self, freqs, times, params=None):
        """
        Complex gain model, as a function of frequency, time, and a set of 
        parameters.
        
        The model function should have a well-defined default behaviour if 
        ``params=None``.
        
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
        
        # Use built-in parameters, or override
        if params is None:
            params = self.params
        
        # Get frequencies and times
        freqs = np.unique(uvd.freq_array)
        times = np.unique(uvd.time_array)
        
        # Loop over known antennas and polarisations and compute the gain model
        gains = {}
        for pol in self.pols:
            for ant in self.ants:
                gains[(ant, pol)] = self.model(freqs, times, params[pol][ant])
        
        # Loop over antenna pairs and polarisations in the UVData object
        for ant1, ant2, pol in uvd.get_antpairpols():
            assert ant1 in self.ants, "Unexpected antenna %d in uvd_in" % ant1
            assert ant2 in self.ants, "Unexpected antenna %d in uvd_in" % ant2
            assert pol in self.pols, "Unexpected polarization %d in uvd_in" % pol
            
            # Get gains
            g1 = gains[(ant1, pol)]
            g2 = gains[(ant2, pol)]
            
            # Find correct elements of the data array
            # uvd.data_array shape: (Nblts, 1, Nfreqs, Npols)
            idxs = uvd.antpair2ind(ant1, ant2)  # blts index
            ipol = list(self.pols).index(pol)  # polarization index
            
            # Explicitly check the order of the times
            if check_order:
                assert np.allclose(uvd.time_array[idxs], times), \
                    "Times in the UVData object do not match expected ordering"
            
            # Apply gains
            if mode == 'multiply':
                uvd.data_array[idxs, 0, :, ipol] *= g1 * g2.conj()
            else:
                uvd.data_array[idxs, 0, :, ipol] /= (g1 * g2.conj())
        
        return uvd



class FactorizedFourierGainModel(BaseGainModel):
    
    def __init__(self, uvd, freq_range, time_range, freq_modes, time_modes, 
                 default_params, params=None):
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
        
        default_params : array_like
            Array of parameter values to use as the default for any antennas 
            that do not have parameters explicitly set (see ``params`` below 
            for the format/ordering of the parameter array).
        
        params : dict, optional
            Dict of parameters to assign, of the form `params[pol][ant]`. The 
            parameter vector for each key should have this number of elements:
            
            ``2*num_freq_modes + 2*num_time_modes``
            
            The ordering is the following:
            
            ``[real freq coeffs|imag freq coeffs|real time coeffs|imag time coeffs]``
        """
        # Initialise superclass
        super().__init__(uvd, default_params=default_params)
        
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
        
        # Check that default_params has the right shape
        Nparams = 2 * (self.freq_modes.size + self.time_modes.size)
        assert default_params.shape == (Nparams,), \
            "`default_params` array has the wrong shape"
        
        # Base names for gain parameters (per polarisation and antenna)
        self.paramnames = ["gain_re_cf%03d" % i for i in self.freq_modes] \
                        + ["gain_im_cf%03d" % i for i in self.freq_modes] \
                        + ["gain_re_ct%03d" % i for i in self.time_modes] \
                        + ["gain_im_ct%03d" % i for i in self.time_modes]
    
    
    def parameter_vector(self, coeffs_freq, coeffs_time):
        """
        Build a 1D parameter vector of the form used by the ``model()`` method.
        
        Parameters
        ----------
        coeffs_freq, coeffs_time : array_like
            Arrays of complex Fourier coefficients, for the frequency and time 
            Fourier series respectively. The array elements are taken to 
            correspond to modes in ``self.freq_modes`` and ``self.time_modes``, 
            in the same order.
        
        Returns
        -------
        params : array_like
            Block parameter array (1D, real) of Fourier coefficients, in the 
            form expected by the ``model()`` method.
        """
        assert len(coeff_freq) == self.freq_modes.size, \
            "`coeffs_freq` has length %d, but %d freq. modes expected" \
            % (len(coeff_freq), self.freq_modes.size)
        assert len(coeff_time) == self.time_modes.size, \
            "`coeffs_time` has length %d, but %d time modes expected" \
            % (len(coeff_time), self.time_modes.size)
        
        # Build parameter vector
        Nf = self.freq_modes.size
        Nt = self.time_modes.size
        params = np.zeros(2*Nf + 2*Nt)
        params[:Nf] = np.real(coeffs_freq)
        params[Nf:2*Nf] = np.imag(coeffs_freq)
        params[2*Nf:2*Nf+Nt] = np.real(coeffs_time)
        params[2*Nf+Nt:] = np.imag(coeffs_time)
        return params
        
    
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
            
        params : array_like, optional
            Array of model parameters. The ordering of the array, in blocks, is:
            
            ``[real freq coeffs|imag freq coeffs|real time coeffs|imag time coeffs]``
            
            If ``params=None``, the model will use ``self.default_params``.
        """
        # Define default behaviour (g = 1)
        if params is None:
            params = self.default_params
        
        # Check length of parameter vector
        assert len(params) == 2 * (self.freq_modes.size + self.time_modes.size), \
            "`params` should have 2*num_freq_modes + 2*num_time_modes elements"
        
        # Parse parameter array to get complex frequency and time coeffs
        Nf = self.freq_modes.size
        tparams = params[2*Nf:] # split time params from freq params
        cf = params[:Nf] + 1.j*params[Nf:2*Nf]
        ct = tparams[:self.time_modes.size] + 1.j*tparams[self.time_modes.size:]
        
        # Pre-factors for Fourier exponents
        freq_fac = 1.j * 2.*np.pi * (freqs - self.freq_min) / self.freq_period
        time_fac = 1.j * 2.*np.pi * (times - self.time_min) / self.time_period
        
        # Frequency modes
        gf = 0
        for i, n in enumerate(self.freq_modes):
            gf += cf[i] * np.exp(n * freq_fac)
        
        # Time modes
        gt = 0
        for i, n in enumerate(self.time_modes):
            gt += ct[i] * np.exp(n * time_fac)
        
        # Return total gain model
        return gf[np.newaxis,:] * gt[:,np.newaxis] # (Ntimes, Nfreqs)
        
