
import numpy as np
from pyuvdata import UVData

from hera_sim.visibilities import VisCPU
try:
    import healvis
except:
    print("healvis import failed; diffuse mode unavailable")
from hera_sim.beams import PolyBeam, PerturbedPolyBeam
from .gains import BaseGainModel
import time, copy, sys


HEALVIS_OPTS_DEFAULT = {
    'obs_latitude':     -30.7215277777,
    'obs_longitude':    21.4283055554,
    'obs_height':       1073.,
    'nside':            64,
    'beam_pol':         'XX', 
    'Nprocs':           1,
}


class VisModel(object):
    
    def __init__(self, uvd_init, 
                 ptsrc_ra_dec, ptsrc_flux,
                 beam_model=PerturbedPolyBeam, 
                 gain_model=None,
                 default_beam_params={},
                 free_params_antpos=[], free_params_beam=[], 
                 free_params_ptsrc=[], free_params_gains=[],
                 free_beams=[], free_ants=[], free_ants_gains=[], 
                 free_ptsrcs=[], 
                 extra_opts_viscpu={},
                 healvis_opts=HEALVIS_OPTS_DEFAULT,
                 include_diffuse=False, verbose=False, comm=None):
        """
        Construct a model for a set of visibility data from a series 
        of data model components. This object handles caching of 
        results to avoid expensive recomputations where possible.
        
        Parameters
        ----------
        uvd_init : UVData object
            UVData object with the correct data shape (number of frequencies, 
            times, polarisations, antennas).
        
        ptsrc_ra_dec, ptsrc_flux : array_like
            Point source position and flux arrays.
        
        beam_model : UVBeam, optional
            Beam model class to be used for the primary beams.
            Default: PerturbedPolyBeam
        
        gain_model : GainModel class, optional
            Class that takes in gain model parameters, frequencies, and LSTs 
            for a given antenna and outputs gain values. Must be a subclass of 
            ``BaseGainModel``. Default: None (creates a new instance of 
            ``BaseGainModel``).
        
        default_beam_params : dict
            Default values of all beam model parameters.
        
        free_params_antpos, free_params_beam, free_params_ptsrc, free_params_gains : lists of str
            Ordered lists of names of parameters to be varied. The ordering 
            maps to the parameter vector.
        
        free_beams, free_ants, free_ants_gains, free_ptsrcs : list of int
            Ordered list of which beams/antennas/point sources are to have 
            their parameters varied. The ordering maps to the parameter vector.
        
        extra_opts_viscpu : dict, optional
            Extra kwargs to pass to the VisCPU Simulate() class. Default: {}.
        
        healvis_opts : dict, optional
            Dictionary of settings for healvis. Default: HEALVIS_OPTS_DEFAULT
        
        include_diffuse : bool, optional
            Whether to include a diffuse model in the visibility model. This 
            uses healvis and GSM as the model. Default: False.
        
        verbose : bool, optional
            Whether to print debug messages. Default: True.
        
        comm : MPI communicator, optional
            MPI communicator to be passed to simulator. Default: None.
        """
        # Set basic parameters
        assert isinstance(uvd_init, UVData), \
            "uvd_init must be a valid UVData object"
        self.verbose = verbose
        
        # Store basic data object
        self.uvd = copy.deepcopy(uvd_init)
        
        # General settings
        self._freqs = np.unique(self.uvd.freq_array)
        self._times = np.unique(self.uvd.time_array)
        self._ant_index = self.uvd.get_ants() # only data antennas
        self._pols = self.uvd.get_pols() # not currently used
        
        # Extra precision options for VisCPU
        self.extra_opts_viscpu = extra_opts_viscpu
        
        # Initial antenna positions (for antennas with free positions)
        self.antpos_initial = {}
        for ant in free_ants:
            pos = self.uvd.antenna_positions[np.where(self.uvd.antenna_numbers == ant)]
            self.antpos_initial[ant] = pos.flatten()
        
        # Set beam type
        self.BeamClass = beam_model
        self.beams = [[] for i in range(self.uvd.Nants_data)]
        self.default_beam_params = default_beam_params
        
        # Initial gain models for all antennas with free gains
        # (gain_model will be used to provide default gain params for other ants)
        if gain_model is None:
            gain_model = BaseGainModel(self.uvd)
        assert isinstance(gain_model, BaseGainModel), \
            "gain_model must be a subclass of BaseGainModel"
        self.gain_model = gain_model
        self.gain_params = {pol: {} for pol in self._pols} # per polarization
        
        # Check that requested free gain parameters exist in the GainModel 
        # object, and find their indices in the parameter vector
        self.free_params_gains_idxs = []
        for p in free_params_gains:
            assert p in self.gain_model.paramnames, \
                "Parameter '%s' not found in gain model specification. " \
                "Available parameters are: %s" % (p, self.gain_model.paramnames)
            
            # Get index of this parameter
            self.free_params_gains_idxs.append(self.gain_model.paramnames.index(p))
        self.free_params_gains_idxs = np.array(self.free_params_gains_idxs)
        
        # Store all gain parameters for free antennas, even if not varied
        pol = self._pols[0] # Use the first polarization in the array for now
        for ant in free_ants_gains:
            self.gain_params[pol][ant] = self.gain_model.params[pol][ant]
        
        # Point source catalogue
        self.ptsrc_ra_dec = ptsrc_ra_dec.copy()
        self.ptsrc_flux = ptsrc_flux.copy()
        self.ptsrc_delta_radec = np.zeros(self.ptsrc_ra_dec.shape)
        self.ptsrc_flux_factor = np.ones(self.ptsrc_ra_dec.shape[0])
        
        # Mappings between param vector and param names
        self.free_params_antpos = free_params_antpos
        self.free_params_beam = free_params_beam
        self.free_params_ptsrc = free_params_ptsrc
        self.free_params_gains = free_params_gains
        self.free_beams = free_beams
        self.free_ants = free_ants
        self.free_ants_gains = free_ants_gains
        self.free_ptsrcs = free_ptsrcs
        
        # Set up diffuse model
        self.include_diffuse = include_diffuse
        self.healvis_opts = healvis_opts
        self.diffuse_vis = 0
        healvis_enabled = True
        try:
            healvis.__file__
        except:
            healvis_enabled = False
            if self.include_diffuse:
                raise ImportError("`include_diffuse` set to True, but failed "
                                  "to import healvis")
        
        # Create healvis baseline spec
        if self.include_diffuse:
            
            # Set up healvis array model
            self._set_antpos_healvis()
            
            # Create GSM sky model
            self.diffuse_gsm = healvis.sky_model.construct_skymodel(
                                              'gsm', 
                                              freqs=self._freqs, 
                                              ref_chan=0,
                                              Nside=self.healvis_opts['nside'] )
        
        # MPI handling
        self.comm = comm
        
        # Construct beam model with new parameter values
        for i in range(self.uvd.Nants_data):
            self.beams[i] = self.BeamClass(**self.default_beam_params)
        
        # Model visibility:
        #   Antenna positions
        #   Primary beam models
        #   Sky model: point sources
        #   Sky model: Gaussian sources
        #   Sky model: diffuse
    
    
    def set_antpos(self, ant, params):
        """
        Set position parameters of named antenna.
        
        Parameters
        ----------
        ant : int
            Antenna ID.
        
        params : array_like
            Vector of parameters. Names and ordering are defined 
            in `self.free_params_antpos`.
        """
        # Extract antenna position parameters
        vec = np.zeros(3)
        j = 0
        for i, p in enumerate(['antpos_dx', 'antpos_dy', 'antpos_dz']):
            if p in self.free_params_antpos:
                vec[i] = params[j]
                j += 1
        
        # Update antenna positions
        self.uvd.antenna_positions[ant] = self.antpos_initial[ant] + vec
        
        # Update healvis baseline spec if needed
        if self.include_diffuse:
            self._set_antpos_healvis()
    
    
    def _set_antpos_healvis(self):
        """
        Update healvis baselines when position of antennas has changed.
        Positions are taken from `self.uvd.antenna_positions`.
        """
        # Construct antenna list
        pos_array = []
        ants = self._ant_index
        for ant in ants:
            pos = self.uvd.antenna_positions[
                                np.where(self.uvd.antenna_numbers == ant) ]
            pos_array.append(pos)
        
        # Construct baseline list
        healvis_bls = []
        for i in range(len(ants)):
            for j in range(i, len(ants)):
                _bl = healvis.observatory.Baseline(pos_array[i], 
                                                   pos_array[j], 
                                                   ants[i], 
                                                   ants[j])
                healvis_bls.append(_bl)
        self.healvis_bls = healvis_bls
        
        # Set times
        times = np.unique(self.uvd.time_array)
        Ntimes = times.size

        # Create Observatory object
        fov = 360. # deg
        healvis_opts = self.healvis_opts
        obs = healvis.observatory.Observatory(healvis_opts['obs_latitude'], 
                                              healvis_opts['obs_longitude'], 
                                              healvis_opts['obs_height'],
                                              array=self.healvis_bls, 
                                              freqs=self._freqs)
        obs.set_pointings(times)
        obs.set_fov(fov)
        
        # Update beam list
        obs.set_beam(self.beams)
        
        self.healvis_observatory = obs
    
    
    def set_beam(self, ant, params):
        """
        Set the parameters of a beam.
        
        Parameters
        ----------
        ant : int
            Antenna ID.
            
        params : array_like
            Vector of parameters. Names and ordering are defined 
            in `self.free_params_beams`.
        """
        assert len(params) == len(self.free_params_beam)
        
        # Copy default parameter values
        param_dict = copy.copy(self.default_beam_params)
        
        # Update parameter values that have changed
        for i, pname in enumerate(self.free_params_beam):
            param_dict[pname] = params[i]
        
        # Construct beam model with new parameter values
        idx = np.where(self._ant_index == ant)[0][0]
        self.beams[idx] = self.BeamClass(**param_dict)
        # First arg is: perturb_coeffs=[0.,]
        
        # Update beams for diffuse model (healvis) if needed
        if self.include_diffuse:
            self.healvis_observatory.set_beam(self.beams)
    
    
    def set_ptsrc_params(self, i, params):
        """
        Set the parameters of a point source.
        
        Parameters
        ----------
        i : int
            Point source index in the catalogue.
            
        params : array_like
            Vector of parameters. Names and ordering are defined 
            in `self.free_params_ptsrcs`.
        """
        # Set point source parameters if specified
        if 'delta_ra' in self.free_params_ptsrc:
            self.ptsrc_delta_radec[i][0] = \
                             params[self.free_params_ptsrc.index('delta_ra')]
        if 'delta_dec' in self.free_params_ptsrc:
            self.ptsrc_delta_radec[i][1] = \
                             params[self.free_params_ptsrc.index('delta_dec')]
        if 'flux_factor' in self.free_params_ptsrc:
            self.ptsrc_flux_factor[i] = \
                             params[self.free_params_ptsrc.index('flux_factor')]
    
    
    def set_gain_params(self, ant, params, pol=None):
        """
        Set the free parameters of the complex gain model for an antenna. This 
        only updates the internal set of parameters in the VisModel object, not 
        the parameters in the GainModel (``self.gain_model``).
        
        Parameters
        ----------
        ant : int
            Antenna ID.
            
        params : array_like
            Vector of free gain parameters. Names and ordering are defined in 
            `self.free_params_gains`.
            
            Note that the parameter arrays in ``self.gain_params`` are full 
            sets of parameters for the gain model, whereas ``params`` is an 
            (ordered) array of free gain parameters only.
        
        pol : str, optional
            Which polarization to update. Currently not needed; the first 
            available polarization will be used by default.
        """
        if pol is None:
            pol = self._pols[0] # FIXME
        
        if ant not in self.gain_params[pol].keys():
            raise KeyError("Antenna '%s' not found." % ant)
        
        # If this is the first time params have been set explicitly for this 
        # antenna, start out with the default parameter set
        if self.gain_params[pol][ant] is None:
            self.gain_params[pol][ant] = self.gain_model.default_params.copy()
        
        # Set only free parameters
        self.gain_params[pol][ant][self.free_params_gains_idxs] = params.copy()
    
    
    def gains_for_antenna(self, ant, freqs=None, lsts=None, pol=None):
        """
        Return the complex gains as a 2D array in frequency and time for a 
        given antenna.
        
        Parameters
        ----------
        ant : int
            Antenna ID.
        
        freqs, lsts : array_like, optional
            1D arrays of frequencies and times, defining the grid that the 
            gains will be evaluated on. If None, these will be taken from the 
            `self.uvd` UVData object.
            Default: None.
        
        pol : str, optional
            Which polarization to return the gains for. Currently not needed; 
            the first available polarization will be used by default.
        
        Returns
        -------
        gains : array_like, complex
            2D complex gain array.
        """
        if pol is None:
            pol = self._pols[0] # FIXME
        
        if ant not in self.gain_params.keys():
            raise KeyError("Antenna '%s' not found." % ant)
        
        # Get freqs/lsts arrays if not specified
        if freqs is None:
            freqs = np.unique(self.uvd.freq_array)
        if lsts is None:
            lsts = np.unique(self.uvd.lst_array)
        
        # Evaluate model and return
        return self.gain_model.model(freqs, lsts, self.gain_params[pol][ant])
    
    
    def apply_gains(self, uvd):
        """
        Use the attached ``gain_model`` to apply gains to the visibilities. 
        This uses stored gain parameters managed by ``self.set_gain_params``, 
        as well as the default gain parameters stored in the ``gain_model`` 
        object.
        
        Parameters
        ----------
        uvd : UVData object
            UVData object to which gains will be applied.
        """
        # Copy default parameter set from gain_model
        p = copy.deepcopy(self.gain_model.params)
        
        # Update with input parameters (self.gain_params contains the full set 
        # of gain parameters needed, not just the free ones)
        for pol in self.gain_params.keys():
            for ant in self.gain_params[pol].keys():
                p[pol][ant] = self.gain_params[pol][ant].copy()
        
        # Apply gains
        # N.B. `check_order=True` is more careful but may be slower
        _uvd = self.gain_model.apply_gains(uvd, 
                                           params=p, 
                                           mode='multiply', 
                                           inplace=False, 
                                           check_order=True)
        return _uvd
    
    
    def param_names(self):
        """
        Return list of parameter names in order.
        """
        pnames = []
        
        # Antenna position parameters
        for ant in self.free_ants:
            for p in self.free_params_antpos:
                pnames.append("%s_%03d" % (p, ant))
        
        # Beam parameters
        for ant in self.free_beams:
            for p in self.free_params_beam:
                pnames.append("%s_%03d" % (p, ant))
        
        # Gain parameters
        pol = self._pols[0] # FIXME: Only takes the first polarization for now
        for ant in self.free_ants_gains:
            for p in self.free_params_gains:
                pnames.append("%s_%s_%03d" % (p, pol, ant))
        
        # Point source parameters
        for i in self.free_ptsrcs:
            for p in self.free_params_ptsrc:
                pnames.append("%s_%06d" % (p, i))
        
        return pnames
    
    
    def simulate_point_sources(self, comm=None):
        """
        Simulate point source visibilities for the current state of the model.
        
        Parameters
        ----------
        comm : MPI communicator
            MPI communicator for vis_cpu (parallelised by frequency channel).
            Default: None.
        
        Returns
        -------
        uvd : UVData object
            Simulated visibilities.
        """
        # Need to zero the array, which is usually returned by 
        # reference rather than copied
        self.uvd.data_array *= 0.
        
        # Construct new simulator class with updated settings
        simulator = VisCPU(
            uvdata=self.uvd,                          # fixed
            beams=self.beams,                         # varies
            beam_ids=self._ant_index,                 # fixed
            sky_freqs=self._freqs,                    # fixed
            point_source_pos=self.ptsrc_ra_dec + self.ptsrc_delta_radec, # varies
            point_source_flux=self.ptsrc_flux * self.ptsrc_flux_factor,  # varies
            polarized = False,                        # assumes Stokes I for now
            precision=2,                              # fixed
            use_pixel_beams=False,                    # Do not use pixel beams
            bm_pix=1,
            mpi_comm=comm,
            **self.extra_opts_viscpu
        )
        
        # Run simulation
        if self.verbose:
            print("  Beginning simulation")
        tstart = time.time()
        simulator.simulate()
        if self.verbose:
            print("  Simulation took %2.1f sec" % (time.time() - tstart))
    
        #if myid != 0:
        #    # Wait for root worker to finish IO before ending all other worker procs
        #    comm.Barrier()
        #    sys.exit(0)
        
        return simulator.uvdata
    
    
    def simulate_diffuse(self, comm=None):
        """
        Simulate diffuse model visibilities for the current state of the model.
        
        Parameters
        ----------
        comm : MPI communicator
            MPI communicator for vis_cpu (parallelised by frequency channel).
            Default: None.
        
        Returns
        -------
        uvd : UVData object
            Simulated visibilities.
        """
        obs = self.healvis_observatory
        
        # Run simulation
        if self.verbose:
            print("  Beginning simulation")
        tstart = time.time()
        
        # Compute visibilities
        # FIXME: Use vis_cpu for diffuse emission instead
        # FIXME: Need to use polarisation info properly
        gsm_vis, _times, _bls = obs.make_visibilities(
                                        self.diffuse_gsm,
                                        beam_pol=self.healvis_opts['beam_pol'], 
                                        Nprocs=self.healvis_opts['Nprocs'] )
        
        if self.verbose:
            print("  Simulation took %2.1f sec" % (time.time() - tstart))
        
        self.diffuse_vis = gsm_vis
        return gsm_vis, _times, _bls
    
    
    def parameter_vector(self, antpos_params, beam_params, gain_params, 
                         ptsrc_params):
        """
        Construct a global parameter vector from individual parameter vectors 
        for each type of parameter.
        
        Parameters
        ----------
        antpos_params : array_like
            Antenna position parameters, of shape (Nfree_ants, Nparams_ants).
            
        beam_params : array_like
            Beam model parameters, of shape (Nfree_beams, Nparams_beams).
            
        gain_params : array_like
            Gain model parameters, of shape (Nfree_ants_gains, Nparams_gains).
            
        ptsrc_params : array_like
            Point source parameters, of shape (Nfree_ptsrcs, Nparams_ptsrcs).
        
        Returns
        -------
        params : array_like
            Array of parameters in the expected format.
        """
        # Count free parameters and antennas/beams
        Nfree_ants = len(self.free_ants)
        Nfree_ants_gains = len(self.free_ants_gains)
        Nfree_beams = len(self.free_beams)
        Nfree_ptsrcs = len(self.free_ptsrcs)
        Nparams_ants = len(self.free_params_antpos)
        Nparams_beams = len(self.free_params_beam)
        Nparams_gains = len(self.free_params_gains)
        Nparams_ptsrcs = len(self.free_params_ptsrc)
        
        # Check shapes
        assert antpos_params.shape == (Nfree_ants, Nparams_ants)
        assert beam_params.shape == (Nfree_beams, Nparams_beams)
        assert gain_params.shape == (Nfree_ants_gains, Nparams_gains)
        assert ptsrc_params.shape == (Nfree_ptsrcs, Nparams_ptsrcs)
        
        # Initialise empty parameter array
        Ntot = Nfree_ants * Nparams_ants + Nfree_beams * Nparams_beams \
             + Nfree_ants_gains * Nparams_gains + Nfree_ptsrcs * Nparams_ptsrcs
        params = np.zeros(Ntot)
        
        # Insert parameters in blocks
        iants = Nfree_ants * Nparams_ants
        ibeams = Nfree_beams * Nparams_beams
        igains = Nfree_ants_gains * Nparams_gains
        iptsrcs = Nfree_ptsrcs * Nparams_ptsrcs
        params[0:iants] = antpos_params.flatten()
        params[iants:iants+ibeams] = beam_params.flatten()
        params[iants+ibeams:iants+ibeams+igains] = gain_params.flatten()
        params[iants+ibeams+igains:iants+ibeams+igains+iptsrcs] = ptsrc_params.flatten()
        
        return params
    
    
    def model(self, params):
        """
        Calculate model visibilities for a given parameter vector.
        
        Parameters
        ----------
        params : array_like
            Vector of parameter values. The number of parameters 
            and position of each parameter in the vector must 
            follow the specification 
        """
        params = np.array(params)
        
        # Count free parameters and antennas/beams
        Nfree_ants = len(self.free_ants)
        Nfree_ants_gains = len(self.free_ants_gains)
        Nfree_beams = len(self.free_beams)
        Nfree_ptsrcs = len(self.free_ptsrcs)
        Nparams_ants = len(self.free_params_antpos)
        Nparams_beams = len(self.free_params_beam)
        Nparams_gains = len(self.free_params_gains)
        Nparams_ptsrcs = len(self.free_params_ptsrc)
        
        # Extract parameters in blocks and reshape
        iants = Nfree_ants * Nparams_ants
        ibeams = Nfree_beams * Nparams_beams
        igains = Nfree_ants_gains * Nparams_gains
        iptsrcs = Nfree_ptsrcs * Nparams_ptsrcs
        antpos_params = params[0:iants].reshape((Nfree_ants, Nparams_ants))
        beam_params = params[iants:iants+ibeams].reshape((Nfree_beams, Nparams_beams))
        gain_params = params[iants+ibeams:iants+ibeams+igains].reshape(
                                              (Nfree_ants_gains, Nparams_gains))
        ptsrc_params = params[iants+ibeams+igains:iants+ibeams+igains+iptsrcs].reshape(
                                              (Nfree_ptsrcs, Nparams_ptsrcs))
        
        # (1) Antenna positions
        for i, ant in enumerate(self.free_ants):
            self.set_antpos(ant, antpos_params[i])
        
        # (2) Beam parameters
        for i, ant in enumerate(self.free_beams):
            self.set_beam(ant, beam_params[i])
        
        # (3) Gain model parameters (not applied to data yet)
        for i, ant in enumerate(self.free_ants_gains):
            self.set_gain_params(ant, gain_params[i])
        
        # (4) Point source parameters
        for i in self.free_ptsrcs:
            self.set_ptsrc_params(i, ptsrc_params[i])
        
        # Run point source simulation
        _uvd = self.simulate_point_sources()
        
        # Run diffuse model simulation
        if self.include_diffuse:
            vis_gsm, _times, _bls = self.simulate_diffuse()
            
            # Check that ordering of healvis output matches existing uvd object
            antpairs_hvs = [(self.healvis_bls[i].ant1, self.healvis_bls[i].ant2) 
                            for i in _bls]
            antpairs_uvd = [_uvd.baseline_to_antnums(_b) 
                            for _b in _uvd.baseline_array]
            
            assert antpairs_hvs == antpairs_uvd, \
                                     "healvis 'bls' array does not match the " \
                                     "ordering of existing UVData.baseline_array"
            assert np.all(_times == uvd.time_array), \
                                     "healvis 'times' array does not match the " \
                                     "ordering of existing UVData.time_array"
            
            # Add diffuse data to UVData object
            _uvd.data_array[:,:,:,0] += vis_gsm
        
        # Apply gain model
        _uvd = self.apply_gains(_uvd)
        
        # Update internal UVData array and return
        self.uvd = _uvd
        return _uvd

        
