
import numpy as np
from pyuvdata import UVData

from hera_sim.visibilities import VisCPU
try:
    import healvis
except:
    print("healvis import failed; diffuse mode unavailable")
from hera_sim.beams import PolyBeam, PerturbedPolyBeam
from .models import bandpass_only_gains
import time, copy, sys


HEALVIS_OPTS_DEFAULT = {
    'obs_latitude':     -30.7215277777,
    'obs_longitude':    21.4283055554,
    'obs_height':       1073.,
    'nside':            64,
}


class VisModel(object):
    
    def __init__(self, uvd_init, 
                 ptsrc_ra_dec, ptsrc_flux,
                 beam_model=PerturbedPolyBeam, 
                 gain_model=bandpass_only_gains,
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
            UVData object with the correct data shape (number of 
            frequencies, times, polarisations, antennas).
        
        ptsrc_ra_dec, ptsrc_flux : array_like
            Point source position and flux arrays.
        
        beam_model : UVBeam, optional
            Beam model class to be used for the primary beams.
            Default: PerturbedPolyBeam
        
        gain_model : func, optional
            Function that takes in gain model parameters, frequencies, 
            and LSTs for a given antenna and outputs gain values. Function 
            must have call signature: `fn(freqs, lsts, params)`, where 
            `freqs` and `lsts` are 1D arrays.
        
        default_beam_params : dict
            Default values of all beam model parameters.
        
        free_params_antpos, free_params_beam, free_params_ptsrc, free_params_gains : lists of str
            Ordered lists of names of parameters to be varied. 
            The ordering maps to the parameter vector.
        
        free_beams, free_ants, free_ants_gains, free_ptsrcs : list of int
            Ordered list of which beams/antennas/point sources 
            are to have their parameters varied. The ordering 
            maps to the parameter vector.
        
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
        
        # Initial gain models for all antennas
        self.gain_model = gain_model
        self.gain_params = {}
        for ant in self.uvd.get_ants():
            self.gain_params[ant] = None
        
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
        
        # Set times
        times = np.unique(self.uvd.time_array)
        Ntimes = times.size

        # Create Observatory object
        fov = 360. # deg
        healvis_opts = self.healvis_opts
        obs = healvis.observatory.Observatory(healvis_opts['obs_latitude'], 
                                              healvis_opts['obs_longitude'], 
                                              healvis_opts['obs_height'],
                                              array=healvis_bls, 
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
    
    
    def set_gain_params(self, ant, params):
        """
        Set the parameters of the complex gain model for an antenna.
        
        Parameters
        ----------
        ant : int
            Antenna ID.
            
        params : array_like
            Vector of parameters. Names and ordering are defined in 
            `self.free_params_gains`.
        """
        if ant not in self.gain_params.keys():
            raise KeyError("Antenna '%s' not found." % ant)
        self.gain_params[ant] = params
    
    
    def gains_for_antenna(self, ant, freqs=None, lsts=None):
        """
        Return the complex gains as a 2D array in frequency and 
        time for a given antenna.
        
        Parameters
        ----------
        ant : int
            Antenna ID.
        
        freqs, lsts : array_like, optional
            1D arrays of frequencies and times, defining the grid that the 
            gains will be evaluated on. If None, these will be taken from the 
            `self.uvd` UVData object.
            Default: None.
        
        Returns
        -------
        gains : array_like, complex
            2D complex gain array.
        """
        if ant not in self.gain_params.keys():
            raise KeyError("Antenna '%s' not found." % ant)
        
        # Get freqs/lsts arrays if not specified
        if freqs is None:
            freqs = np.unique(self.uvd.freq_array)
        if lsts is None:
            lsts = np.unique(self.uvd.lst_array)
        
        # Evaluate model and return
        return self.gain_model(freqs, lsts, self.gain_params[ant])
    
    
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
        for ant in self.free_ants_gains:
            for p in self.free_params_gains:
                pnames.append("%s_%03d" % (p, ant))
        
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
            precision=2,                              # fixed
            use_pixel_beams=False, # Do not use pixel beams
            bm_pix=10,
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
        
        ###obs.set_beam(beam_list) # beam list
        
        # Run simulation
        if self.verbose:
            print("  Beginning simulation")
        tstart = time.time()
        
        # Compute visibilities
        gsm_vis, _times, _bls = obs.make_visibilities(self.diffuse_gsm,
                                              beam_pol=cfg_diffuse['beam_pol'], 
                                              Nprocs=cfg_diffuse['nprocs'])
        
        if self.verbose:
            print("  Simulation took %2.1f sec" % (time.time() - tstart))
        
        self.diffuse_vis = gsm_vis
        return gsm_vis, _times, _bls
    
    
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
        self.uvd = _uvd
        
        # Run diffuse model simulation
        if self.include_diffuse:
            vis_gsm, _times, _bls = self.simulate_diffuse()
            
            # Add to UVData object
            # FIXME
            
        
        return _uvd
        
