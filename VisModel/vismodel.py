
import numpy as np
from pyuvdata import UVData
from pyradiosky import SkyModel

from hera_sim.visibilities import VisibilitySimulation, ModelData, VisCPU
from hera_sim.beams import PolyBeam, PerturbedPolyBeam

from astropy.units import sday, rad
from astropy import units
from astropy.coordinates.angles import Latitude, Longitude

from .gains import BaseGainModel
from .utils import gsm_sky_model
import time, copy, sys


class VisModel(object):
    
    def __init__(self, uvd_init=None, 
                 ptsrc_ra=None, ptsrc_dec=None, ptsrc_flux=None,
                 beam_model=PerturbedPolyBeam, 
                 gain_model=None,
                 default_beam_params={},
                 free_params_antpos=[], free_params_beam=[], 
                 free_params_ptsrc=[], free_params_gains=[],
                 free_beams=[], free_ants=[], free_ants_gains=[], 
                 free_ptsrcs=[], 
                 extra_opts_viscpu={},
                 include_diffuse=False, 
                 diffuse_nside=32,
                 verbose=False, comm=None):
        """
        Construct a model for a set of visibility data from a series 
        of data model components. This object handles caching of 
        results to avoid expensive recomputations where possible.
        
        Parameters
        ----------
        uvd_init : UVData object
            UVData object with the correct data shape (number of frequencies, 
            times, polarisations, antennas).
        
        ptsrc_ra, ptsrc_dec, ptsrc_flux : array_like
            Point source position and flux arrays. RA and Dec should be in radians.
        
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
        
        include_diffuse : bool, optional
            Whether to include a diffuse model in the visibility model. This 
            uses GSM as the model. Default: False.

        diffuse_nside : int, optional
            Healpix map nside for the diffuse model. Default: 32.
        
        verbose : bool, optional
            Whether to print debug messages. Default: True.
        
        comm : MPI communicator, optional
            MPI communicator to be passed to simulator. Default: None.
        """
        tstart = time.time()
        
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
        self.ptsrc_ra = ptsrc_ra.copy()
        self.ptsrc_dec = ptsrc_dec.copy()
        self.ptsrc_flux = ptsrc_flux.copy()
        self.ptsrc_delta_ra = np.zeros(self.ptsrc_ra.shape)
        self.ptsrc_delta_dec = np.zeros(self.ptsrc_dec.shape)
        self.ptsrc_flux_factor = np.ones(self.ptsrc_ra.size)
        assert self.ptsrc_ra.size == self.ptsrc_dec.size == self.ptsrc_flux.shape[-1], \
            "ptsrc_ra, ptsrc_dec, ptsrc_flux have inconsistent shapes"
        assert self.ptsrc_ra.size >= len(free_ptsrcs), \
            "free_ptsrcs must not have more sources than the total no. of sources in the sky model"
        
        # Mappings between param vector and param names
        self.free_params_antpos = free_params_antpos
        self.free_params_beam = free_params_beam
        self.free_params_ptsrc = free_params_ptsrc
        self.free_params_gains = free_params_gains
        self.free_beams = free_beams
        self.free_ants = free_ants
        self.free_ants_gains = free_ants_gains
        self.free_ptsrcs = free_ptsrcs
        
        # MPI handling
        self.comm = comm
        
        # Construct beam model with new parameter values
        for i in range(self.uvd.Nants_data):
            self.beams[i] = self.BeamClass(**self.default_beam_params)
        
        # Set up diffuse model
        self.include_diffuse = include_diffuse
        self.diffuse_nside = diffuse_nside
        if self.include_diffuse:
            
            # Build SkyModel from GSM (pygdsm)
            self.gsm_sky = gsm_sky_model(self._freqs, 
                                         resolution="hi", 
                                         nside=self.diffuse_nside)

        # Model visibility:
        #   Antenna positions
        #   Primary beam models
        #   Sky model: point sources
        #   Sky model: Gaussian sources
        #   Sky model: diffuse

        if self.verbose:
            print("  VisModel init took %2.1f sec" % (time.time() - tstart))
    
    
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
            self.ptsrc_delta_ra[i] = \
                             params[self.free_params_ptsrc.index('delta_ra')]
        if 'delta_dec' in self.free_params_ptsrc:
            self.ptsrc_delta_dec[i] = \
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
    

    def info(self):
        """
        Print basic info about the VisModel object, including the shape and size of 
        the data model and sky model. 
        """
        # Basic info about the data
        print("")
        print("="*60)
        print("-"*40)
        print("VisModel data layout")
        print("-"*40)
        print("Freqs:     %d (%6.2f -- %6.2f MHz)" % (self._freqs.size, 
                                                     self._freqs.min()/1e6,
                                                     self._freqs.max()/1e6))
        print("Times:     %d (%6.2f -- %6.2f)" % (self._times.size, 
                                                 self._times.min(),
                                                 self._times.max()))
        print("Pols:      %d" % len(self._pols))
        print("          ", self._pols)
        ants = self.uvd.get_ants()
        print("Data ants: %d" % len(ants))
        print("          ", ants)

        # Basic info about the sky model
        print("")
        print("-"*40)
        print("Sky model")
        print("-"*40)
        print("Point srcs:    %d" % self.ptsrc_ra.size)
        print("               %d free parameters" % len(self.free_params_ptsrc))
        print("Diffuse model: %s" % self.include_diffuse)
        if self.include_diffuse:
            print("    nside:     %d" % self.diffuse_nside)

        # Simple memory usage
        print("")
        print("-"*40)
        print("Basic memory usage")
        print("-"*40)
        print("uvdata:         %7.2f MB" % (self.uvd.data_array.nbytes/(1024**2)))
        print("ptsrc model:    %7.2f MB" % \
            ((self.ptsrc_flux.nbytes + 2*self.ptsrc_ra.nbytes)/(1024**2)))
        if self.include_diffuse:
            print("gsm_sky model:  %7.2f MB" % (self.gsm_sky.stokes.nbytes/(1024**2)))
        print("="*60)


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

        # Construct SkyModel object
        ra_new = self.ptsrc_ra + self.ptsrc_delta_ra
        dec_new = self.ptsrc_dec + self.ptsrc_delta_dec
        flux = self.ptsrc_flux * self.ptsrc_flux_factor
        nsrc = ra_new.size
        sky_model = SkyModel(
                            ra=Longitude(ra_new, unit='rad'),
                            dec=Latitude(dec_new, unit='rad'),
                            stokes=np.array(
                                [
                                    flux,                       # Stokes I
                                    np.zeros((len(self._freqs), nsrc)), # Stokes Q = 0
                                    np.zeros((len(self._freqs), nsrc)), # Stokes U = 0
                                    np.zeros((len(self._freqs), nsrc)), # Stokes V = 0
                                ]
                            ),
                            name=np.array(["sources"] * nsrc),
                            spectral_type="full",
                            freq_array=self._freqs,
                        )
        data_model = ModelData(uvdata=self.uvd, 
                               sky_model=sky_model,
                               beams=self.beams)

        # Initialise VisCPU handler object
        viscpu = VisCPU(use_pixel_beams=False, 
                        precision=2, 
                        mpi_comm=comm, 
                        **self.extra_opts_viscpu)

        # Create a VisibilitySimulation object
        simulator_ptsrc = VisibilitySimulation(data_model=data_model, 
                                               simulator=viscpu)
        
        # Run simulation
        if self.verbose:
            print("  Beginning ptsrc simulation")
        tstart = time.time()
        vis = simulator_ptsrc.simulate()
        if self.verbose:
            print("  Simulation took %2.1f sec" % (time.time() - tstart))
        
        return simulator_ptsrc.uvdata
    
    
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
        if not self.include_diffuse:
            raise ValueError("Diffuse model is disabled because include_diffuse=False")

        # Copy the UVData object
        uvd = copy.deepcopy(self.uvd)

        # Need to zero the array, which is usually returned by 
        # reference rather than copied
        uvd.data_array *= 0.

        # Construct a data model with the latest beams
        data_model = ModelData(uvdata=uvd, 
                               sky_model=self.gsm_sky,
                               beams=self.beams)

        # Initialise VisCPU handler object
        viscpu = VisCPU(use_pixel_beams=False, 
                        precision=2, 
                        mpi_comm=comm, 
                        **self.extra_opts_viscpu)

        # Create a VisibilitySimulation object
        simulator_diffuse = VisibilitySimulation(data_model=data_model, 
                                                 simulator=viscpu)
        
        # Run simulation
        if self.verbose:
            print("  Beginning diffuse simulation")
        tstart = time.time()
        vis = simulator_diffuse.simulate()
        if self.verbose:
            print("  Simulation took %2.1f sec" % (time.time() - tstart))
        
        return simulator_diffuse.uvdata
    
    
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
            uvd_gsm = self.simulate_diffuse()
            
            # Add diffuse data to UVData object
            _uvd.data_array[:,:,:,:] += uvd_gsm.data_array[:,:,:,:]
        
        # Apply gain model
        _uvd = self.apply_gains(_uvd)
        
        # Update internal UVData array and return
        self.uvd = _uvd
        return _uvd


def VisModelPerWorker(**kwargs):
    """
    Splits a VisModel object into multiple instances, each with a different frequency 
    range. Each instance will be owned by a different MPI worker.

    NOTE: Must have more frequencies than workers.
    NOTE: Each returned `VisModel` instance will have `comm=None`, so it thinks it is a 
          single-process instance.

    Parameters
    ----------
    **kwargs : dict
        Arguments for the `VisModel` constructor.
    """
    # Return a single instance if no MPI comm is specified 
    if 'comm' not in kwargs.keys() or kwargs.get('comm') is None:
        return VisModel(**kwargs)

    # Get MPI communicator
    comm = kwargs.pop('comm')
    myid = comm.Get_rank()
    nworkers = comm.Get_size()

    # Get UVData object and down-select frequency channels
    uvd_init = kwargs.pop('uvd_init')
    freqs = np.unique(uvd_init.freq_array)
    if nworkers > freqs.size:
        raise ValueError("More MPI workers (%d) than frequency channels (%d)." \
                         % (nworkers, freqs.size))

    # Split frequency array into chunks managed by each worker
    freq_chunk = np.array_split(freqs, nworkers)[myid]
    idxs = np.array_split(np.arange(freqs.size), nworkers)[myid]

    # Create a VisModel per worker (array_split() splits into roughly equal-length 
    # contiguous parts)
    # NOTE: This will pass comm=None to the VisModel constructor, so each worker's 
    # copy of VisModel thinks it is a single-process instance.
    my_uvd_init = uvd_init.select(frequencies=freq_chunk, inplace=False)
    
    # Down-select frequency channels from input point-source flux array
    ptsrc_flux = kwargs.pop('ptsrc_flux')
    my_ptsrc_flux = None
    if ptsrc_flux is not None:
        my_ptsrc_flux = ptsrc_flux[idxs,:]

    # Return VisModel instance for each worker
    return VisModel(uvd_init=my_uvd_init, ptsrc_flux=my_ptsrc_flux, **kwargs)
