
import numpy as np
from pyuvdata import UVData

from hera_sim.visibilities import VisCPU
from hera_sim.beams import PolyBeam, PerturbedPolyBeam
import time, copy, sys


class VisModel(object):
    
    def __init__(self, uvd_init, 
                 ptsrc_ra_dec, ptsrc_flux,
                 beam_model=PerturbedPolyBeam, 
                 default_beam_params={},
                 free_params_antpos=[], free_params_beam=[], 
                 free_params_ptsrc=[],
                 free_beams=[], free_ants=[], free_ptsrcs=[],
                 verbose=False, comm=None):
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
        
        default_beam_params : dict
            Default values of all beam model parameters.
        
        free_params_antpos, free_params_beam, free_params_ptsrc : lists of str
            Ordered lists of names of parameters to be varied. 
            The ordering maps to the parameter vector.
        
        free_beams, free_ants, free_ptsrcs : list of int
            Ordered list of which beams/antennas/point sources 
            are to have their parameters varied. The ordering 
            maps to the parameter vector.
        
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
        self._ant_index = self.uvd.antenna_numbers
        
        # Initial antenna positions (for antennas with free positions)
        self.antpos_initial = {}
        for ant in free_ants:
            pos = self.uvd.antenna_positions[np.where(self.uvd.antenna_numbers == ant)]
            self.antpos_initial[ant] = pos.flatten()
        
        # Set beam type
        self.BeamClass = beam_model
        self.beams = [[] for i in range(self.uvd.Nants_data)]
        self.default_beam_params = default_beam_params
        
        # Point source catalogue
        self.ptsrc_ra_dec = ptsrc_ra_dec.copy()
        self.ptsrc_flux = ptsrc_flux.copy()
        self.ptsrc_delta_radec = np.zeros(self.ptsrc_ra_dec.shape)
        self.ptsrc_flux_factor = np.ones(self.ptsrc_ra_dec.shape[0])
        
        # Mappings between param vector and param names
        self.free_params_antpos = free_params_antpos
        self.free_params_beam = free_params_beam
        self.free_params_ptsrc = free_params_ptsrc
        self.free_beams = free_beams
        self.free_ants = free_ants
        self.free_ptsrcs = free_ptsrcs
        
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
            self.ptsrc_delta_radec[i][0] = \
                             params[self.free_params_ptsrc.index('delta_ra')]
        if 'delta_dec' in self.free_params_ptsrc:
            self.ptsrc_delta_radec[i][1] = \
                             params[self.free_params_ptsrc.index('delta_dec')]
        if 'flux_factor' in self.free_params_ptsrc:
            self.ptsrc_flux_factor[i] = \
                             params[self.free_params_ptsrc.index('flux_factor')]
    
    
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
        
        # Point source parameters
        for i in self.free_ptsrcs:
            for p in self.free_params_ptsrc:
                pnames.append("%s_%06d" % (p, i))
        
        return pnames
    
    
    def simulate(self):
        """
        Simulate visibilities for the current state of the model.
        
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
            mpi_comm=self.comm
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
        Nfree_beams = len(self.free_beams)
        Nfree_ptsrcs = len(self.free_ptsrcs)
        Nparams_ants = len(self.free_params_antpos)
        Nparams_beams = len(self.free_params_beam)
        Nparams_ptsrcs = len(self.free_params_ptsrc)
        
        # Extract parameters in blocks and reshape
        iants = Nfree_ants * Nparams_ants
        ibeams = Nfree_beams * Nparams_beams
        iptsrcs = Nfree_ptsrcs * Nparams_ptsrcs
        antpos_params = params[0:iants].reshape((Nfree_ants, Nparams_ants))
        beam_params = params[iants:iants+ibeams].reshape((Nfree_beams, Nparams_beams))
        ptsrc_params = params[iants+ibeams:iants+ibeams+iptsrcs].reshape(
                                                (Nfree_ptsrcs, Nparams_ptsrcs) )
        
        # (1) Antenna positions
        for i, ant in enumerate(self.free_ants):
            self.set_antpos(ant, antpos_params[i])
        
        # (2) Beam parameters
        for i, ant in enumerate(self.free_beams):
            self.set_beam(ant, beam_params[i])
        
        # (3) Point source parameters
        for i in self.free_ptsrcs:
            self.set_ptsrc_params(i, ptsrc_params[i])
        
        # Run simulation
        _uvd = self.simulate()
        self.uvd = _uvd
        return _uvd
        
