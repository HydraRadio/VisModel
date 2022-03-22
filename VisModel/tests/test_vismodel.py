
from mpi4py import MPI
comm = MPI.COMM_WORLD

import numpy as np

import VisModel as vm
from VisModel import utils
import uvtools
import hera_cal as hc
from pyuvdata import UVData
from pyuvsim.simsetup import initialize_uvdata_from_keywords

from hera_sim.visibilities import VisCPU, conversions
from hera_sim.beams import PolyBeam, PerturbedPolyBeam
from hera_sim.io import empty_uvdata
import vis_cpu

from astropy.time import Time
import time, copy, sys


def default_cfg():
    """
    Set parameter defaults.
    """
    # Simulation specification
    cfg_spec = dict( nfreq=64,
                     start_freq=1.e8,
                     channel_width=1e6,
                     start_time=2458902.33333,
                     integration_time=40.,
                     ntimes=3,
                     cat_name="gleamegc.dat",
                     apply_gains=True,
                     apply_noise=True,
                     ant_pert=False,
                     seed=None,
                     ant_pert_sigma=0.0,
                     use_legacy_array=False,
                     hex_spec=(4,5), 
                     hex_ants_per_row=None, 
                     hex_ant_sep=14.6,
                     use_ptsrc=True )
                        
    # Diffuse model specification
    cfg_diffuse = dict( use_diffuse=False,
                        nside=64,
                        obs_latitude=-30.7215277777,
                        obs_longitude = 21.4283055554,
                        obs_height = 1073,
                        beam_pol='XX',
                        diffuse_model='GSM',
                        eor_random_seed=42,
                        nprocs=1 )
    
    # Beam model parameters
    cfg_beam = dict( ref_freq=1.e8,
                     spectral_index=-0.6975,
                     #seed=None,
                     perturb_scale=0.0,
                     mainlobe_scale=1.0,
                     xstretch=1.0,
                     ystretch=1.0,
                     rotation=0.0,
                     mainlobe_width=0.3, 
                     beam_coeffs=[ 0.29778665, -0.44821433, 0.27338272, 
                                  -0.10030698, -0.01195859, 0.06063853, 
                                  -0.04593295,  0.0107879,  0.01390283, 
                                  -0.01881641, -0.00177106, 0.01265177, 
                                  -0.00568299, -0.00333975, 0.00452368,
                                   0.00151808, -0.00593812, 0.00351559
                                 ] )
    
    # Fluctuating gain model parameters
    cfg_gain = dict(nmodes=8, seed=None)
    
    # Noise parameters
    cfg_noise = dict(nsamp=1., seed=None, noise_file=None)
    
    # Combine into single dict
    cfg = { 'sim_beam':     cfg_beam,
            'sim_spec':     cfg_spec,
            'sim_diffuse':  cfg_diffuse,
            'sim_noise':    cfg_noise,
            'sim_gain':     cfg_gain,
           }
    return cfg


def setup_uvdata():
	# Load template configuration
	cfg = default_cfg()
	cfg_spec = cfg['sim_spec']

	# Observation time and location
	obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")
	hera_lat, hera_lon, hera_alt = -30.7215, 21.4283, 1073.0

	# Initialise shape of data and antenna array
	ants = utils.build_hex_array(hex_spec=cfg_spec['hex_spec'], 
	                             ants_per_row=cfg_spec['hex_ants_per_row'], 
	                             d=cfg_spec['hex_ant_sep'])


	#uvd_init = utils.empty_uvdata(array_layout=ants, **cfg_spec)
	uvd_init = initialize_uvdata_from_keywords(array_layout=ants,
	                                           telescope_name="test_array",
	                                           x_orientation="east",
	                                           phase_type="drift",
	                                           vis_units="Jy",
	                                           complete=True,
	                                           write_files=False,
	                                           polarization_array=np.array(["XX",]),
	                                           telescope_location=(hera_lat, hera_lon, hera_alt),
	                                           Nfreqs=cfg_spec['nfreq'],
	                                           start_freq=cfg_spec['start_freq'],
	                                           start_time=obstime.jd,
	                                           integration_time=cfg_spec['integration_time'],
	                                           Ntimes=cfg_spec['ntimes'],
	                                           channel_width=cfg_spec['channel_width'],
	                                           )
	return uvd_init

def test_init_vismodel(uvd_init, comm=None):
	"""
	Simple test to make sure that VisModel object can be initialised.
	"""
	# Default configuration params
	cfg = default_cfg()

	# Setup point source catalogue
	ra_dec, flux = utils.load_ptsrc_catalog('/home/phil/hera/non-redundant-pipeline/catBC.txt', 
	                                        freq0=100e6, freqs=np.unique(uvd_init.freq_array), 
	                                        usecols=(0,1,2,3))
	# Sort by flux
	idxs = np.argsort(flux[0,:])[::-1]
	flux_sorted = flux[:,idxs]
	ra_dec_sorted = ra_dec[idxs,:]
	
	# Select only the first few brightest sources
	ptsrc_ra = ra_dec_sorted[:10,0]
	ptsrc_dec = ra_dec_sorted[:10,1]
	ptsrc_flux = flux_sorted[:,:10]

	"""
	model = vm.VisModel(uvd_init=uvd_init, 
	                    ptsrc_ra=ptsrc_ra, 
	                    ptsrc_dec=ptsrc_dec, 
	                    ptsrc_flux=ptsrc_flux,
	                    beam_model=PerturbedPolyBeam, 
	                    default_beam_params=cfg['sim_beam'],
	                    free_params_antpos=['antpos_dx', 'antpos_dy'], 
	                    free_params_beam=['xstretch', 'ystretch', 'spectral_index'], 
	                    free_params_ptsrc=['delta_ra', 'delta_dec', 'flux_factor'],
	                    free_ants=np.arange(10),
	                    free_beams=np.arange(10),
	                    free_ptsrcs=np.arange(ptsrc_ra.size),
	                    include_diffuse=True,
	                    verbose=True,
	                    comm=None)
	print("VisModel:", model)
	print("Parameter names:", model.param_names())

	# Print info about the VisModel object
	model.info()

	# Raise error for complex warnings
	#import warnings
	#warnings.filterwarnings(action="error", category=np.ComplexWarning)

	# Test simulation routines
	uvd_ptsrc = model.simulate_point_sources()

	uvd_diffuse = model.simulate_diffuse()
	"""


	# MPI-enabled routine
	comm.Barrier()
	verbose = False
	if myid == 0:
		verbose = True
	mpi_model = vm.VisModelPerWorker(
						uvd_init=uvd_init, 
	                    ptsrc_ra=ptsrc_ra, 
	                    ptsrc_dec=ptsrc_dec, 
	                    ptsrc_flux=ptsrc_flux,
	                    beam_model=PerturbedPolyBeam, 
	                    default_beam_params=cfg['sim_beam'],
	                    free_params_antpos=['antpos_dx', 'antpos_dy'], 
	                    free_params_beam=['xstretch', 'ystretch', 'spectral_index'], 
	                    free_params_ptsrc=['delta_ra', 'delta_dec', 'flux_factor'],
	                    free_ants=np.arange(10),
	                    free_beams=np.arange(10),
	                    free_ptsrcs=np.arange(ptsrc_ra.size),
	                    include_diffuse=True,
	                    verbose=verbose,
	                    comm=comm)
	uvd_ptsrc = mpi_model.simulate_point_sources()
	print("Frequencies (worker=%d):" % myid, mpi_model._freqs)


if __name__ == '__main__':
	
	t0 = time.time()
	myid = comm.Get_rank()
	uvd = setup_uvdata()
	test_init_vismodel(uvd, comm=comm)
	comm.Barrier()
	if myid == 0:
		print("Run took %4.1f sec" % (time.time() - t0))