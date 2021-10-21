
import numpy as np
import vis_cpu
import pylab as plt


def source_az_za(lsts, ra, dec, latitude, orientation="uvbeam", periodic_azimuth=False):
    """
    Get the azimuth and zenith angle for a set of sources as a function of LST.
    
    Parameters
    ----------
    lsts : array_like
        1D array of LST values.
    
    ra, dec : array_like
        1D arrays of RA and Dec values for the sources.
    
    latitude : float
        Latitude of the observing site, in radians.
    
    orientation : str, optional
        Orientation convention used for the azimuth angle. The default is
        ``'astropy'``. See ``vis_cpu.conversions.enu_to_az_za`` for more 
        information.

    periodic_azimuth : bool, optional
        if True, constrain az to be between 0 and 2 * pi. See 
        ``vis_cpu.conversions.enu_to_az_za`` for more information.
    
    Returns
    -------
    az, za : array_like
        Azimuth and zenith angle for each source as a fucntion of LST.
    """
    # Get equatorial to topocentric coordinate transforms as a function of LST
    eq2tops = np.array([vis_cpu.conversions.eci_to_enu_matrix(lst, latitude) 
                        for lst in lsts])

    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = vis_cpu.conversions.point_source_crd_eq(ra, dec)
    
    # Get the azimuth and zenith angle for each source vs LST
    az = []
    za = []
    for eq2top in eq2tops:
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        _az, _za = vis_cpu.conversions.enu_to_az_za(enu_e=tx, enu_n=ty, 
                                                  orientation=orientation, 
                                                  periodic_azimuth=periodic_azimuth)
        
        # Flip for sources below the horizon (tz < 0)
        _za[np.where(tz < 0.)] = np.pi - _za[np.where(tz < 0.)]
        az.append(_az)
        za.append(_za)
    return np.array(az), np.array(za)


def plot_source_positions(ra, dec, lst, flux, latitude):
    """
    Plot the positions of sources in azimuth and altitude above horizon at a 
    given LST. Each source is plotted with a circle with size proptional to 
    flux.
    
    Parameters
    ----------
    ra, dec : array_like
        RA and Dec locations of sources.
    
    lst : float
        LST of the observation, in radians.
    
    flux : array_like
        Flux of each source.
    
    latitude : float
        Latitude of the array, in radians.
    
    Returns
    -------
    ax : matplotlib.Axes
        Plot object.
    """
    # Get azimuth and zenith angle
    az, za = source_az_za(np.atleast1d(lst), ra, dec, 
                          latitude=latitude, 
                          orientation="uvbeam", 
                          periodic_azimuth=False)
    
    # Plot locations of sources with circle sizes that depend on flux
    ax = plt.subplot(111)
    plt.scatter(az*180./np.pi, (0.5*np.pi - za)*180./np.pi, s=0.5*flux, alpha=0.3)
    plt.plot(az*180./np.pi, (0.5*np.pi - za)*180./np.pi, 'bx', alpha=0.3)
    
    # Below-horizon shading
    plt.fill_between([-90., 270.], [-90., -90.], [0., 0.], color='k', alpha=0.3)
    
    plt.ylim(-90., 90.)
    plt.xlim(-90., 270.)
    plt.gcf().set_size_inches((14., 7.))
    return ax
