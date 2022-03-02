import numpy as np
import scipy.optimize
import healpy as hp

from astropy import units
from astropy.coordinates import SkyCoord
from numba import jit

C = 299792458.0  # m/s, speed of light


def enu_vector(alt, az, deg=True):
    """
    Convert alt/az coordinates to a Cartesian East-North-Up unit vector. This 
    function uses a "North of East" azimuth convention.
    
    Parameters:
        alt, az (array_like):
            Altitude and azimuth coordinates in degrees or radians.
        deg (bool):
            If `True`, `alt` and `az` are in degrees (in radians otherwise).
    
    Returns:
        enu (array_like):
            Unit vector in East-North-Up coordinates.
    """
    # North of East azimuth convention
    if deg:
        alt = np.deg2rad(alt)
        az = np.deg2rad(az)
    return np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])


def enu_vector_to_angle(e, n=None, u=None):
    """
    Convert a Cartesian East-North-Up unit vector to an altitude and azimuth 
    angle. This function uses the same "North of East" azimuth convention as 
    the `enu_vector()` function.
    
    Parameters:
        e (array_like):
            East unit vector coordinate OR a (3, N) array containing all 3 ENU 
            coordinates.
        n (array_like):
            North unit vector coordinate. If `None`, the `e` argument will be 
            used as a 2D aray of ENU coordinates instead.
        u (array_like):
            Up unit vector coordinate. If `None`, the `e` argument will be used 
            as a 2D aray of ENU coordinates instead.
    
    Returns:
        alt, az (array_like):
            Altitude and azimuth coordinates, in degrees.
    """
    # r = np.sqrt(e**2. + n**2. + u**2.) # should be 1
    if n is None and u is None:
        assert len(e.shape) == 2 and e.shape[0] == 3, \
            "If 'n' and 'u' aren't specified, 'e' must be a (3, N) array"
        n = e[1]
        u = e[2]
        e = e[0]
    theta = 0.5 * np.pi - np.arccos(u)  # np.arccos(u)
    phi = np.arctan2(n, e)
    return np.rad2deg(theta), np.rad2deg(phi)


def rotate_eq_to_altaz(ra, dec, frame_altaz=None, R=None, use_astropy=False):
    """
    Rotate RA and Dec coordinates to Alt/Az coordinates.
    
    Parameters:
        ra, dec (array_like):
            Equatorial RA and Dec coordinates in degrees.
        
        frame_altaz (astropy.AltAz object):
            AltAz frame with a location and time specified.
        
        R (array_like):
            3x3 rotation matrix. If specified, this will be used 
            to do the Eq->AltAz coordinate conversion. Use this 
            if you want to reuse a precomputed R matrix for speed-up.
        
        use_astropy (bool):
            Whether to use astropy to do the coordinate conversion 
            directly (True), or use a rotation matrix derived from 
            an astropy coordinate conversion of the frame axes.
        
    Returns:
        alt, az (array_like):
            Altitude and azimuth coordinates.
    
    Example of setting up an Alt/Az coordinate system:
    
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.coordinates.builtin_frames import AltAz, ICRS
        from astropy.time import Time
        
        # HERA location
        location = EarthLocation.from_geodetic(lat=-30.7215,
                                               lon=21.4283,
                                               height=1073.)
        # Observation time
        obstime = Time('2018-08-31T04:02:30.11', format='isot', scale='utc')

        # Define AltAz frame
        frame_altaz = AltAz(obstime=obstime, location=location)
    """
    if frame_altaz is None and R is None:
        raise ValueError("Must specify frame_altaz, or R")

    # Use astropy if requested
    if use_astropy:
        c = SkyCoord(ra=ra * units.degree, dec=dec * units.degree, frame="icrs")
        c_altaz = c.transform_to(frame_altaz)
        return c_altaz.alt.deg, c_altaz.az.deg

    # Calculate rotation matrix
    if R is None:
        R = rotation_matrix(frame_altaz)

    # Convert using rotation matrix
    e, n, u = enu_vector(dec, ra)
    x_e, x_n, x_u = np.dot(R, np.array([e, n, u]))
    x_alt, x_az = enu_vector_to_angle(x_e, x_n, x_u)
    return x_alt, x_az


def fit_rotation_matrix(frame_altaz, frame_eq="icrs", nsamp=40):
    """
    Find the best-fit rotation matrix for an astropy coordinate transformation 
    on a set of points.
    
    Note that the best-fit transform will leave an approximately dipole 
    residual compared to an exact point-by-point transform. This is due to 
    aberration etc, which means that there is no exact rigid-body transform 
    from equatorial to AltAz.
    
    Example usage:
        # Alt/Az frame
        frame_altaz = AltAz(...)
        
        # Get rotation matrix
        rot = fit_rotation_matrix(frame_altaz, frame_eq="icrs")
        
        # Apply rotation to sky coordinates
        coords = SkyCoord(...) # source coords
        vec_eq = np.array(coords.cartesian.xyz) # Equatorial coords (Cartesian)
        vec_aa = np.dot(rot, vec_eq) # Alt/Az coords (Cartesian)
        alt, az = enu_vector_to_angle(vec_aa) # Alt/Az angles
    
    Parameters:
        frame_altaz (astropy.coordinates.builtin_frames.AltAz):
            An astropy AltAz reference frame, at the time and location of the 
            observer.
        frame_eq (astropy.coordinates.builtin_frames.ICRS):
            An astropy equatorial frame, in which the astronomical source 
            coordinates are defined.
        nsamp (int):
            Number of randomly-sampled points on the sphere to use when 
            determining the fit.
    
    Returns:
        rot_mat (array_like):
            3x3 rotation matrix to rotate Cartesian xyz vectors in equatorial 
            to AltAz coordinates.
    """
    # Random RA and Dec values
    ra_vals = np.random.uniform(low=0.0, high=359.99, size=nsamp)
    dec_vals = np.random.uniform(low=-90.0, high=90.0, size=nsamp)

    # Get SkyCoord object and Alt/Az coords
    coords = SkyCoord(
        ra=ra_vals * units.degree, dec=dec_vals * units.degree, frame=frame_eq
    )
    coords_altaz = coords.transform_to(frame_altaz)

    # Get Cartesian representations
    vec_eq = np.array(coords.cartesian.xyz)
    vec_aa = np.array(coords_altaz.cartesian.xyz)
    vec_eq[2] *= -1.0  # flip the handedness of the coords

    def resid(p, input_vec, target_vec):
        rot = p.reshape((3, 3))
        return (np.dot(rot, input_vec) - target_vec).flatten()

    # Do leastsq fit to find rotation matrix
    p0 = np.eye(3)
    p_bf, _ = scipy.optimize.leastsq(resid, p0, args=(vec_eq, vec_aa))

    # Apply rotation to input coords
    rot = p_bf.reshape((3, 3))
    return rot


@jit
def delay(alt, az, bl, deg=False):
    """
    Calculate the time delay between a baseline vector and an alt/az coordinate.
    
    This function assumes that alt/az are in a "North of East" azimuth 
    convention (where Azimuth = 0 in the East).
        
    Parameters:
        alt, az (array_like):
            Altitude and azimuth coordinates in radians or degrees (if 
            `deg=True`).
        bl (array_like):
            Baseline vector, `[x, y, z]` in metres, where x points East.
        deg (bool):
            Whether `alt` and `az` are in degrees (`True`) or radians (`False`).
    
    Returns:
        tau (array_like):
            Time delay, in ns.
    """
    # Convert degrees to radians
    if deg:
        alt = np.deg2rad(alt)
        az = np.deg2rad(az)
        
    # Calculate zenith angle
    alt = np.atleast_1d(alt)
    az = np.atleast_1d(az)
    za = 0.5 * np.pi - alt

    # Source vector
    src_vec = np.array([np.sin(za) * np.cos(az), np.sin(za) * np.sin(az), np.cos(za)]).T

    # Return dot product
    bl = np.atleast_2d(bl).T
    return np.dot(src_vec, bl) / C * 1e9  # ns


@jit
def fringe(alt, az, bl, freq=100.0, deg=False):
    """
    Calculate the fringe pattern for point sources at given alt/az coordinates 
    and frequencies.
    
    Parameters:
        alt, az (array_like):
            Altitude and azimuth coordinates in radians or degrees (if 
            `deg=True`).
        bl (array_like):
            Baseline vector, `[x, y, z]` in metres, where x points East.
        freq (float):
            Frequency in MHz.
        deg (bool):
            Whether `alt` and `az` are in degrees (`True`) or radians (`False`).
    
    Returns:
        fringe (array_like):
            Complex fringe pattern for the baseline, evaluated at the given 
            alt/az coordinates.
    """
    # Calculate delay
    tau = delay(alt, az, bl, deg=deg)  # in ns
    
    # Calculate complex fringe pattern
    return np.exp(1.0j * 2.0 * np.pi * tau * freq * 1e-3)


@jit
def horizon(alt, az=None, deg=False):
    """
    Implement a horizon cut, which masks out contributions from points below 
    the local horizon.
    
    Parameters:
        alt, az (array_like):
            Altitude and azimuth coordinates in radians or degrees (if 
            `deg=True`).
        deg (bool):
            Whether `alt` and `az` are in degrees (`True`) or radians (`False`).
    
    Returns:
        horizon (array_like):
            Array with values of 1 or 0 depending on if a point is above or 
            below the horizon.
    """
    if deg:
        alt = np.deg2rad(alt)
        
    m = np.ones(alt.size)
    m[alt < 0.0] = 0.0
    return m


@jit
def gaussian_beam(alt, az, freq, diameter=14.0, deg=False):
    """
    Implement a Gaussian primary beam with a width that scales like lambda / D.
    
    Parameters:
        alt, az (array_like):
            Altitude and azimuth coordinates in radians or degrees (if 
            `deg=True`).
        freq (float):
            Frequency in MHz.
        diameter (float):
            Dish diameter in metres.
        deg (bool):
            Whether `alt` and `az` are in degrees (`True`) or radians (`False`).
    
    Returns:
        beam (array_like):
            Primary beam value at each source position.
    """
    if deg:
        alt = np.deg2rad(alt)
    
    width = C / (1e6 * freq) / diameter  # rad
    return np.exp(-0.5 * (alt - 0.5 * np.pi) ** 2.0 / width ** 2.0)


# FIXME: healpy function nside2npix seems to cause issues unless forceobj=True
# @jit(parallel=True, forceobj=True)
def vis_sh_response(bl, freqs, lmax, nside=64):
    """
    Calculate the response of a baseline to a spherical harmonic mode with unit 
    amplitude (a_lm = 1) on the sky with a flat frequency spectrum.
    
    Parameters:
        bl (array_like):
            Baseline [x,y,z] position in metres.
        freqs (array_like):
            Array of frequency values in MHz.
        lmax (int):
            Maximum ell value of spherical harmonic modes to calculate.
        nside (int):
            Healpix nside to use for the calculation (longer baselines should 
            use higher nside).
    
    Returns:
        response (array_like):
            Visibility `V_ij` for each (l,m) mode, with shape 
            `(N_alm, N_freqs)`.
    """
    # Construct empty vector for a_lm coefficients
    nmodes = lmax * (2 * lmax + 1 - lmax) // 2 + lmax + 1
    alm = np.zeros(nmodes, dtype=np.complex128)

    # Get alt/az values for each pixel
    npix = hp.nside2npix(nside)
    ii = np.arange(npix)
    az, alt = hp.pix2ang(nside, ii, lonlat=True)
    alt = np.deg2rad(alt.flatten())  # radians
    az = np.deg2rad(az.flatten())  # radians

    # Empty visibility response array
    vis = np.zeros((nmodes, freqs.size), dtype=np.complex128)

    # Horizon mask
    mask_horizon = horizon(alt, az)

    # Loop over frequencies
    for i in range(freqs.size):

        # Calculate fringe pattern times Gaussian primary beam squared times horizon mask
        fringe_times_beamsq = (
            fringe(alt, az, bl, freqs[i]).flatten()
            * gaussian_beam(alt, az, freqs[i]) ** 2.0
            * mask_horizon
        )

        # Loop over (l,m) modes
        for j in range(alm.size):

            # Switch on one a_lm mode at a time and generate map
            alm *= 0.0
            alm[j] = 1.0
            m = hp.sphtfunc.alm2map(alm, nside=nside)

            # Calculate integral (FIXME: unnormalised!)
            vis[j, i] = np.sum(m * fringe_times_beamsq)

    return vis


if __name__ == "__main__":
    
    import pylab as plt
    
    # Find best-fit rotation matrix
    rot, _alt, _az, alt_true, az_true = fit_rotation_matrix(frame_altaz, nsamp=2000)

    # Test properties of rotation matrix
    print(np.linalg.det(rot))

    plt.scatter(
        _az, _alt, c=(_alt - alt_true) * 3600.0, vmin=-60.0, vmax=60.0, cmap="RdBu"
    )

    plt.ylabel("Altitude [deg]", fontsize=14.0)
    plt.xlabel("Azimuth [deg]", fontsize=14.0)
    cbar = plt.colorbar()
    cbar.set_label("Delta altitude [arcsec]", fontsize=14.0)
    plt.ylim((-90.0, 90.0))
    plt.xlim((-180.0, 180.0))
    plt.gcf().set_size_inches((10.0, 6.0))

    print(rot)

    # Get rotation transform
    r = Rotation.from_matrix(rot)
    euler_angs = r.as_euler("XYZ", degrees=False)
    euler_angs
