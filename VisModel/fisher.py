
import numpy as np


def calc_fisher(model, fiducial, var, dx=0.01, verbose=True):
    """
    Calculate Fisher matrix from a model.
    
    Parameters
    ----------
    model : VisModel object
        Contains model spec.
    
    fiducial : array_like
        Fiducial vector of model parameters, in the same ordering as 
        `model.param_names()`.
    
    var : float
        Estimate of the noise variance of the data.
    
    dx : float, optional
        Finite difference for all parameters. Default: 0.01.
    
    
    verbose : bool, optional
        Whether to print debug messages. Default: True.
    
    Returns
    -------
    fisher : array_like
        Fisher matrix.
    """
    pnames = model.param_names()
    assert len(pnames) == len(fiducial), \
        "Fiducial parameter vector has different size to expected number of parameters"
    
    # Fisher matrix and fiducial parameter vector
    F = np.zeros((len(pnames), len(pnames)))
    fid = fiducial.copy()
    
    # Loop over parameters
    for i in range(len(pnames)):
        if verbose:
            print("%d / %d" % (i, len(pnames)))

        # i derivative
        vmi = fid.copy()
        vpi = fid.copy()
        vmi[i] = vmi[i] - dx
        vpi[i] = vpi[i] + dx
        model_p = model.model(vpi).copy()
        model_m = model.model(vmi).copy()
        dmodel_dxi = (model_p.data_array - model_m.data_array) / (2.*dx)

        for j in range(i, len(pnames)):

            # j derivative
            vmj, vpj = fid.copy(), fid.copy()
            vmj[j] -= dx
            vpj[j] += dx
            model_pj = model.model(vpj).copy()
            model_mj = model.model(vmj).copy()
            dmodel_dxj = (model_pj.data_array - model_mj.data_array) / (2.*dx)

            # Calculate Fisher matrix element
            F[i,j] = F[j,i] = np.sum( (dmodel_dxi.real * dmodel_dxj.real 
                                     + dmodel_dxi.imag * dmodel_dxj.imag) / var )
    
    return F


def calc_fisher_bigmem(model, fiducial, var, dx=0.01, verbose=True):
    """
    Calculate Fisher matrix from a model. Stores the derivatives in memory for 
    faster computation (but uses much more memory).
    
    Parameters
    ----------
    model : VisModel object
        Contains model spec.
    
    fiducial : array_like
        Fiducial vector of model parameters, in the same ordering as 
        `model.param_names()`.
    
    var : float
        Estimate of the noise variance of the data.
    
    dx : float, optional
        Finite difference for all parameters. Default: 0.01.
    
    verbose : bool, optional
        Whether to print debug messages. Default: True.
    
    Returns
    -------
    fisher : array_like
        Fisher matrix.
    """
    pnames = model.param_names()
    assert len(pnames) == len(fiducial), \
        "Fiducial parameter vector has different size to expected number of parameters"
    
    # Fisher matrix and fiducial parameter vector
    F = np.zeros((len(pnames), len(pnames)))
    fid = fiducial.copy()
    dmodel_dx = []
    
    # Loop over parameters to get derivatives
    for i in range(len(pnames)):
        if verbose:
            print("%d / %d" % (i, len(pnames)))

        # i derivative
        vmi = fid.copy()
        vpi = fid.copy()
        vmi[i] = vmi[i] - dx
        vpi[i] = vpi[i] + dx
        model_p = model.model(vpi).copy()
        model_m = model.model(vmi).copy()
        dmodel_dxi = (model_p.data_array - model_m.data_array) / (2.*dx)
        dmodel_dx.append(dmodel_dxi)
    
    # Loop over pairs of derivatives to calculate Fisher matrix
    for i in range(len(pnames)):
        dmodel_dxi = dmodel_dx[i]
        for j in range(i, len(pnames)):
            dmodel_dxj = dmodel_dx[j]

            # Calculate Fisher matrix element
            F[i,j] = F[j,i] = np.sum( (dmodel_dxi.real * dmodel_dxj.real 
                                     + dmodel_dxi.imag * dmodel_dxj.imag) / var )
    
    return F


def fisher_prior(pnames, priors):
    """
    Return Fisher matrix prior for a set of parameters of different types.
    
    Parameters
    ----------
    pnames : list of str
        Ordered list of parameter names.
    
    priors : dict of float
        Dictionary containing sigma values for each type of parameter.
    
    Returns
    -------
    Fprior : array_like
        Prior Fisher matrix.
    """
    Fprior = np.zeros((len(pnames), len(pnames)))
    sigma_vals = np.zeros(len(pnames))
    
    # Loop over parameter types
    for i in range(sigma_vals.size):
        for key in priors.keys():
            if key in pnames[i]:
                sigma_vals[i] = priors[key]
    
    # Put into prior matrix
    Fprior[np.diag_indices(Fprior.shape[0])] = 1. / sigma_vals**2.
    return Fprior
    
