# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:36:13 2021

@author: Saskia
"""
from sys import maxsize
from time import time
from typing import Union
import numpy as np
import matplotlib
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(precision=4, linewidth=100)


###############################################
### SIMULATION of a Detector for the Toy-MC ###
###############################################

def makeStandardGeometry(NStringX:int=7, NStringY:int=7, NDOM:int=5, 
                         Z0:float=-1500., dZ:float=-10., dString:float=125.
                         ) -> np.ndarray:
    """
    Generate an idealized geometry (rectangular, straight strings). 
    It is used as first guess for the fit and as reference.

    Parameters
    ----------
    NStringX : int, optional
        Number of strings in x projection. Total number of strings is 
        NStringX * NStringsY. The default is 7.
    NStringY : int, optional
        Number of strings in x projection. The default is 7.
    NDOM : int, optional
        Number of DOMs per string. The default is 5.
    Z0 : float, optional
        Depth of the first DOM in meter. The default is -1500.
    dZ : float, optional
        DOM spacing along the string in meter. The default is -10..
    dString : float, optional
        Spacing between adjacent strings in meter. The default is 125.

    Returns
    -------
    standardGeometry : np.ndarray
        Returns (x,y,z) for each DOM.
    """
    dY = np.sqrt(dString**2 - (dString/2)**2)
    
    standardGeometry = np.zeros((NStringX*NStringY, NDOM, 3), dtype=float)
    # define z coordinates:
    standardGeometry[:,:,2] = Z0 + np.arange(NDOM)*dZ
    
    x = np.zeros(NStringX)
    x[1::2]+=0.5
    x = np.tile(x, NStringY)
    x += np.repeat(np.arange(NStringX, dtype=float), NStringY)
    for i in range(len(x)):
        standardGeometry[i,:,0] = x[i]*dString
            
    y = np.tile(np.arange(NStringY), NStringX)*dY
    for i in range(len(y)):
        standardGeometry[i,:,1] = y[i]
    return standardGeometry

    

def simulateDetector(NStringX:int=7, NStringY:int=7, NDOM:int=5, 
                     Z0:float=-1500., dZ:float=-10., dString:float=125., 
                     scale=1., maxdeviation=1.) -> np.ndarray:
    """
    Simulates the "real" detector using a random walk along each string using 
    a standardnormal distribution.

    Parameters
    ----------
    NStringX : int, optional
        Number of strings in x projection. Total number of strings is 
        NStringX * NStringsY. The default is 7.
    NStringY : int, optional
        Number of strings in x projection. The default is 7.
    NDOM : int, optional
        Number of DOMs per string. The default is 5.
    Z0 : float, optional
        Depth of the first DOM in meter. The default is -1500.
    dZ : float, optional
        DOM spacing along the string in meter. The default is -10..
    dString : float, optional
        Spacing between adjacent strings in meter. The default is 125.
    scale : TYPE, optional
        standard deviation . The default is 1..
    maxdeviation : TYPE, optional
        Maximum deviation to the standard geometry in sqrt(x^2+y^2) in meter. 
        The default is 1..

    Returns
    -------
    np.ndarray
        Returns (x,y,z) for each DOM.
    """

    standardGeometry = makeStandardGeometry(NStringX, NStringY, NDOM, Z0, dZ, dString)
    realDetector = np.zeros((NStringX*NStringY, NDOM, 3), dtype=float)
    for i in range(NStringX*NStringY):
        x0, y0 = np.random.normal(loc=0, scale=scale, size=2) #+ standardGeometry[i,0,:2]
        z0 =  (np.random.random()*0.01)*Z0
        dx = np.random.normal(loc=0, scale=scale, size=NDOM-1)
        dy = np.random.normal(loc=0, scale=scale, size=NDOM-1)
        z = np.concatenate(([0.], -dZ - np.sqrt(dZ**2 - dx**2 - dy**2))).cumsum(0) + z0
        #print(z, dx, dy)
        #x = np.concatenate(([x0], dx)).cumsum(0)
        #y = np.concatenate(([y0], dy)).cumsum(0)
        for j in range(NDOM-1):
            realDetector[i,j] = [x0, y0, z[j]]
            if np.abs(x0+dx[j]) > maxdeviation:
                if np.abs(x0-dx[j]) < np.abs(x0+dx[j]):
                    dx *= -1
            if np.abs(y0+dy[j]) > maxdeviation:
                if np.abs(y0-dy[j]) < np.abs(y0+dy[j]):
                    dy *= -1
            if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation:
                if np.abs(x0+dx[j]) > np.abs(y0+dy[j]) and np.abs(x0+dx[j]) > np.abs(x0-dx[j]):
                    dx *= -1
                    if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation and np.abs(y0+dy[j]) > np.abs(y0-dy[j]):
                        dy *= -1
                else:
                    if np.abs(y0+dy[j]) > np.abs(y0-dy[j]):
                        dy *= -1
                        if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation and np.abs(x0+dx[j]) > np.abs(x0-dx[j]):
                            dx *= -1
            dxCache, dyCache = dx, dy
            if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation:
                #print("I hate my life")
                dx,dy = dy,dx
                if np.abs(x0+dx[j]) > maxdeviation:
                    if np.abs(x0-dx[j]) < np.abs(x0+dx[j]):
                        dx *= -1
                if np.abs(y0+dy[j]) > maxdeviation:
                    if np.abs(y0-dy[j]) < np.abs(y0+dy[j]):
                        dy *= -1
                if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation:
                    if np.abs(x0+dx[j]) > np.abs(y0+dy[j]) and np.abs(x0+dx[j]) > np.abs(x0-dx[j]):
                        dx *= -1
                        if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation and np.abs(y0+dy[j]) > np.abs(y0-dy[j]):
                            dy *= -1
                    else:
                        if np.abs(y0+dy[j]) > np.abs(y0-dy[j]):
                            dy *= -1
                            if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > maxdeviation and np.abs(x0+dx[j]) > np.abs(x0-dx[j]):
                                dx *= -1
            if np.sqrt((x0+dx[j])**2 + (y0+dy[j])**2) > np.sqrt((x0+dxCache[j])**2 + (y0+dyCache[j])**2):
                dx, dy = dxCache, dyCache
            x0 = x0 + dx[j]
            y0 = y0 + dy[j]
        realDetector[i,NDOM-1] = [x0, y0, z[NDOM-1]]
    return realDetector + standardGeometry




def calculateDistances(detector:np.ndarray) -> np.ndarray:
    """
    Caluclates the distance between two DOMs.

    Parameters
    ----------
    detector : np.ndarray
        Detector with (x,y,z). Shape like from makeStandardgeometry()

    Returns
    -------
    distances : np.ndarray
        DESCRIPTION.
    """
    distances = squareform(pdist(detector.reshape((-1,3))))
    return distances




def randomDistances(detector:np.ndarray, sigma:float=1e-3, shift:float=1, 
                    distanceDependent:bool=True, masked:bool=True) -> np.ndarray:
    """
    Simulates the distance estimastion between DOMs using a Normal Distribution. 

    Parameters
    ----------
    detector : np.ndarray
        Detector with (x,y,z) like from makeStandardGeometry().
    sigma : float, optional
        Standard deviation . The default is 1e-3.
    shift : float, optional
        DESCRIPTION. The default is 1.
    distanceDependent : bool, optional
        DESCRIPTION. The default is True.
    masked : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    normalDistributedDistances : np.ndarray
        Returns the estimated distances and the used sigmas..
    """
    nStrings, nDOM = np.shape(detector[:,:,-1])
    trueDistances = calculateDistances(detector)
                
    if distanceDependent:
        sig = sigma*trueDistances
        dist = trueDistances*shift
    else:
        sig = sigma
        dist = trueDistances + shift
    normalDistributedDistances = np.random.normal(dist, sig)
    
    if masked:
        mask = np.logical_and(trueDistances < 250, trueDistances != 0)
        mask2 = np.kron(np.eye(nStrings), np.ones((nDOM,nDOM)))
        normalDistributedDistances[np.logical_or(mask==False, mask2==1)] = 0
    return normalDistributedDistances


def sigmaTrue(detector:np.ndarray, sigma:float=1e-3, shift:float=1, 
              distanceDependent:bool=True, masked:bool=True) -> np.ndarray:
    """
    Calculate standard deviations for the simulation of measured distances.

    Parameters
    ----------
    detector : np.ndarray
        DESCRIPTION.
    sigma : float, optional
        DESCRIPTION. The default is 1e-3.
    shift : float, optional
        DESCRIPTION. The default is 1.
    distanceDependent : bool, optional
        DESCRIPTION. The default is True.
    masked : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    sig : TYPE
        returns standard deviation for the multilateration fit of toy MC.

    """
    trueDistances = calculateDistances(detector)
                
    if distanceDependent:
        sig = sigma*trueDistances
    else:
        sig = sigma
    if masked:
        mask = np.logical_and(trueDistances < 250, trueDistances != 0)
        sig[mask==False] = 0
    return sig



################
### ANALYSIS ###
################

def LikelihoodDefault(fit:np.ndarray, standardDetector:np.ndarray, measuredDistances:np.ndarray,
               sigmaEstimated:Union[np.ndarray, float], prior:bool=False, fixedDOMs:list=[], constrain:bool=True,
               nStrings:int=49, nDOM:int=5, dZ:float=-10., deviationFromStandard:float=0.1) -> float:
    """
    Returns the sum of the negative logarithmic likelihood neglecting the constant terms 1/(sqrt(2pi) sigma). 
    The Likelihood function is defined as a 1D Normal Distribution comparing measured DOM distances to the 
    calculated distances of the fitted detector.
    
    fixedDOMs = [(String0,DOM0), (String1,DOM1), ...]
    """
    if constrain:
        detectorXY = fit[nStrings:]
        detector = np.empty((nStrings, nDOM, 3))
        detector[:,:,:2] = detectorXY.reshape(nStrings, nDOM, 2)
        z0 = fit[:nStrings]
        z = np.concatenate((z0[:,None], -np.sqrt(dZ**2 - np.diff(detector[:,:,0], axis=1)**2 
                                                 - np.diff(detector[:,:,1], axis=1)**2)), axis=1).cumsum(axis=1)
        detector[:,:,2] = z
    else:
        detector = fit.reshape((nStrings, nDOM, 3))
    mask = measuredDistances != 0
    fitDistances = calculateDistances(detector)
    if isinstance(sigmaEstimated, float):
        negativeLogL = ((measuredDistances[mask] - fitDistances[mask])/sigmaEstimated)**2
    else:
        negativeLogL = ((measuredDistances[mask] - fitDistances[mask])/sigmaEstimated[mask])**2
    softprior = 0.
    if prior:
        softprior += np.sum((detector - standardDetector)**2)
    hardprior = 0.
    if fixedDOMs:
        for DOM in fixedDOMs:
            hardprior += np.sum((detector[DOM] - standardDetector[DOM])**2) / deviationFromStandard**2
    return 0.5*(np.sum(negativeLogL) + hardprior + softprior)


def performFitDefault(standardDetector:np.ndarray, measuredDistances:np.ndarray, sigmaEstimated:Union[np.ndarray, float], 
               fixedDOMs:list=[], constrain:bool=True, Z0:float=-1500., dZ:float=-10., deviationFromStandard:float=0.1,
               tol:float=1e-10, maxiter:int=1000):
    """
    Minimzes the Likelihood and returns the resulting detector geometry.
    """
    tBenchmark = time()
    nStrings, nDOM = np.shape(standardDetector)[:-1]
    if constrain:  
        result = minimize(Likelihood, np.concatenate((np.ones(nStrings)*Z0, standardDetector[:,:,0:2].flatten())), 
                          (standardDetector, measuredDistances, sigmaEstimated, fixedDOMs, 
                           True, nStrings, nDOM, dZ, deviationFromStandard), 
                          #options={"disp":True, "xtol":tol, "maxiter":maxiter}, method="trust-constr")
                          options={"iprint":0, "ftol":tol, "maxiter":maxiter, "maxfun":maxsize}, method="L-BFGS-B")
        fittedDetector = np.zeros((nStrings, nDOM, 3))
        fittedDetector[:,:,:2] = result["x"][nStrings:].reshape((nStrings,nDOM,2))
        z0 = result["x"][:nStrings]
        z = np.concatenate((z0[:,None], -np.sqrt(dZ**2 - np.diff(fittedDetector[:,:,0], axis=1)**2 
                                                 - np.diff(fittedDetector[:,:,1], axis=1)**2)), axis=1).cumsum(axis=1)
        fittedDetector[:,:,2] = z
    else:
        result = minimize(Likelihood, standardDetector.flatten(), (standardDetector, measuredDistances, sigmaEstimated, 
                                                                   fixedDOMs, False, nStrings, nDOM, dZ, deviationFromStandard), 
                          # options={"disp":True, "xtol":tol, "maxiter":maxiter}, method="trust-constr")
                          options={"iprint":0, "ftol":tol, "maxiter":maxiter, "maxfun":maxsize}, method="L-BFGS-B")
        fittedDetector = result["x"].reshape((nStrings, nDOM, 3))
    print(f"Fit result ({'constrained' if constrain else 'unconstrained'}): logL = {Likelihood(fittedDetector.flatten(), standardDetector, measuredDistances, sigmaEstimated, fixedDOMs, False, nStrings, nDOM, dZ, deviationFromStandard)}, logL = {Likelihood(fittedDetector.flatten(), standardDetector, measuredDistances, sigmaEstimated, [], False, nStrings, nDOM, dZ, deviationFromStandard)},   message: {result.message}")
    print(f"Required time: {time() - tBenchmark}")
    return fittedDetector

def Likelihood(fit:np.ndarray, standardDetector:np.ndarray, measuredDistances:np.ndarray,
               sigmaEstimated:Union[np.ndarray, float], prior:bool=False, constrain:bool=True,
               nStrings:int=49, nDOM:int=5, dZ:float=-10., deviationFromStandard:float=1.) -> float:
    """
    Returns the sum of the negative logarithmic likelihood neglecting the constant terms 1/(sqrt(2pi) sigma). 
    The Likelihood function is defined as a 1D Normal Distribution comparing measured DOM distances to the 
    calculated distances of the fitted detector.
    """
    if constrain:
        detectorXY = fit[nStrings:]
        detector = np.empty((nStrings, nDOM, 3))
        detector[:,:,:2] = detectorXY.reshape(nStrings, nDOM, 2)
        z0 = fit[:nStrings]
        z = np.concatenate((z0[:,None], -np.sqrt(dZ**2 - np.diff(detector[:,:,0], axis=1)**2 
                                                 - np.diff(detector[:,:,1], axis=1)**2)), axis=1).cumsum(axis=1)
        detector[:,:,2] = z
    else:
        detector = fit.reshape((nStrings, nDOM, 3))
    mask = measuredDistances != 0
    fitDistances = calculateDistances(detector)
    if isinstance(sigmaEstimated, float):
        negativeLogL = ((measuredDistances[mask] - fitDistances[mask])/sigmaEstimated)**2
    else:
        negativeLogL = ((measuredDistances[mask] - fitDistances[mask])/sigmaEstimated[mask])**2
    Prior = 0.
    if prior == "Square":
        Prior += (np.sum(np.sqrt(np.sum((detector - standardDetector)**2, axis=2)))) / deviationFromStandard
    elif prior == "Gauss":
        Prior += np.sum((detector - standardDetector)**2) / deviationFromStandard**2
    return 0.5*(np.sum(negativeLogL) + Prior)


def performFit(standardDetector:np.ndarray, measuredDistances:np.ndarray, sigmaEstimated:Union[np.ndarray, float],
               constrainString:bool=True, constrainFit=False, prior=False,   Z0:float=-1500., dZ:float=-10., deviationFromStandard:float=1.,
               tol:float=1e-13, maxiter:int=10000, radius:float=1.):
    """
    Minimzes the Likelihood and returns the resulting detector geometry.
    """
    tBenchmark = time()
    nStrings, nDOM = np.shape(standardDetector)[:-1]
    if constrainString:
        if constrainFit:
            def constrainFunc(x):
                detectorXY = x[nStrings:]
                detector = np.empty((nStrings, nDOM, 3))
                detector[:,:,:2] = detectorXY.reshape(nStrings, nDOM, 2)
                z0 = x[:nStrings]
                z = np.concatenate((z0[:,None], -np.sqrt(dZ**2 - np.diff(detector[:,:,0], axis=1)**2 
                                                         - np.diff(detector[:,:,1], axis=1)**2)), axis=1).cumsum(axis=1)
                detector[:,:,2] = z
                return np.min(radius**2*np.ones((nStrings, nDOM)) - np.sum((detector - standardDetector)**2, axis=2))
                              
            result = minimize(Likelihood, np.concatenate((np.ones(nStrings)*Z0, standardDetector[:,:,0:2].flatten())), 
                              (standardDetector, measuredDistances, sigmaEstimated, prior, 
                               True, nStrings, nDOM, dZ, deviationFromStandard), 
                              options={"disp":True, "xtol":tol, "maxiter":maxiter}, method="trust-constr", 
                              constraints={'type': 'ineq', 'fun': constrainFunc})
        else:
            result = minimize(Likelihood, np.concatenate((np.ones(nStrings)*Z0, standardDetector[:,:,0:2].flatten())), 
                              (standardDetector, measuredDistances, sigmaEstimated, prior, 
                               True, nStrings, nDOM, dZ, deviationFromStandard), 
                              options={"iprint":0, "ftol":tol, "maxiter":maxiter, "maxfun":maxsize}, method="L-BFGS-B")            
        fittedDetector = np.zeros((nStrings, nDOM, 3))
        fittedDetector[:,:,:2] = result["x"][nStrings:].reshape((nStrings,nDOM,2))
        z0 = result["x"][:nStrings]
        z = np.concatenate((z0[:,None], -np.sqrt(dZ**2 - np.diff(fittedDetector[:,:,0], axis=1)**2 
                                                 - np.diff(fittedDetector[:,:,1], axis=1)**2)), axis=1).cumsum(axis=1)
        fittedDetector[:,:,2] = z
    else:
        if constrainFit:
            result = minimize(Likelihood, standardDetector.flatten(), (standardDetector, measuredDistances, sigmaEstimated, prior, 
                                                                       False, nStrings, nDOM, dZ, deviationFromStandard), 
                          options={"disp":True, "xtol":tol, "maxiter":maxiter}, method="trust-constr", 
                          constraints={'type': 'ineq', 'fun': lambda x: np.min(radius**2*np.ones((nStrings, nDOM)) - np.sum((x.reshape(nStrings, nDOM, 3)- standardDetector)**2, axis=2))})
        else:
            result = minimize(Likelihood, standardDetector.flatten(), (standardDetector, measuredDistances, sigmaEstimated, prior, 
                                                                       False, nStrings, nDOM, dZ, deviationFromStandard), 
                          options={"iprint":0, "ftol":tol, "maxiter":maxiter, "maxfun":maxsize}, method="L-BFGS-B")
        fittedDetector = result["x"].reshape((nStrings, nDOM, 3))
    resStr = f"Fit result ({'constrained' if constrainFit else 'unconstrained'}): logL = {Likelihood(fittedDetector.flatten(), standardDetector, measuredDistances, sigmaEstimated, False, False, nStrings, nDOM, dZ, deviationFromStandard)},  message: {result.message}"
    resStr += f"\nRequired time: {time() - tBenchmark}"
    print(resStr)
    return fittedDetector, resStr


### Rotation of the detector ###

def rotateShiftDetector(var:tuple, detector:np.ndarray) -> np.ndarray:
    """
    Rotates and shifts the detector

    Parameters
    ----------
    var : tuple
        Variables: spherical angles a,b,c and coordinates x,y,z.
    detector : np.ndarray
        Detector which should be shifted.

    Returns
    -------
    changedDetector : TYPE
        Shifted and roteated detector.
    """
    # unpack fit variables 
    a,b,c,dX,dY,dZ = var
    x = detector[::3]
    y = detector[1::3]
    z = detector[2::3]
    
    # make rotation
    X = x*np.cos(b)*np.cos(c) + y*np.sin(c)*np.cos(b) + z*np.sin(b) + dX
    Y = x*(-np.sin(a)*np.sin(b)*np.cos(c) - np.sin(c)*np.cos(a)) + y*(-np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c)) + z*np.sin(a)*np.cos(b) + dY
    Z = x*(np.sin(a)*np.sin(c) - np.sin(b)*np.cos(a)*np.cos(c)) + y*(-np.sin(a)*np.cos(c) - np.sin(b)*np.sin(c)*np.cos(a)) + z*np.cos(a)*np.cos(b) + dZ
    changedDetector = np.empty_like(detector)
    changedDetector[::3] = X
    changedDetector[1::3] = Y
    changedDetector[2::3] = Z
    return changedDetector


def LikelihoodRotationShift(var:tuple, fit:np.ndarray, standardGeometry:np.ndarray) -> float:
    """
    Likelihood to shift and rotate the fitted detector compared to the standard
    geometry.

    Parameters
    ----------
    var : tuple
        Variables: spherical angles a,b,c and coordinates x,y,z..
    fit : np.ndarray
        Fitted detector.
    standardGeometry : np.ndarray
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    detector = rotateShiftDetector(var, fit)
    #LLH = np.sum(np.sqrt( (detector[::3] - standardGeometry[::3])**2 + (detector[1::3] - standardGeometry[1::3])**2 + (detector[2::3] - standardGeometry[2::3])**2))
    LLH = np.sum((detector - standardGeometry)**2)
    return LLH
    

def fitDetectorToStandard(detector:np.ndarray, standardGeometry:np.ndarray
                          ) -> np.ndarray:
    """
    Perform the fit to shift and rotate the fitted detector to the standard 
    geometry.

    Parameters
    ----------
    detector : np.ndarray
        Fitted detector.
    standardGeometry : np.ndarray
        Standard geometry .

    Returns
    -------
    res : np.ndarray
        Returns the fitted, rotated and shifted detector.

    """
    result = minimize(LikelihoodRotationShift, (0.,0.,0.,0.,0.,-10.), (detector, standardGeometry), 
                      options={"iprint":0, "ftol":1e-15, "maxiter":1000, "maxfun":maxsize}, method="L-BFGS-B")
    res = result["x"]
    print(f"Fit result: logL = {LikelihoodRotationShift(res, detector, standardGeometry)}, x={res},  message: {result.message}")
    return res

##################
### Evaluation ###
##################

def deltaLLH(result:np.ndarray, standardDetector:np.ndarray, sigma:np.ndarray, 
             measuredDistances:np.ndarray, dZ:float=-10) -> float:
    """
    Calculates the difference of the Likelihood for the fit result to the 
    standard geometry. Prior is False as it is always zero for the standard 
    geometry and is not of interest for a test of the correct calibration!

    Parameters
    ----------
    result : np.ndarray
        Fit result.
    standardDetector : np.ndarray
        standard geometry.
    sigma : np.ndarray
        Individual uncertainty of the distance estimation.
    measuredDistances : np.ndarray
        Measured distances.
    dZ : float, optional
        DOM spacing along the string. The default is -10.

    Returns
    -------
    float
        Delta LLH value to the standard geometry. Gives how much better the fit 
        is than the standard geometry.
    """
    nStrings, nDOM = np.shape(result)[:-1]
    LLHstandard = Likelihood(standardDetector.flatten(), [], measuredDistances, sigma, [], False, nStrings, nDOM, dZ, 0)
    LLHresult = Likelihood(result.flatten(), [], measuredDistances, sigma, [], False, nStrings, nDOM, dZ, 0)
    dLLH = (LLHresult - LLHstandard)
    return dLLH



def significance(dLLH:float) -> float:
    """
    Calculate the significance: How much more significant is the improvement 
    compared to the standard geometry.

    Parameters
    ----------
    dLLH : float
        Delta LLH value between fit and standard geometry.

    Returns
    -------
    float
        Sigma.

    """
    if dLLH < 0:
        sig = np.sqrt(-2*dLLH)
    else:
        sig = -np.sqrt(2*dLLH)
    return sig




def compareToTruth(detector:np.ndarray, trueDetector:np.ndarray) -> tuple:
    """
    Returns mean, minimum, and maximum of the difference between fit and truth.
    Can be done for simuation.
    For real data you cn compare it the standard geometry the meaningfullness 
    is different.

    Parameters
    ----------
    detector : np.ndarray
        Fitted detector.
    trueDetector : np.ndarray
        True detector or detector to compare with.

    Returns
    -------
    tuple
        mean, min, max of the difference.

    """
    nStrings, nDOM = np.shape(detector)[:-1]
    diff = detector - trueDetector
    xyDiff = np.sqrt((diff[:,:,0]**2 + diff[:,:,1]**2).flatten())
    # mean deviation:
    meanDiff = np.mean(xyDiff)
    errDiff = np.std(xyDiff)/np.sqrt(nStrings*nDOM - 1)
    # max deviation
    maxDiff = np.max(xyDiff)
    minDiff = np.min(xyDiff)
    return meanDiff, minDiff, maxDiff
    
def evaluateResult(result:np.ndarray, standardDetector:np.ndarray, 
                   sigma:np.ndarray, measuredDistances:np.ndarray, 
                   trueDetector:np.ndarray=[]) -> tuple:
    """
    Evaluate the fit result

    Parameters
    ----------
    result : np.ndarray
        Fitted detector.
    standardDetector : np.ndarray
        Standard geometry.
    sigma : np.ndarray
        Uncertainties per DOM pair distance estimation.
    measuredDistances : np.ndarray
        Measured distances.
    trueDetector : np.ndarray, optional
        True detector to compare the toy MC. If used for real data set empty 
        list. The default is [].

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    dLLH = deltaLLH(result, standardDetector, sigma, measuredDistances, dZ=-10)
    signi = significance(dLLH)
    if len(trueDetector)!=0:
        meanFit, minFit, maxFit = compareToTruth(result, trueDetector)
        meanStandard, minStandard, maxStandard = compareToTruth(standardDetector, trueDetector)
        dLLHTrue = deltaLLH(result, trueDetector, sigma, measuredDistances, dZ=-10)
        signiTrue = significance(dLLHTrue)
        return dLLH, signi, dLLHTrue, signiTrue, [meanFit, minFit, maxFit], [meanStandard, minStandard, maxStandard]
    return dLLH, signi

