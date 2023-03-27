# -*- coding: utf-8 -*-
"""
@author: Saskia
"""

#######################
### IMPORT PACKAGES ###
#######################

# BASIC PACKAGES
import numpy as np
import pickle as pickle
from time import time
from typing import Optional, Union
import pandas as pd
#import sys, os, glob

# MATPLOLIB
import matplotlib
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
import cycler

# SCIPY
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.stats import lognorm, chisquare, kstest, norm, hmean, gmean
from scipy.optimize import minimize, curve_fit
from scipy.special import gammaln


# SCIKIT-LEARN
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ICECUBE FUNCTIONS
from icecube import icetray, dataio, dataclasses, hdfwriter, phys_services
from icecube import lilliput, gulliver, gulliver_modules, simclasses



#################
### READ DATA ###
#################

def ReadGeometry(file:str="Level2pass2_IC86.2016_data_Run00129004_0101_89_290_GCD.i3.zst",
                 location:str="/data/exp/IceCube/2017/filtered/level2pass2/0101/Run00129004/"
                 ) -> dict:
    """
    Parameters
    ----------
    file : string
        Name of the geometry file to read in. Be aware that the the geometry 
        file is that one, that is used for simulation! The default is 
        "Level2pass2_IC86.2016_data_Run00129004_0101_89_290_GCD.i3.zst"
    location : string, optional
        Location of the file. The default is 
        "/data/exp/IceCube/2017/filtered/level2pass2/0101/Run00129004/".

    Returns
    -------
    dic : dictionary
        Dictionary containing for each receiving DOM time stamps and charge 
        of the hits.
    """
    f = dataio.I3File(f"{location}/{file}")
    for frame in f:
        if frame.keys()[0] == "I3Geometry":
            break
    geo = {}
    geokeys = frame['I3Geometry'].omgeo.keys()
    for i in geokeys:
        geo[str(i)] = []
    
    for i in range(len(geokeys)):
        g = frame['I3Geometry'].omgeo.items()[i][1].position
        geo[str(geokeys[i])].extend([g.x, g.y, g.z])
    return geo



def ReadFlasher_Simulation(file:str, flasher:str, 
                           location:str="/data/user/sphilippen/calibration/PPC/",
                           minNumber:int=1000) -> dict:
    dic = {}
    for string in range(1,87):
        for dom in range(1,61):
            dic["OMKey(%s,%s,0)"%(string,dom)] = [[], [], []]
            
    # asign data to dictionary
    f = dataio.I3File(f"{location}/{file}")
    for frame in f:
        pulses = frame['FlasherShiftedPulses']
        for key,val in pulses.items():
            if str(key).split(",")[0] != flasher.split(",")[0]:
                dic[str(key)][0].extend([PE.time for PE in val])
                dic[str(key)][1].extend([PE.charge for PE in val])   
                dic[str(key)][2].extend([PE.flags for PE in val]) 
                
    # delete keys with less than minNumber hits:
    delKeys = []
    for k in dic.keys():
        if np.sum(dic[k][1]) < minNumber:
            delKeys.append(k)
        else:
            # cast to arrays:
            dic[k] = np.array(dic[k])
    for d in delKeys:
        del dic[d] 
    receiver = np.array(list(dic.keys()))
    arrTime = []
    charge = []
    flag = []
    for r in receiver:
        arrTime.append(dic[r][0])
        charge.append(dic[r][1]) 
        flag.append(dic[r][2]) 
       
    dic = {"flasher":[flasher]*len(arrTime), "receiver":np.array(receiver),
           "time":np.array(arrTime), "charge":np.array(charge), 
           "flag":np.array(flag), "DOMSpacing":DOMSpacing(receiver, flasher)}
    return dic
    
    
def ReadFlasher_MCHits(file:str, flasher:str, 
                           location:str="/data/user/sphilippen/calibration/PPC/",
                           minNumber:int=1000) -> dict:
    """
    Parameters
    ----------
    file : string
        Name of the file to read in; usually in format STRING-DOM.i3.zst.
    flasher : string
        Key of te flashing DOM 
    location : string, optional
        Location of the file. The default is "/data/user/sphilippen/calibration/PPC/".
    minNumber : int
        Minimum number of hits for a receiver DOM to be read in.

    Returns
    -------
    dic : dictionary
        Dictionary containing for each receiving DOM time stamps and charge 
        of the hits. It exclude DOMs deployed on the string of the flashing DOM
        - exclude hole ice.
    """
    # make a empty dictionary:
    dic = {}
    for string in range(1,87):
        for dom in range(1,61):
            dic["OMKey(%s,%s,0)"%(string,dom)] = [[],[]]
            
    # asign data to dictionary
    f = dataio.I3File(f"{location}/{file}")
    for frame in f:
    	mcMap = frame['MCPESeriesMap']
    	for key,val in mcMap.items():
            if str(key).split(",")[0] != flasher.split(",")[0]:
                dic[str(key)][0].extend([mcPE.time for mcPE in val])
                dic[str(key)][1].extend([mcPE.npe for mcPE in val])
            
    # delete keys with less than minNumber hits:
    delKeys = []
    for k in dic.keys():
        if np.sum(dic[k][1]) < minNumber:
            delKeys.append(k)
        else:
            # cast to arrays:
            dic[k] = np.array(dic[k])
    for d in delKeys:
        del dic[d] 
    receiver = np.array(list(dic.keys()))
    arrTime = []
    charge = []
    for r in receiver:
        arrTime.append(dic[r][0])
        charge.append(dic[r][1])
    dic = {"flasher":[flasher]*len(arrTime), "receiver":np.array(receiver),
           "time":np.array(arrTime), "charge":np.array(charge), 
           "DOMSpacing":DOMSpacing(receiver, flasher)}
    return dic
    

def ReadFlasher_Experimental(file:str, flasher:str, location:str, 
                             minNumber:int=1000) -> dict:
    """
    Parameters
    ----------
    file : str
        file name.
    location : str
        location path.

    Returns
    -------
    dict
        dictionary of receiver, arrival times, and corresponding charge to each
        arrival time.

    """
    f = pd.read_hdf(f"{location}/{file}", "FlasherShiftedPulses") 
    receiver = []
    times = []
    charge = []
    for string in range(1,87):
        if string != int(flasher.split("(")[1].split(",")[0]):
            for dom in range(1,61):
                index = np.logical_and(f["string"] == string, f["om"] == dom)
                char = np.array(f["charge"][index])
                if np.sum(char) > minNumber:
                    receiver.append(f"OMKey({string},{dom},0)")
                    times.append(np.array(f["time"][index]))
                    charge.append(char)
    dic = {"flasher":[flasher]*len(times), "receiver":np.array(receiver), 
           "time":np.array(times), "charge":np.array(charge), 
           "DOMSpacing":DOMSpacing(receiver, flasher)}
    return dic


def ReadFlasher(file:str, flasher:str, location:str, sim:bool=False,
                MCPE:bool=False, minNumber:int=1000) -> dict:
    if sim:
        if MCPE:
            dic = ReadFlasher_MCHits(file=file, flasher=flasher, 
                                     location=location, minNumber=minNumber)
        else:
            dic = ReadFlasher_Simulation(file=file, flasher=flasher, 
                                         location=location, minNumber=minNumber)
    else:
        dic = ReadFlasher_Experimental(file=file, flasher=flasher, 
                                     location=location, minNumber=minNumber)
    return dic



#######################
### TIME - DISTANCE ###
#######################

def ToDistance(t:Union[float,np.ndarray]) -> Union[float,np.ndarray]:
    """
    Parameters
    ----------
    t : float or array of float
        Takes time in ns.

    Returns
    -------
    float or array of float
        Return distance in meter. Using speed of light in matter v = c/n with 
        refraction index n = 1.32 for the wavelength spectrum in IceCube and 
        speed of light in vaccum c = 0.299,792,458 m/ns.
    """
    return 0.22711549848*t


def ToTime(d:Union[float,np.ndarray]) -> Union[float,np.ndarray]:
    """
    Parameters
    ----------
    d : float or array of float
        Takes distance in m.

    Returns
    -------
    float or array of float
        Return time in nanosecound. Using speed of light in matter v = c/n with 
        refraction index n = 1.32 for the wavelength spectrum in IceCube and 
        speed of light in vaccum c = 0.299,792,458 m/ns.
    """
    return d/0.22711549848


def DOMSpacing(receiver:dict.keys, flasher:str) -> np.ndarray:
    """
    Parameters
    ----------
    receiver : dict.keys
        Keys of receiving DOMs.
    flasher : str, optional
        Key of the flashing DOM.

    Returns
    -------
    np.ndarray
        DOM spacing.
    """    
    geometry = ReadGeometry()
    f = geometry[flasher]
    d = []
    for r in receiver:
        d.append(np.sqrt(np.sum((np.array(f) - np.array(geometry[r]))**2)))
    return np.array(d)



################################
### PARAMETRIZE LIGHT CURVES ###
################################


def MakeFeatures(arrivalTimes:np.ndarray, charge:np.ndarray, 
                 capTimeRange:bool=False, 
                 quantiles:np.ndarray=np.arange(0.1, 1, 0.1)) -> tuple:
    """
    Parameters
    ----------
    arrivalTimes : np.ndarray
        arrival times of a light curve for a single DOM pair.
    charge : np.ndarray
        charge of photon hits at the corresponding photon arrival times in unit
        of PE.
    quantiles : np.ndarray, optional
        kind of quantiles. 
        The default is array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).

    Returns
    -------
    tuple
        quantiles for full curve, mode, arithmetic mean,  median, 
        standard deviation.
    """
    t0 = performFit(arrivalTimes, charge, func="constant+parabola", 
                    LLHfunc="leastSquare")
    sortInd = arrivalTimes.argsort()
    arrivalTimes =  arrivalTimes[sortInd]
    charge = charge[sortInd]
    cumulative = np.cumsum(charge)/np.sum(charge)
    #cap400ns = np.argmin(arrivalTimes < t0+400)
    #cumulativeCap = np.cumsum(charge[:cap400ns])/np.sum(charge[:cap400ns])
    #quantiles of full curve:
    quant = []
    #quantCap = []
    for i in quantiles:
        quant.append(arrivalTimes[np.argmax(cumulative > i)])
        #quantCap.append(arrivalTimes[np.argmax(cumulativeCap > i)])

    # calculate mode:
    try:
        mode = performFit(arrivalTimes, charge, 
                          "parabola", LLHfunc="leastSquare")
    except:
        mode = 0    
    # mean, meadian, standard deciation:
    #timeCharge = np.repeat(arrivalTimes, charge)
    arithmeticMean = np.mean(arrivalTimes)
    median = np.median(arrivalTimes)
    std = np.std(arrivalTimes)
    return (quant, mode, arithmeticMean, median, std, t0)



def MakeDictionary(dic:dict) -> dict:
    """
    Parameters
    ----------
    dicts : list
        format: "String,DOM".

    Returns
    -------
    dict
        DESCRIPTION.

    """
    features = []
    for d in range(len(dic["receiver"])):
        features.append(MakeFeatures(dic["time"][d], dic["charge"][d]))
    
    q10 = []
    q20 = []
    q30 = []
    q40 = []
    q50 = []
    q60 = []
    q70 = []
    q80 = []
    q90 = []
    q10Cap = []
    q20Cap = []
    q30Cap = []
    q40Cap = []
    q50Cap = []
    q60Cap = []
    q70Cap = []
    q80Cap = []
    q90Cap = []
    mode = []
    mean = []
    median = []
    standard = []
    nPhoton = []
    t0Ges = []
    for d in range(len(dic["receiver"])):
        feature = features[d]
        mode.append(feature[1])
        mean.append(feature[2])
        median.append(feature[3])
        standard.append(feature[4])
        t0 = feature[5]
        t0Ges.append(t0)
        q10.append(feature[0][0] - t0)
        q20.append(feature[0][1] - t0)
        q30.append(feature[0][2] - t0)
        q40.append(feature[0][3] - t0)
        q50.append(feature[0][4] - t0)
        q60.append(feature[0][5] - t0)
        q70.append(feature[0][6] - t0)
        q80.append(feature[0][7] - t0)
        q90.append(feature[0][8] - t0)
        """
        q10Cap.append(feature[6][0] - t0)
        q20Cap.append(feature[6][1] - t0)
        q30Cap.append(feature[6][2] - t0)
        q40Cap.append(feature[6][3] - t0)
        q50Cap.append(feature[6][4] - t0)
        q60Cap.append(feature[6][5] - t0)
        q70Cap.append(feature[6][6] - t0)
        q80Cap.append(feature[6][7] - t0)
        q90Cap.append(feature[6][8] - t0)
        """
        nPhoton.append(np.sum(dic["charge"][d]))
        t0Ges.append(t0)
    dicNew = {"nPhoton":np.array(nPhoton), "q10":np.array(q10), 
              "q20":np.array(q20), "q30":np.array(q30), "q40":np.array(q40), 
              "q50":np.array(q50), "q60":np.array(q60), "q70":np.array(q70),
              "q80":np.array(q80), "q90":np.array(q90), "q10_400ns":np.array(q10Cap), 
              "q20_400ns":np.array(q20Cap), "q30_400ns":np.array(q30Cap), 
              "q40_400ns":np.array(q40Cap), "q50_400ns":np.array(q50Cap), 
              "q60_400ns":np.array(q60Cap), "q70_400ns":np.array(q70Cap),
              "q80_400ns":np.array(q80Cap), "q90_400ns":np.array(q90Cap), 
              "mode":np.array(mode), "mean":np.array(mean), 
              "median":np.array(median), "std":np.array(standard), 
              "t0":np.array(t0Ges)}
    dic.update(dicNew)
    return dic


def CheckFlasher(dic:dict) -> bool:
    """
    Parameters
    ----------
    dic : dict
        Dictionary with arrival times, charge and DOM spacing of the standard 
        geometry.

    Returns
    -------
    bool
        Checks if a single flasher works correctly: E.g. 81-52 is broken.
    """
    diff = ToTime(dic["DOMSpacing"]) - dic["t0"]
    std = np.std(diff)
    deviations = np.sum(diff>0)/len(diff)
    if deviations > 0.2:
        return deviations, std
    else:
        return True


def CombineFlasherSets(files:list, flasher:list, location:str, sim:bool=False,
                       MCPE:bool=False, minNumber:int=1000, 
                       save:Optional[str]=None) -> dict:
    """
    Parameters
    ----------
    dicts : list
        List of dictionaries to be combined in one. Same keys in all needed.
    save : Optional[str], optional
        If given: where to save the dictionary. The default is None.

    Returns
    -------
    dic : dict
        Combined dictionaries.

    """
    dic = ReadFlasher(file=files[0], flasher=flasher[0], location=location, 
                      sim=sim, MCPE=MCPE, minNumber=minNumber)
    dic = MakeDictionary(dic)
    for d in range(1, len(files)):
        dicExt = ReadFlasher(file=files[d], flasher=flasher[d], 
                             location=location, sim=sim, MCPE=MCPE, 
                             minNumber=minNumber)
        dicExt = MakeDictionary(dicExt)
        for k in dic.keys():
            dic[k] = np.concatenate((dic[k], dicExt[k]))


    if save is not None:
        pickle.dump(dic, open(save, "wb"))
    return dic

##################
### Fix Curves ###
##################

def fit(x,y, maximum):
    def expo(fitparameter, x):
        a,b,c,x0 = fitparameter
        return a* np.exp(-b*(x-x0))+c

    def LLHExp(fitparameter, x, y):
        fit = expo(fitparameter, x)
        return LLH(y, fit, LLHfunc="poisson")
    
    result = minimize(LLHExp, (0.1,0.9, 0, maximum), (x, y), 
                      method="L-BFGS-B", options={"iprint":0, "ftol":1e-10, 
                                                  "maxiter":1500,"maxfun":1500, 
                                                  "gtol":1e-7}) 
    return result

def FindGap(heights:np.ndarray) -> bool:
    """
    Parameters
    ----------
    heights : np.ndarray
        heights of the bins.

    Returns
    -------
    bool
        If the light curve has a gab in between that needs to be filled.

    """
    h = heights/max(heights)
    hDiff =  (h[1:] - h[:-1])
    drop = np.argmin(hDiff)-1
    dropMin = np.argmin(h[drop:])
    dropEnd = np.argmax(h[dropMin+drop:]) + dropMin + drop
    print(hDiff[drop+1])
    print(h[dropEnd] - h[dropMin])
    if (hDiff[drop+1] < -0.12) and (h[dropEnd] - h[dropMin] > 0.1):
        return True
    else:
        return False

def FillGap(heights:np.ndarray, kind:str="linear") -> np.ndarray:
    """
    Parameters
    ----------
    heights : np.ndarray
        heights of the bins of the light curve.
    kind : str, optional
        "exp" for an expnential fit or "linear" for a linear interpolation. 
        The default is "linear".

    Returns
    -------
    TYPE
        Returns the complete light curve where the gap is filled.

    """
    def expo(fitparameter, x):
        a,b,c,x0 = fitparameter
        return a* np.exp(-b*(x-x0))+c

    def LLHExp(fitparameter, x, y):
        fit = expo(fitparameter, x)
        return LLH(y, fit, LLHfunc="leastSquare")
    
    
    h = heights/max(heights)
    hDiff =  (h[1:] - h[:-1])
    maximum = np.argmax(heights)
    drop = np.argmin(hDiff)-1
    dropMin = np.argmin(h[drop:])
    dropEnd = np.argmax(h[dropMin+drop:]) + dropMin + drop
    
    x = np.arange(0, len(h))
    if kind == "linear":
        hCut = np.concatenate((heights[:drop], heights[dropEnd:]))
        xCut = np.concatenate((x[:drop], x[dropEnd:]))
        interpolation = interp1d(xCut, hCut, kind="linear")
        new = interpolation(x)
    elif kind == "exponential":
        hCut = np.concatenate((heights[drop-5:drop], heights[dropEnd:dropEnd+10]))
        xCut = np.concatenate((x[drop-5:drop], x[dropEnd:dropEnd+10]))
        result = minimize(LLHExp, (heights[drop],0.1, heights[dropEnd], maximum), (xCut, hCut), 
                          method="L-BFGS-B", 
                          options={"iprint":0, "ftol":1e-10, "maxiter":1500,
                                   "maxfun":1500, "gtol":1e-7}) 
        gap = expo(result["x"], np.arange(drop, dropEnd))
        
        
        plt.plot(expo(result["x"], np.arange(drop-15, dropEnd+15)))
        new = np.concatenate((heights[:drop], gap, heights[dropEnd:]))
        print(result)
    return new




#################
### Histogram ###
#################

def Histogram(hits:np.ndarray, charge:np.ndarray, 
              binWidth:Optional[float]=None) -> tuple:
    """
    Parameters
    ----------
    hits : array of float
        DESCRIPTION.
    charge : array of int
        Measured charge of each hit. Used as weight in the histogram.
    binWidth : float
        Width of the bins in ns. If None, an adaptive binning is done based on
        the number of hits. Default is None.

    Returns
    -------
    hight : array of int
        Height of histograms bins.
    bins : arrayy of float
        Center position of bins.
    """   
    if binWidth == None:
        n = len(hits*charge)
        if   n >150000:                binWidth =  0.1
        elif n >100000 and n <=150000: binWidth =  0.25
        elif n > 80000 and n <=100000: binWidth =  0.5
        elif n > 50000 and n <= 80000: binWidth =  1
        elif n > 30000 and n <= 50000: binWidth =  2
        elif n > 10000 and n <= 30000: binWidth =  3
        elif n >  8000 and n <= 10000: binWidth =  6
        elif n >  5000 and n <=  8000: binWidth =  9
        elif n >  2000 and n <=  5000: binWidth = 15
        elif n >  1000 and n <=  2000: binWidth = 25
        elif n <= 1000:                binWidth = 40        
    bins = np.arange(0, 5000, binWidth)
    height, bins = np.histogram(hits, bins=bins, weights=charge)
    bins = (bins[1:]+bins[:-1])*0.5
    return height, bins

############
### FITs ###
############

def LLH(exp:np.ndarray, fit:np.ndarray, LLHfunc:str="poisson") -> float:
    """
    Parameters
    ----------
    exp : array of ints
        "Histogram of experimental data."
    fit : array of floats
        "Fit function evaluated on the same x position as the expermiental histogram."
    LLHfunc : string, optional
        "LLH function used for minimizaton. The default is 'poisson'."
        
    Returns
    -------
    float
        "Value raised from the differnce of experimental histogram an evaluated 
        fit function, calculated with corresponding LLH function.""
    """
    if LLHfunc == "poisson":
        return -np.sum(np.ma.masked_invalid(exp*np.log(fit) - gammaln(exp + 1.) - fit))
    elif LLHfunc == "leastSquare":
        return np.sum((fit - exp)**2)
    else:
        raise Exception("funcLLH must be either poisson or leastSquare")


def Parabola(fitparameter:tuple, x:np.ndarray, func:str, setToZero:bool=True
             ) -> tuple:
    """
    Parameters
    ----------
    fitparameter : tuple
        Varaibles of the function:
        vertex,width,y0 for parabola
        vertex, width,(y0) for constant+parabola (y0 if setToZero=True)
    x : np.ndarray
        x values.
    func : str
        options are parabola for the fit of the mode or 
        constant+parabola for the fit of t0.
    setToZero : bool, optional
        for constant+parabola fit if the . The default is True.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    tuple
        return the y values for coresponding function.
    """
    if func == "parabola":
        vertex,width,y0 = fitparameter
        fit = -width*(x - vertex)**2 + y0
    elif func == "constant+parabola":
        if setToZero:
            vertex,width = fitparameter
            linMask = x < vertex
            lin = 0*x[linMask]
            par = width*(x[~linMask] - vertex)**2
            fit = np.concatenate((lin,par))
        else:
            vertex,width,y0 = fitparameter
            linMask = x < vertex
            lin = 0*x[linMask] + y0
            par = width*(x[~linMask] - vertex)**2 + y0
            fit = np.concatenate((lin,par))
    else:
        raise Exception("func is parabola or constant+parabola")
    return fit


def LLHValue(fitparameter:tuple, x:np.ndarray, y:np.ndarray, func:str, LLHfunc:str,
             setToZero:str) -> float:
    """
    Parameters
    ----------
    fitparameter : tuple
        DESCRIPTION.
    x : np.ndarray
        DESCRIPTION.
    y : np.ndarray
        DESCRIPTION.
    func : str
        DESCRIPTION.
    LLHfunc : str
        DESCRIPTION.
    setToZero : str
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    fit = Parabola(fitparameter, x, func, setToZero)
    return LLH(y, fit, LLHfunc=LLHfunc)



def FirstGuess(x:np.ndarray, y:np.ndarray, func:str, setToZero:bool=True
               ) -> tuple:
    """
    Parameters
    ----------
    x : array of float
        Center of bins of the histogram.
    y : array of int
        Height of the histogram.
    func : string, optional
        The function to fit to the histogram. 
        Options: "parabola" or "constant+parabola"
    setToZero : bool, optional
        For parabola and linear+parabola. If True the linear and parabolas 
        apex is fixed to zero. The default is False.

    Returns
    -------
    firstGuess : tuple
        Returns the first guess for the fit.
    """
    if func == "constant+parabola":
        vertex = x[np.argmax(y>2)]
        width = -(y[np.argmax(y>2)] - y[np.argmax(y>2) + 20]) / (vertex - x[np.argmax(y>2)+ 20])**2
        if setToZero:
            firstGuess = (vertex, width)
        else:
            firstGuess = (vertex, width, 0)
    elif func == "parabola":
        indexMax = np.nanargmax(y) 
        index = np.where(y > y[indexMax]*0.5)[0]
        vertex = x[indexMax]
        y0 = np.max(y)
        width = -(y[index[0]] - y[indexMax]) / (x[index[0]] - x[indexMax])**2
        firstGuess = (vertex, width, y0)
    else:
        raise KeyError("func must be constant+parabola or parabola")
    return firstGuess
   

def performFit(arrivalTime:np.ndarray, charge:np.ndarray, func:str,
               LLHfunc:str="leastSquare", setToZero:bool=True) -> tuple:
    h,b = Histogram(arrivalTime, charge)#, binWidth=25)
    fg = FirstGuess(b, h, func, setToZero=setToZero)
    
    # cut fitrange for parabola fits:
    if func == "parabola":
        index = np.where(h > h[np.nanargmax(h)]*0.5)[0]
        h = h[index[0]:index[-1]]
        b = b[index[0]:index[-1]]
    elif func == "constant+parabola":
        maxIndex = np.argmax(h >= np.max(h)*0.4)
        h = h[:maxIndex]
        b = b[:maxIndex]
        
    result = minimize(LLHValue, fg, (b, h, func, LLHfunc, setToZero), 
                      method="L-BFGS-B", options={"iprint":0, "ftol":1e-10, 
                                                  "maxiter":1500,"maxfun":1500, 
                                                  "gtol":1e-7})                          
    return result["x"][0]



    

####################
### RandomForest ###
####################

def trainRandomForrest(dicts:pd.DataFrame, features:list, test_size:float=0.4, 
                       capMaxDistance:Optional[float]=None):
    X = dicts[features]
    y = dicts['DOMSpacing']  # Truth
    if capMaxDistance is not None:
        ind = y < capMaxDistance
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(X_train, y_train, sample_weight=1/y_train)
    return clf, (X_train, X_test, y_train, y_test)
    

#######################
### GOODNESS OF FIT ###
#######################

def chi2_pValue(exp:np.ndarray, fit:np.ndarray, b:list=[], 
                 xMax:Optional[float]=None) -> float:
    """
    Parameters
    ----------
    exp : np.ndarray
        DESCRIPTION.
    fit : np.ndarray
        DESCRIPTION.
    b : list, optional
        DESCRIPTION. The default is [].
    xMax : Optional[float], optional
        DESCRIPTION. The default is None.

    Returns
    -------
    float
        p-value determined using the chi-square method.

    """
    if xMax is None:
        return chisquare(exp[fit>=3], fit[fit>=3])[1]*100
    else:
        index = b <= xMax
        return chisquare(exp[index][fit[index]>=3], fit[index][fit[index]>=3])[1]*100




def KS_pValue(rawData:np.ndarray, fitParameter:tuple, func:str="logNormal", 
              xMax:Optional[float]=None) -> float:
    """
    Parameters
    ----------
    rawData : np.ndarray
        Raw travel time data.
    fitParameter : tuple
        For logNormal: (sigma, x0, w, h), which were determined by the fit.
    func : str, optional
        Fitted function to be evaluated. The default is "logNormal".
    xMax : Optinal[float], optional
        If the fit should be evaluated just up to a certain point. The default is None.

    Returns
    -------
    float
        p-value determined using the Kolmogorow-Smirnow test.
    """
    if func == "logNormal":
        sigma, x0, w, h = fitParameter
        fitfit = lognorm(s=sigma, loc=x0, scale=w)
        if xMax is None:
            return kstest(rawData, fitfit.cdf)[1]*100
        cdfMax = fitfit.cdf(xMax)
        def myCDF(x):
            return fitfit.cdf(x) / cdfMax
        return kstest(rawData[rawData < xMax], myCDF)[1]*100
    """
    else:
        if func == "parabola":
            fitfit = 
            
            or func == "linear+parabola" or func == "linear":
    """
    

    
    
############
### PLOT ###
############

def linearFunc(x, a, b):
    return a*x + b

def expFunc(x, a, b, c):
    return np.exp(-a*x) + b

def Correlations(dic:dict, parameter:str, plot:bool=True, 
                 save:Optional[str]=None, chi:Optional[float]=None):
    if parameter == "sigma":
        label = r"$\sigma$"
        y = dic[parameter]
    elif parameter == "normed_h":
        label = r"$\frac{h}{\mathrm{ns}}$"
        y = dic["h"]/dic["binWidth"]
    elif parameter == "nPhoton":
        label = r"$n_\gamma$"
        y = dic[parameter]
    else:
        label = parameter
        y = dic[parameter]
    if plot:
        if chi is None:
            suc = dic["sigma"] < 0.5
        else:
            suc = np.logical_and(dic["sigma"] < 0.5, dic["chi"]>chi)
        fig,(ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3.5))
        x = (ToDistance(dic["t0_parabola"]) - dic["d"])[suc]
        popt, pcov = curve_fit(linearFunc, x, y[suc])
        ax1.plot(x, y[suc], ".")
        ax1.plot(range(0, 80), linearFunc(range(0, 80), *popt), color="C0")
        ax1.text(0.05,0.95, 
                 r"$(%.2e\pm %.2e)\cdot x + (%.2e\pm %.2e)$"%(popt[0], pcov[0,0], popt[1], pcov[1,1]),
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax1.transAxes)
        x1 = (ToDistance(dic["t0_parabola"]) - (dic["sigma"] - popt[1])/popt[0])[suc]

        
        x = (ToDistance(dic["t0_parabola"])-dic["d"])[suc]/dic["d"][suc]
        popt, pcov = curve_fit(linearFunc, x, y[suc])
        ax2.plot(x, y[suc], ".")
        ax2.plot(np.arange(-0.1, 0.3, 0.1), linearFunc(np.arange(-0.1, 0.3, 0.1), *popt), color="C0")
        ax2.text(1.15,0.95, 
                 r"$(%.2e\pm %.2e)\cdot x + (%.2e\pm %.2e)$"%(popt[0], pcov[0,0], popt[1], pcov[1,1]),
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax1.transAxes)
        
        x = ToDistance(dic["t0_parabola"])[suc]
        poptF, pcovF = curve_fit(linearFunc, x, y[suc])
        ax3.plot(x, y[suc], ".", label="fit", color="C0")
        ax3.plot(range(0,500), linearFunc(range(0,500), *poptF), color="C0")
        ax3.text(2.25,0.95, 
                 r"$(%.2e\pm %.2e)\cdot x + (%.2e\pm %.2e)$"%(poptF[0], pcovF[0,0], poptF[1], pcovF[1,1]),
                 horizontalalignmentalignment='left', verticalalignment='top',
                 transform=ax1.transAxes)
        
        
        x = dic["d"][suc]
        poptT, pcovT = curve_fit(linearFunc, x, y[suc])
        ax3.plot(x, y[suc], ".", label="truth", color="C1")
        ax3.plot(range(0,500), linearFunc(range(0,500), *poptT), color="C1")
        ax3.text(2.25,0.87, 
                 r"$(%.2e\pm %.2e)\cdot x + (%.2e\pm %.2e)$"%(poptT[0], pcovT[0,0], poptT[1], pcovT[1,1]),
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax1.transAxes)
        x3 = ((dic["sigma"] - poptT[1])/poptT[0])[suc]
        
        
        
        ax1.set_xlim(0, 80)
        ax2.set_xlim(-0.1, 0.3)
        ax3.set_xlim(0,500)
        ax3.legend(loc="lower left")
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlabel(r"distance difference $d_\mathrm{fit}-d_\mathrm{true}$ in m")
        ax2.set_xlabel(r"realtive distance difference $\frac{d_\mathrm{fit}-d_\mathrm{true}}{d_\mathrm{true}}$ in m")
        ax3.set_xlabel("DOM spacing in m")
        ax1.set_ylabel(label)
        plt.tight_layout()
        plt.show()
        if save is not None:
            fig.savefig("%s-%s.png"%(parameter,save), dpi=256)
        return dic["d"][suc], ToDistance(dic["t0_parabola"][suc]), x1, x3

def CorrelationMatrix(dic:dict, chi:Optional[float]=None, save:Optional[str]=None):
    if chi is None:
        suc = dic["sigma"] < 0.5
    else:
        suc = np.logical_and(dic["sigma"] < 0.5, dic["chi"]>chi)
    delta = (ToDistance(dic["t0_parabola"]) - dic["d"])[suc]
    arr = np.array([dic["sigma"][suc], dic["w"][suc], (dic["h"]/dic["binWidth"])[suc],
                    dic["t0_parabola"][suc], dic["nPhoton"][suc], delta,
                    delta/dic["d"][suc], dic["d"][suc]])
    corr = np.corrcoef(arr)
    fig = plt.figure(figsize=(5,4))
    plt.pcolormesh(corr, vmin=-1, vmax=1, cmap="RdBu")
    locs, labels = plt.yticks()
    ticks = [r"$\sigma$", r"$w$", r"$h/$ns", r"$t_0$", r"$n_\gamma$", 
             r"$\Delta x$/m", r"$\Delta x/x$", r"$x_\mathrm{true}$/m"]
    plt.xticks(np.arange(0.5, 8.5), ticks, rotation=90)
    plt.yticks(np.arange(0.5, 8.5), ticks)
    plt.xlim(0,8)
    plt.ylim(0,8)
    plt.colorbar()
    plt.tight_layout()
    if save is not None:
        fig.savefig("%s.png"%save)


