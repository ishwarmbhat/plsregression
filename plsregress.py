#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 10:39:45 2018
Filename: pls-decomp-test.py
Purpose: To implement the partial least squares regression algorithm 
discussed in Smoliak et. al (2010)
@author: mudraje
"""

import numpy as np
from scipy.stats import linregress

class pls_decompose:
    
    """Projection to Latent Space (PLS) based decomposition method. 
    Algorithm used is described in Smoliak et.al (2010), Abdi (2010)"""
    
    def __init__(self, X,Y, ncomp = 2):
        
        x_shape = X.shape
        y_shape = Y.shape
        
        ndmx = len(x_shape)
        ndmy = len(y_shape)
        
        ntimex = x_shape[0] # time dimension size
        ntimey = y_shape[0]
        
        # Check for correctness of the inputs
        if(ndmx <= 1 or (not isinstance(X, np.ndarray))):
            raise ValueError("Expect value of X is numpy array of at least 2 dimensions")
        elif(ndmy != 1 or (not isinstance(X, np.ndarray))):
            raise ValueError("Expect value of Y is numpy array of 1 dimension")
        elif(len(X.shape) >= 2):
            print "X has more than 2 dimensions. Reshaping to 2 dimensions"
            X = X.reshape(ntimex, -1)
        
        if(ntimex != ntimey):
            raise ValueError("Time axis does not match between X and Y arrays")
        
        self.X = X
        self.Y = Y        
        self.n_components = ncomp
        self.tlen = ntimex
        self.ngrid = X.shape[1]
        self.isfit = 0
        
        
    def __calculate_corr__(self, X, Y):
        
        """Calculate correlation between 2D X and 1D Y"""
        
        Xbar = X.mean(axis = 0)
        Xstd = X.std(axis = 0)
        
        Ybar = Y.mean()
        Ystd = Y.std()
        
        # time axis lenght - Degrees of freedom of the time series
        tlen = X.shape[0]
        
        Xnorm = X - Xbar
        Ynorm = (Y - Ybar)/tlen
        
        # Dot prodct for covariance and normalize for correlation
        mat_corr = np.dot(Xnorm.T, Ynorm)
        mat_corr = mat_corr/(Xstd*Ystd)
        
        return mat_corr
    
    def __regfunc__(self, a, b):
        
        sl, inc, rsq, pv, sterr = linregress(b, a)
        return sl, inc, rsq, pv, sterr
    
    def __variance_explained__(self, Y, Yhat):
        return 1 - np.var(Y - Yhat)/np.var(Y)
        
    def fit(self):
        
        ncomp = self.n_components
        ngrid = self.ngrid
        ntime = self.tlen
        
        X1 = self.X
        Y1 = self.Y
        
        # Initialize 
        self.Z = np.empty([ncomp, ntime])
        self.BETAX = np.empty([ncomp, ngrid])
        self.INTX = np.empty([ncomp, ngrid])
        self.BETAY = np.empty([ncomp]) # Since there is only one variable in Y
        self.INTY = np.empty([ncomp])
        self.VAREXPY = np.empty([ncomp])
        self.VAREXPX = np.empty([ncomp])
        self.R = np.empty([ncomp, ngrid])
        
        for i in range(ncomp):
            
            # First iteration, correlation between corresponding X and Y
            ri = self.__calculate_corr__(X1,Y1)
            
            # Project input X onto the correlation map ri to get predictor time series
            # shape of correlation map ri - (ngrid x 1)
            # shape of X1 - (ntime, ngrid)
            zi = np.dot(X1, ri) 
            
            # Regress zi against both X (pointwise) and Y
            
            # Against X           
            sl_, intc_, r_, p_, err_ = np.apply_along_axis(self.__regfunc__, 0, X1, zi)
            
            # Against Y
            sl, intc, r, p, err = self.__regfunc__(Y1, zi)
                        
            # Obtain residuals from the regressions
            
            # From X
            zi_re = np.broadcast_to(zi[:, None], [ntime, sl_.shape[0]])
            X1hat = sl_ * zi_re + intc_
            Xtemp = X1 - X1hat
            
            # From Y
            Y1hat = sl * zi + intc
            Ytemp = Y1 - Y1hat
            
            # Store all the results from this iteration
            self.Z[i] = zi
            self.BETAX[i] = sl_
            self.INTX[i] = intc_
            self.BETAY[i] = sl
            self.INTY[i] = intc
            self.VAREXPX[i] = self.__variance_explained__(self.X, X1hat)
            self.VAREXPY[i] = self.__variance_explained__(self.Y, Y1hat)
            self.R[i] = ri
            
            X1 = Xtemp
            Y1 = Ytemp
        
        self.isfit = 1
            
    def predict(self, X):
        
        if(not self.fit):
            raise ValueError("The PLS regression model has not been fitted")
        
        x_shape = X.shape
        
        ndmx = len(x_shape)
        
        ntimex = x_shape[0] # time dimension size
        
        # Check for correctness of the inputs
        if(ndmx <= 1 or (not isinstance(X, np.ndarray))):
            raise ValueError("Expect value of X is numpy array of at least 2 dimensions")
        elif(len(X.shape) >= 2):
            print "X has more than 2 dimensions. Reshaping to 2 dimensions"
            X = X.reshape(ntimex, -1)
        
        ncomp = self.n_components
        
        zibar = np.empty([ncomp, ntimex])
        Yhat = np.zeros(ntimex)
        
        for i in range(ncomp):
            # Load X on each ri
            zibar[i] = np.dot(X, self.R[i])
            yi = zibar[i] * self.BETAY[:,None] + self.INTY[:,None]
            zibar_re = np.broadcast_to(zibar[i][:, None], [ntimex, self.BETAX[i].shape[0]])
            xi = self.BETAX[i] * zibar_re + self.INTX[i]
            Yhat = Yhat + yi
            X = X - xi
        
        return Yhat/np.float32(ncomp)
    
if(__name__ == "__main__"):
    print "Please import and run the module"