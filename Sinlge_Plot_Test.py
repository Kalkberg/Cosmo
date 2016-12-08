# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:21:07 2016

@author: pyakovlev
"""

import numpy as np
import matplotlib.pyplot as plot
import scipy

TexR=np.random.gamma(3,1,10000)*100
EroR=np.random.gamma(5,2,10000)

def DenSurf(x,y):
    # Calculate point density
    location = np.vstack([x,y])
    density = scipy.stats.gaussian_kde(location)(location)
    # Set up a regular grid of interpolation points
    x_i, y_i = np.linspace(x.min(), x.max(), 100), \
    np.linspace(y.min(), y.max(), 100)
    x_i, y_i = np.meshgrid(x_i, y_i)
    # Interpolate over grid
    density_i = scipy.interpolate.griddata((x, y), density,
                (x_i, y_i), method='cubic', fill_value=0)
    return density_i, density
    
# Create plot
plot.xlim(TexR.min(),TexR.max())
plot.ylim(EroR.min(),EroR.max())
# Calculate point density
density_i, density = DenSurf (TexR,EroR)
# Plot and label interpolated density surface
plot.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[TexR.min(),TexR.max(),EroR.max(),EroR.min()], aspect='auto')
# Plot best fit, mean and median models
plot.plot(TexR.mean()-.25*TexR.mean(),EroR.mean()-.25*EroR.mean(), 'r', marker=(5,1)) 
plot.plot(TexR.mean(),EroR.mean(), 'ks') 
plot.plot(np.median(TexR),np.median(EroR), 'co')
# Set Axes
plot.locator_params(nbins=5)
plot.xlabel('Exposure Age (ka)')
plot.ylabel('Erosion Rate (cm/kyr)')