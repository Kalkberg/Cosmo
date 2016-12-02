# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:35:35 2016

@author: pyakovlev
"""
import numpy as np
import matplotlib.pyplot as plot
import scipy
import scipy.stats
import scipy.interpolate

#Create dummy variables
TexR=np.random.normal(10^9,10^6,10000)
RdR=np.random.normal(2,1,10000)
InhR=np.random.normal(10^6,10^5,10000)
EroR=np.random.normal(.0005,.00001,10000)
BestTex=TexR.mean()-TexR.mean()*.1
BestRd=RdR.mean()-RdR.mean()*.1
BestInh=InhR.mean()-InhR.mean()*.1
BestEro=EroR.mean()-EroR.mean()*.1
TexM=np.mean(TexR)
TexMed=np.median(TexR) # Median returns a matrix, so need indices
TexSD=np.std(TexR)
InhM=np.mean(InhR)
InhMed=np.median(InhR)
InhSD=np.std(InhR)
RdM=np.mean(RdR)
RdMed=np.median(RdR)
RdSD=np.std(RdR)
EroM=np.mean(EroR)
EroMed=np.median(EroR)
EroSD=np.std(EroR)

# Define function to create density surface from unequaly spaced data
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

#Make plot 2
plot.title('Kernel Density of Retaind Models')
f, ((ax1, ax2), (ax3, ax4)) = plot.subplots(2,2)
ax1.plot(np.sort(TexR),scipy.stats.gaussian_kde(np.sort(TexR))(np.sort(TexR)))
ax1.locator_params(nbins=4)
#ax1.tick_params(axis='y', pad=20)
ax1.set_xlabel('Exposure Age (ka)')
ax2.plot(np.sort(RdR),scipy.stats.gaussian_kde(np.sort(RdR))(np.sort(RdR)))
ax2.locator_params(nbins=4)
#ax2.tick_params(axis='y', pad=20)
ax2.set_xlabel('Rock Density (g/cm^3)')
ax3.plot(np.sort(InhR/(10^6)),scipy.stats.gaussian_kde(np.sort(InhR))(np.sort(InhR)))
#ax3.tick_params(axis='y', pad=20)
ax3.locator_params(nbins=4)
ax3.set_xlabel('Inheritance\n(Atoms 36Cl/g x 10^6)')
ax4.plot(np.sort(EroR)*1000,scipy.stats.gaussian_kde(np.sort(EroR))(np.sort(EroR)))
#ax4.tick_params(axis='y', pad=20)
ax4.locator_params(nbins=4)
ax4.set_xlabel('Erosion Rate (cm/kyr)')
#plot.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
plot.tight_layout()
plot.savefig('f3.pdf')
plot.close('all')

# Create figure 4
#
plot.title('Crossplots Colored by Density')
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plot.subplots(2, 3, subplot_kw=dict(adjustable='datalim'))

# Create subplot 1
# Plot best fit, mean and median models
ax1.plot(BestTex,BestRd, 'r', marker=(5,1), label='Best Fit') #Plot best fit value as a red star
ax1.plot(TexM,RdM, 'ks', label='Mean') #Plot mean as black square
ax1.plot(TexMed,RdMed, 'co', label='Median') #Plot median as cyan circle
# Calculate point density
density_i, density = DenSurf (TexR,RdR)
# Plot and label interpolated density surface
ax1.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[TexR.min(),TexR.max(),RdR.min(),RdR.max()])
ax1.set_xlim(TexR.min(),TexR.max())
ax1.set_ylim(RdR.min(),RdR.max())
ax1.set_xlabel('Exposure Age (ka)')
ax1.set_ylabel('Rock Density (g/cm^3)')

# Create subplot 2
# Calculate point density
density_i, density = DenSurf (TexR/1000,EroR*1000)
# Plot and label interpolated density surface
ax2.imshow(density_i, vmin=density.min(),vmax=density.max())
# Plot best fit, mean and median models
ax2.plot(BestTex/1000,BestEro*1000, 'r', marker=(5,1)) 
ax2.plot(TexM/1000,EroM, 'ks') 
ax2.plot(TexMed/1000,EroMed, 'co')
ax2.set_xlabel('Exposure Age (ka)')
ax2.set_ylabel('Erosion Rate (cm/kyr)')

# Create subplot 3
# Calculate point density
density_i, density = DenSurf (TexR/1000,InhR/(10^6))
# Plot and label interpolated density surface
ax3.imshow(density_i, vmin=density.min(),vmax=density.max())
# Plot best fit, mean and median models
ax3.plot(BestTex/1000,BestInh/(10^6), 'r', marker=(5,1)) 
ax3.plot(TexM/1000,InhM/(10^6), 'ks') 
ax3.plot(TexMed/1000,InhMed/(10^6), 'co')
ax3.set_xlabel('Exposure Age (ka)')
ax3.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)')

# Create subplot 4
# Calculate point density
density_i, density = DenSurf (RdR,EroR*1000)
# Plot and label interpolated density surface
ax4.imshow(density_i, vmin=density.min(),vmax=density.max())
# Plot best fit, mean and median models
ax4.plot(BestRd,BestEro*1000, 'r', marker=(5,1)) 
ax4.plot(RdM,EroM*1000, 'ks') 
ax4.plot(RdMed,EroMed*1000, 'co')
ax4.set_xlabel('Rock Density (g/cm^3)')
ax4.set_ylabel('Erosion Rate (cm/kyr)')

# Create subplot 5
# Calculate point density
density_i, density = DenSurf (RdR,InhR/(10^6))
# Plot and label interpolated density surface
ax5.imshow(density_i, vmin=density.min(),vmax=density.max())
# Plot best fit, mean and median models
ax5.plot(BestRd,BestInh/(10^6), 'r', marker=(5,1)) 
ax5.plot(RdM,InhM/(10^6), 'ks') 
ax5.plot(RdMed,InhMed/(10^6), 'co')
ax5.set_xlabel('Rock Density (g/cm^3)')
ax5.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)')

# Create subplot 6
# Calculate point density
density_i, density = DenSurf (EroR*1000,InhR/(10^6))
# Plot and label interpolated density surface
ax6.imshow(density_i, vmin=density.min(),vmax=density.max())
# Plot best fit, mean and median models
ax6.plot(BestEro*1000,BestInh/(10^6), 'r', marker=(5,1)) 
ax6.plot(EroM*1000,InhM/(10^6), 'ks') 
ax6.plot(EroMed*1000,InhMed/(10^6), 'co')
ax6.set_xlabel('Erosion Rate (cm/kyr)')
ax6.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)')

#plot.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax5.get_xticklabels(), rotation=30, horizontalalignment='right')
#plot.setp(ax6.get_xticklabels(), rotation=30, horizontalalignment='right')
plot.tight_layout()
#plot.legend(loc=4)
plot.savefig('f4.pdf')
plot.close('all')