# -*- coding: utf-8 -*-
"""
Testing Plotting code from 36Cl_Clac

@author: pyakovlev
"""
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.lines import Line2D
import scipy
import scipy.stats
import scipy.interpolate

#Create dummy variables
TexR=np.random.lognormal(0,.5,10000)*100
RdR=np.random.normal(2,1,10000)
InhR=np.random.normal(10,10,10000)
EroR=np.random.gamma(5,2,10000)+5
BestTex=TexR.mean()-TexR.mean()*.25
BestRd=RdR.mean()-RdR.mean()*.25
BestInh=InhR.mean()-InhR.mean()*.25
BestEro=EroR.mean()-EroR.mean()*.25
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

# Make plot 2
f, ((ax1, ax2), (ax3, ax4)) = plot.subplots(2,2)
ax1.plot(np.sort(TexR),scipy.stats.gaussian_kde(np.sort(TexR))(np.sort(TexR)))
ax1.locator_params(nbins=3)
ax1.set_xlabel('Exposure Age (ka)')
ax2.plot(np.sort(RdR),scipy.stats.gaussian_kde(np.sort(RdR))(np.sort(RdR)))
ax2.locator_params(nbins=3)
ax2.set_xlabel('Rock Density (g/cm^3)')
ax3.plot(np.sort(InhR),scipy.stats.gaussian_kde(np.sort(InhR))(np.sort(InhR)))
ax3.locator_params(nbins=3)
ax3.set_xlabel('Inheritance\n(Atoms 36Cl/g x 10^6)')
ax4.plot(np.sort(EroR),scipy.stats.gaussian_kde(np.sort(EroR))(np.sort(EroR)))
ax4.locator_params(nbins=3)
ax4.set_xlabel('Erosion Rate (cm/kyr)')
f.suptitle('Kernel Density Functions of Retained Models')
f.subplots_adjust(top=.92)
plot.tight_layout()
plot.savefig('f3.pdf')
plot.close('all')


# Create figure 4
#
plot.title('Crossplots Colored by Density')
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plot.subplots(2, 3, subplot_kw=dict(adjustable='datalim'))

# Create subplot 1
ax1.set_xlim(TexR.min(),TexR.max())
ax1.set_ylim(RdR.min(),RdR.max())
# Calculate point density
density_i, density = DenSurf (TexR,RdR)
# Plot and label interpolated density surface
ax1.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[TexR.min(),TexR.max(),RdR.max(),RdR.min()], aspect='auto')
# Plot best fit, mean and median models
ax1.plot(BestTex,BestRd, 'r', marker=(5,1)) #Plot best fit value as a red star
ax1.plot(TexM,RdM, 'gd') #Plot mean as black square
ax1.plot(TexMed,RdMed, 'co') #Plot median as cyan circle
# Set Axes
ax1.locator_params(nbins=5)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_xlabel('Exposure Age (ka)', fontsize=8)
ax1.set_ylabel('Rock Density (g/cm^3)', fontsize=8)

# Create subplot 2
ax2.set_xlim(TexR.min(),TexR.max())
ax2.set_ylim(EroR.min(),EroR.max())
# Calculate point density
density_i, density = DenSurf (TexR,EroR)
# Plot and label interpolated density surface
ax2.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[TexR.min(),TexR.max(),EroR.max(),EroR.min()], aspect='auto')
# Plot best fit, mean and median models
ax2.plot(BestTex,BestEro, 'r', marker=(5,1)) 
ax2.plot(TexM,EroM, 'gd') 
ax2.plot(TexMed,EroMed, 'co')
# Set Axes
ax2.locator_params(nbins=5)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.set_xlabel('Exposure Age (ka)', fontsize=8)
ax2.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)

# Create subplot 3
# Calculate point density
density_i, density = DenSurf (TexR,InhR)
# Plot and label interpolated density surface
ax3.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[TexR.min(),TexR.max(),InhR.max(),InhR.min()], aspect='auto')
# Plot best fit, mean and median models
ax3.plot(BestTex,BestInh, 'r', marker=(5,1)) 
ax3.plot(TexM,InhM, 'gd') 
ax3.plot(TexMed,InhMed, 'co')
# Set Axes
ax3.set_xlim(TexR.min(),TexR.max())
ax3.set_ylim(InhR.min(),InhR.max())
ax3.locator_params(nbins=5)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.set_xlabel('Exposure Age (ka)', fontsize=8)
ax3.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)

# Create subplot 4
# Calculate point density
density_i, density = DenSurf (RdR,EroR)
# Plot and label interpolated density surface
ax4.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[RdR.min(),RdR.max(),EroR.max(),EroR.min()], aspect='auto')
# Plot best fit, mean and median models
ax4.plot(BestRd,BestEro, 'r', marker=(5,1)) 
ax4.plot(RdM,EroM, 'gd') 
ax4.plot(RdMed,EroMed, 'co')
# Set Axes
ax4.set_xlim(RdR.min(),RdR.max())
ax4.set_ylim(EroR.min(),EroR.max())
ax4.locator_params(nbins=5)
ax4.tick_params(axis='both', which='major', labelsize=8)
ax4.set_xlabel('Rock Density (g/cm^3)', fontsize=8)
ax4.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)

# Create subplot 5
# Calculate point density
density_i, density = DenSurf (RdR,InhR)
# Plot and label interpolated density surface
ax5.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[RdR.min(),RdR.max(),InhR.max(),InhR.min()], aspect='auto')
# Plot best fit, mean and median models
ax5.plot(BestRd,BestInh, 'r', marker=(5,1)) 
ax5.plot(RdM,InhM, 'gd') 
ax5.plot(RdMed,InhMed, 'co')
# Set Axes
ax5.set_xlim(RdR.min(),RdR.max())
ax5.set_ylim(InhR.min(),InhR.max())
ax5.locator_params(nbins=5)
ax5.tick_params(axis='both', which='major', labelsize=8)
ax5.set_xlabel('Rock Density (g/cm^3)', fontsize=8)
ax5.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)

# Create subplot 6
# Calculate point density
density_i, density = DenSurf (EroR,InhR)
# Plot and label interpolated density surface
ax6.imshow(density_i, vmin=density.min(),vmax=density.max(), \
           extent=[EroR.min(),EroR.max(),InhR.max(),InhR.min()], aspect='auto')
# Plot best fit, mean and median models
ax6.plot(BestEro,BestInh, 'r', marker=(5,1)) 
ax6.plot(EroM,InhM, 'gd') 
ax6.plot(EroMed,InhMed, 'co')
# Set Axes
ax6.set_xlim(EroR.min(),EroR.max())
ax6.set_ylim(InhR.min(),InhR.max())
ax6.locator_params(nbins=5)
ax6.tick_params(axis='both', which='major', labelsize=8)
ax6.set_xlabel('Erosion Rate (cm/kyr)', fontsize=8)
ax6.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)
plot.tight_layout()

BF = Line2D([0],[0], linestyle='none', marker=(5,1), markerfacecolor='red')
Mean = Line2D([0],[0], linestyle='none', marker='d', markerfacecolor='green') 
Med = Line2D([0],[0], linestyle='none', marker='o', markerfacecolor='cyan')
f.legend((BF, Mean, Med), ("Best Fit", "Mean", "Median"), 'upper right', 
         numpoints=1, fontsize=6, ncol=3)
f.suptitle('Crossplots Colored by Density', x=.3, y=.98, fontsize=14)
f.subplots_adjust(top=.92)
plot.savefig('f4.pdf')
plot.close('all')