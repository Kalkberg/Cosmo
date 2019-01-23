# -*- coding: utf-8 -*-
"""
Code to calculate theoretical 36Cl inventory using outputs of Schimmelpfenning 
et al 2009 spreadsheet calculator. 
Written by Petr Yakovlev, Montana Bureau of Mines and Geology
@author Kalkberg

Required Packages:
    Numpy Matplotlib Glob Shutil Scipy

Inputs:
    36Cl_Calc.py Data Priors Output PDFLatex_Path
    Data - File name of csv file containing sample information with columns:
        1-36Cl concentration, 2-Error on 36Cl, 3-Sampe depth in cm
    Priors - File name of csv file containing model priors, run parameters, and
        input information derived from outher sources.
    Output - Base file name of the output files desired by the user
    PDFLatex_Path - Path to pdflatex.exe from your LaTeX distribution

Outputs:
    Output.pdf - PDf of model statistics and charts.
    Retained_Models.txt - Contains models retained during the run, with columns
        1-Exposure Age, 2-Inheritance, 3-Density, 4-Erosion Rate
    Tested_models.txt - Contains models tested during the run, with columns
        1-Exposure Age, 2-Inheritance, 3-Density, 4-Erosion Rate, 5-Model
        likelihood, 6-Acceptance rate at that step

Note: 36_Calc.py must be located in the same directory as your input files!
        
"""
#%%
# Import packages
#from __future__ import division
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.lines import Line2D
import os
import scipy
from argparse import Namespace
import subprocess
import csv
from PyPDF2 import PdfFileMerger
import progressbar
import seaborn as sns
import win32com.client as win32
import os


#%%
##
## ONLY CHANGE THESE VALUES
##

# File containing sample data
Data_File = 'West_Data.csv'

# File containing model priors
Prior_File ='Priors_West_SpreadCalc2.csv'

# Schimmelpfenning 2009 spreadsheet with elemental and scaling values 
Spreadsheet ='Schimmelpfennig 2009 Kumkuli West 2019.xlsx'

# Root name of PDF document created by this script
Out_File = 'Priors_West_SpreadCalc2'

# File path to PDFLatex exe
PDFLatex = r'C:\Program Files (x86)\MiKTeX 2.9\miktex\bin\pdflatex.exe'   

#%%
# Read input files TODO: Clean up values not obtained from spreadsheet
Cl = np.genfromtxt(Data_File,delimiter=',',usecols=0)
Clerr = np.genfromtxt(Data_File,delimiter=',',usecols=1)
depth = np.genfromtxt(Data_File,delimiter=',',usecols=2)
(Run, Ret, Burn, Thin, TexStep, InhStep, RdStep, EroStep, 
    MaxTexTest, MinTexTest, MaxInhTest, MinInhTest, MaxRdTest, MinRdTest, 
    MaxEroTest, MinEroTest, MaxTotE, MinTotE, 
    TexStart, RdStart, InhStart, EroStart, 
    Sn, St, LCl, Nr, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm
    ) = np.genfromtxt(Prior_File,delimiter=',').tolist()

# Open spreadsheet and calculate
excel = win32.Dispatch('Excel.Application')
wb = excel.Workbooks.Open(os.getcwd()+'\\'+Spreadsheet)
ws1 = wb.Worksheets(r'sample calculations')
ws2 = wb.Worksheets(r'depth profile')
ws1.EnableCalculation = True
ws2.EnableCalculation = True
ws1.Calculate()
ws2.Calculate()

# Get cosmogenic values that don't depend on sample density
Sn = ws1.Cells(13,4).Value
St = ws1.Cells(15,4).Value
LCl = ws1.Cells(28,4).Value
Nr = ws1.Cells(31,4).Value
Af = ws1.Cells(126,4).Value
Am = ws1.Cells(142,4).Value

# Make sure ints are ints
Run = int(Run)
Ret = int(Ret)
Burn = int(Burn)
Thin = int(Thin)

# Pre-allocate matrices
M = np.zeros((Run,6))
RM = np.zeros((Ret*Thin+Burn+1,6))    
RdVars = np.zeros((int((MaxRdTest-MinRdTest)/RdStep)*10,7))

# Set counters
s = 0 # Retained models
Ran = 0 # Tested models
PlotPoints=3000 # Set number of points to plot on scatterplots

# Create vector for plotting depth profiles
dp = np.linspace(0,np.max(depth)*1.2,num=201)

# Create vector of steps to take
Steps=np.array([TexStep, InhStep, RdStep, EroStep])

# Set value of first model
M[0][0:4] = [TexStart,InhStart,RdStart,EroStart]

# Precalculate table of rock density dependent comogenic values using spreadsheet,
# Need to build table slightly greater than search area, as lin approx will fail otherwise
RdVars[:,0]=np.linspace(MinRdTest*.9,MaxRdTest*1.1,len(RdVars[:,0]))
for d in range(len(RdVars[:,0])):
    dens = RdVars[d,0]
    ws1.Cells(10,4).Value=dens
    ws1.Calculate()
    RdVars[d,1] = ws1.Cells(58,4).Value#Js
    RdVars[d,2] = ws1.Cells(59,4).Value#Jeth
    RdVars[d,3] = ws1.Cells(60,4).Value#Jth
    RdVars[d,4] = ws1.Cells(61,4).Value#Jm
    RdVars[d,5] = ws1.Cells(140,4).Value#Leth
    RdVars[d,6] = ws1.Cells(167,4).Value#Lth
    
# release speadsheet
ws = None
wb = None
excel = None

#%%

# Define function to estimate total 36Cl at range of depths defined by vector depth
def ClTot(Tex, Inh, Rd, Ero, Sn, St, LCl, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm,
          Nr, depth):
    # Time factors with Erosion
    TcosmS = (1-np.exp(-(LCl+Rd*Ero/Af)*Tex))/(LCl+Rd*Ero/Af)
    TcosmEth = (1-np.exp(-(LCl+Rd*Ero/Leth)*Tex))/(LCl+Rd*Ero/Leth)
    TcosmTh = (1-np.exp(-(LCl+Rd*Ero/Lth)*Tex))/(LCl+Rd*Ero/Lth)
    TcosmM = (1-np.exp(-(LCl+Rd*Ero/Am)*Tex))/(LCl+Rd*Ero/Am)
    
    # Number of 36Cl atoms produced by each pathway
    Ns = TcosmS*Js*np.exp((-depth*Rd)/Af) # Spallation
    Neth = TcosmEth*Jeth*np.exp((-depth*Rd)/Leth) # Epithermal neutrons
    Nth = TcosmTh*Jth*np.exp((-depth*Rd)/Lth) # Thermal neutrons
    Nm = TcosmM*Jm*np.exp((-depth*Rd)/Am) # Muons
    
    #Total 36Cl
    Ntot=Sn*St*(Ns+Neth+Nth+Nm)+Nr+Inh
    return Ntot

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

# Define function to find density defined cosmogenic values in table, and return
def RdVarFind(Rd,RdVars):
    # If value actually in list, just use it
    if Rd in RdVars[:, 0]:
        RdRow = np.searchsorted(RdVars[:,0],Rd)
        Js = RdVars[RdRow,1]
        Jeth = RdVars[RdRow,2]
        Jth = RdVars[RdRow,3]
        Jm = RdVars[RdRow,4]
        Leth = RdVars[RdRow,5]
        Lth = RdVars[RdRow,6]

    else:
        # Otherwise interpolate a value
        # This causes computation time to go up by a factor of 5 or so
        Js = np.interp(Rd,RdVars[:,0],RdVars[:,1])
        Jeth = np.interp(Rd,RdVars[:,0],RdVars[:,2])
        Jth = np.interp(Rd,RdVars[:,0],RdVars[:,3])
        Jm = np.interp(Rd,RdVars[:,0],RdVars[:,4])
        Leth = np.interp(Rd,RdVars[:,0],RdVars[:,5])
        Lth = np.interp(Rd,RdVars[:,0],RdVars[:,6])
        
    return Js, Jeth, Jth, Jm, Leth, Lth

# Calculate likelihood of first model and store to model matrix
Js, Jeth, Jth, Jm, Leth, Lth = RdVarFind(M[0,2],RdVars)
Ntot=ClTot(M[0,0], M[0,1], M[0,2], M[0,3], Sn, St, LCl, Af, Leth, Lth, Am, Js, 
           Jeth, Jth, Jm, Nr, depth)
L2=np.sum(np.square(Ntot-Cl)/np.square(Clerr))
#M[0,4] = L2
M[0,4]=np.exp(-.5*L2)
M[0,5]=np.nan

#%%
print("Running Model...")

LRatRec = []
bar = progressbar.ProgressBar(max_value=Ret*Thin+Burn)
# Run model
#
for j in range(1,Run):
    # If the number of retained models meets the required value, end the loop
    if s==(Ret*Thin+Burn):
        break
    bar.update(s)
    # Keep track of how many models were tested
    Ran=Ran+1
    # Select next model to test
    trial=M[j-1][0:4]+np.squeeze(np.random.normal(0,1,[1,4])*Steps)
    
    # Calculate new cosmogenic parameters
    # Need to round density in order to avoid overflow errors
    Js, Jeth, Jth, Jm, Leth, Lth = RdVarFind(trial[2],RdVars)
    # Calculate total 36Cl for new parameters
    Ntot=ClTot(trial[0], trial[1], trial[2], trial[3], Sn, St, LCl, Af, Leth, 
               Lth, Am, Js, Jeth, Jth, Jm, Nr, depth)
    
    # Calculate L2 norm and likelihood
    L2=np.sum(np.square((Ntot-Cl))/np.square(Clerr))
    #M[j,4] = L2
    M[j,4]=float(np.exp(-.5*L2))
    # Calculate ratio of likelihoods between current and previous model
    LRat=M[j,4]/M[j-1,4]
    LRatRec += [LRat]
    # Retain models if likelihood ratio is greater than a random number, and 
    # parameters are within limits set by run parameters
    if (LRat > np.random.uniform(0,1) and 
        trial[0] >= MinTexTest and trial[0] <= MaxTexTest and
        trial[1] >= MinInhTest and trial[1] <= MaxInhTest and
        trial[2] >= MinRdTest and trial[2] <= MaxRdTest and
        trial[3] >= MinEroTest and trial[3] <= MaxEroTest and
        trial[0]*trial[3] >= MinTotE and trial[0]*trial[3] <= MaxTotE):
            s+=1 # Add value to counter of retained models
            M[j][0:4]=trial # Add parameters to list of tested models 
            RM[s][0:4]=trial # Add parameters to list of retained models
            RM[s,4]=M[j,4] # Add likelihood to retained model
    else:
        M[j][0:4]=M[j-1][0:4] # Set coordinates of tested model to prev. value
        M[j,4]=M[j-1,4] # Update likelihood of current model to previous value
    M[j,5]=float(s)/float(Ran) # Record acceptance rate at this step

#%%
print("Generating statistics...")   
RM=RM[0:s,:] # Delete empty rows if retained models do not meet goal
RM=RM[Burn:,:] # Delete models from burn-in period
RMThin=RM[1::Thin,:] # Thin retained models for statistical analysis
RMThin=RMThin[np.argsort(-RMThin[:,4],0)] # Sort retained models by likelihood

# Split retained models into parameter vectors for easier interpretation
TexR=RMThin[:,0]/1000    # Exposure Time (yr) now in kyr
InhR=RMThin[:,1]/(10**6)    # Inheritance (atoms 36Cl) now in atoms x 10^6
RdR=RMThin[:,2]     # Rock Density (g/cm^3)
EroR=RMThin[:,3]*1000    # Erosion Rate (cm/yr) now in cm/kyr
LikR=RMThin[:,4]    # Model likelihood

#%%
# Thin out model results for scatter plots
RandInd=np.sort(np.transpose(np.matrix(np.random.choice(len(RMThin),PlotPoints,
        replace=False))),0) # Sorted vertical vector of random indices
TexRPThin=np.transpose(np.squeeze(TexR[RandInd])) # Need to make 2D vert. vec.
InhRPThin=np.transpose(np.squeeze(InhR[RandInd]))
RdRPThin=np.transpose(np.squeeze(RdR[RandInd]))
EroRPThin=np.transpose(np.squeeze(EroR[RandInd]))
LikRPThin=np.transpose(np.squeeze(LikR[RandInd]))

# Resort thinned values by likelihood so they plot on top
LikSort=LikRPThin.argsort() #
LikRPThin=LikRPThin[LikSort[::1]]
InhRPThin=InhRPThin[LikSort[::1]]
RdRPThin=RdRPThin[LikSort[::1]]
EroRPThin=EroRPThin[LikSort[::1]]

# Find parameters of best fit (max likelihood) model
BestTex=np.amax(TexR[np.argmax(LikR)]) # Use max of vector in case of repeats
BestInh=np.amax(InhR[np.argmax(LikR)])
BestRd=np.amax(RdR[np.argmax(LikR)])
BestEro=np.amax(EroR[np.argmax(LikR)])

# Calculate statistics of retained models
TexM=np.mean(TexR)
TexMed=np.median(TexR)
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

# Generate normalized likelihood vector and find stats of normalized values
LikNorm=LikR/np.max(LikR)
LikNormPThin=LikRPThin/np.max(LikR)
LikNormMax=np.max(LikNorm)
LikNormMin=np.min(LikNorm)
LikNormMid=(LikNormMax+LikNormMin)/2

print("Writing model output to csv files...")

# Save retained models to a csv
with open(Out_File+'_Retained_Models.csv', 'w') as RMcsvfile:
    retainedmodels = csv.writer(RMcsvfile)
    for row in RMThin:
        retainedmodels.writerow(row)
        
## Savetested models to a csv
#with open(Out_File+'_Tested_Models.csv', 'w') as TMcsvfile:
#    testedmodels = csv.writer(TMcsvfile)
#    for row in M:
#        testedmodels.writerow(row)
        
#%%
print("Generating Plots...")

# Create table of model results
#

# Generate arguments to plug into LaTeX code
Targs=Namespace(TexM=TexM, TexMed=TexMed, TexSD=TexSD, BestTex=BestTex,
               InhM='%e' %(InhM*(10**6)), InhMed='%e' %(InhMed*(10**6)), 
               InhSD='%e' %(InhSD*(10**6)), BestInh='%e' %(BestInh*(10**6)),
               RdM=RdM, RdMed=RdMed, RdSD=RdSD, BestRd=BestRd,
               EroM=EroM, EroMed=EroMed, EroSD=EroSD, BestEro=BestEro)

# LaTeX code to generate table
content = r'''\documentclass[english]{article}
\usepackage[latin9]{inputenc}
\usepackage{pdflscape}
\makeatletter
\providecommand{\tabularnewline}{\\}
\makeatother
\usepackage{babel}
\begin{document}
\begin{landscape}
\begin{tabular}{|c|c|c|c|c|c|}
\cline{2-6} 
\multicolumn{1}{c|}{} & Mean & Median & StDev & Best Fit & Unit\tabularnewline
\hline 
Exposure Time & %(TexM)s & %(TexMed)s & %(TexSD)s & %(BestTex)s & ka\tabularnewline
\hline 
Inheritance & %(InhM)s & %(InhMed)s & %(InhSD)s & %(BestInh)s & atoms/g\tabularnewline
\hline 
Soil Density & %(RdM)s & %(RdMed)s & %(RdSD)s & %(BestRd)s & g/cm^3\tabularnewline
\hline 
Erosion Rate & %(EroM)s & %(EroMed)s & %(EroSD)s & %(BestEro)s & cm/ka\tabularnewline
\hline 
\end{tabular}
\end{landscape}
\end{document}
 '''

# Write tex file
with open('table.tex','w') as f:
    f.write(content%Targs.__dict__)

# Convert tex file to PDF - this won't work, try argparser?
cmd = [PDFLatex,'-interaction', 'nonstopmode', 'table.tex']
proc = subprocess.Popen(cmd)
proc.communicate()

# Delete tex file and junk generated during conversion
os.unlink('table.tex')
os.unlink('table.log')
os.unlink('table.aux')

#%%
# Calculate depth profiles for best fit, mean, and median models using original units

Js, Jeth, Jth, Jm, Leth, Lth = RdVarFind(BestRd,RdVars)
NtotBest = ClTot(BestTex*1000, BestInh*(10**6), BestRd, BestEro/1000, Sn, St, LCl, Af, Leth, 
              Lth, Am, Js, Jeth, Jth, Jm, Nr, dp)

Js, Jeth, Jth, Jm, Leth, Lth = RdVarFind(RdM,RdVars)
NtotM = ClTot(TexM*1000, InhM*(10**6), RdM, EroM/1000, Sn, St, LCl, Af, Leth, 
              Lth, Am, Js, Jeth, Jth, Jm, Nr, dp)

Js, Jeth, Jth, Jm, Leth, Lth = RdVarFind(RdMed,RdVars)
NtotMed = ClTot(TexMed*1000, InhMed*(10**6), RdMed, EroMed/1000, Sn, St, LCl, 
                Af, Leth, Lth, Am, Js, Jeth, Jth, Jm, Nr, dp)

#%%
# Create figure 1
#
fig1, ax1 = plot.subplots()

# Pre-allocate matrix for depth profile plot
MCl = np.zeros([len(TexRPThin)*len(dp),2]) 

# Set counter of models plotted
q = 0 

# Calculate depth profiles for each retained model in set thinned for plotting
for i in range(len(TexRPThin)):
    
    ws1.Cells(10,4).Value=RdRPThin[i]
    ws1.Calculate()
    Js = ws1.Cells(58,4).Value#
    Jeth = ws1.Cells(59,4).Value#
    Jth = ws1.Cells(60,4).Value#
    Jm = ws1.Cells(61,4).Value#
    Leth = ws1.Cells(140,4).Value#
    Lth = ws1.Cells(167,4).Value#
    NtotPlot = ClTot(TexRPThin[i]*1000, InhRPThin[i]*(10**6), RdRPThin[i], EroRPThin[i]/1000,
                     Sn, St, LCl, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm, Nr, dp)
    q = q + 1 # Increase counter by one
    MCl[q*len(dp)-(len(dp)):q*len(dp),:] = np.transpose(np.array([NtotPlot, dp])) # Add to array of retained values for density plot
    if LikNormPThin[i] <= LikNormMid: # Set color to plot based on likelihood
        color = [1, 1 , (200 * ((LikNormMid - LikNormPThin[i])/
                                (LikNormMid - LikNormMin)))/255]
    elif LikNormPThin[i] > LikNormMid:
        color = [(255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # R
                 (255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # G
                0] #B
    plot.plot(NtotPlot/(10**6),dp, color=color,linewidth=0.5)

plot.plot(NtotM/(10**6), dp, color='g', linewidth=1.5, 
          label='Mean') # Mean model as green line
plot.plot(NtotMed/(10**6), dp, color='c', linewidth=1.5, 
          label='Median') # Median model as cyan line
plot.plot(NtotBest/(10**6), dp, color='r', linewidth=1.5, 
          label='Best Fit') # Best fit model as red line
plot.errorbar(Cl/(10**6), depth, xerr=2*Clerr/(10**6), 
              fmt='bs', markerfacecolor='white', label='data', zorder=10)
plot.legend(loc=4)

# Add info and save
ax1.set_xlim(0,(np.max(Cl+Clerr)*1.5)/10**6)
ax1.set_ylim(0,np.max(depth))
plot.xlabel('Atoms 36Cl/g x 10^6')
plot.ylabel('Depth (cm)')
plot.gca().invert_yaxis()
plot.title('Depth Profile Colored by Likelihood')
plot.savefig('f1.pdf')
plot.close('all')

#%%
# Create figure 2

# Plot interpolated density surface
fig2, ax2 = plot.subplots()

# Create density plot, making sure origin is included
ax2.hexbin(MCl[:,0]/(10**6),MCl[:,1],cmap='viridis', linewidths=0.1,
           extent=[0,(np.max(Cl+Clerr)*1.5)/10**6,
                   0,np.max(depth)],
           rasterized=True)

# Plot model results and data
plot.plot(NtotM/(10**6), dp, 'r:', linewidth=1.5, label='Mean')
plot.plot(NtotMed/(10**6), dp, 'r--', linewidth=1.5, label='Median')
plot.plot(NtotBest/(10**6), dp, 'r', linewidth=1.5, label='Best Fit')
plot.errorbar(Cl/(10**6), depth, xerr=2*Clerr/(10**6), 
              fmt='bs', markerfacecolor='white', label='data', zorder=10)
plot.legend(loc=4)

# Add info and save
ax2.set_xlim(0,(np.max(Cl+Clerr)*1.5)/10**6)
ax2.set_ylim(0,np.max(depth))
plot.xlabel('Atoms 36Cl/g x 10^6')
plot.ylabel('Depth (cm)')
plot.gca().invert_yaxis()
plot.title('Depth Profile Colored by Density')
plot.savefig('f2.pdf')
plot.close('all')

#%%
# Make figure 3
fig, ((ax1, ax2), (ax3, ax4)) = plot.subplots(2,2)
sns.distplot(TexR,ax=ax1)
ax1.locator_params(nbins=3)
ax1.set_xlabel('Exposure Age (ka)')
sns.distplot(RdR,ax=ax2)
ax2.locator_params(nbins=3)
ax2.set_xlabel('Rock Density (g/cm^3)')
sns.distplot(InhR,ax=ax3)
ax3.locator_params(nbins=3)
ax3.set_xlabel('Inheritance\n(Atoms 36Cl/g x 10^6)')
sns.distplot(EroR,ax=ax4)
ax4.locator_params(nbins=3)
ax4.set_xlabel('Erosion Rate (cm/kyr)')
fig.suptitle('Retained Models')
#f.subplots_adjust(top=.9, bottom=.1)
plot.tight_layout(pad=1.5, w_pad=.5,h_pad=1)
plot.savefig('f3.pdf')
plot.close('all')

#%%
# Create figure 4
#
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plot.subplots(2, 3, subplot_kw=dict(adjustable='datalim'))
# Create subplot 1
ax1.set_xlim(TexR.min(),TexR.max())
ax1.set_ylim(RdR.min(),RdR.max())
# Calculate point density
ax1.hexbin(TexR,RdR,cmap='viridis', linewidths=0.1,rasterized=True)

# Plot best fit, mean and median models
ax1.plot(BestTex,BestRd, 'r', marker=(5,1)) #Plot best fit value as a red star
ax1.plot(TexM,RdM, 'gd') #Plot mean as black square
ax1.plot(TexMed,RdMed, 'bo') #Plot median as cyan circle
# Set Axes
ax1.locator_params(nbins=5)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_xlabel('Exposure Age (ka)', fontsize=8)
ax1.set_ylabel('Rock Density (g/cm^3)', fontsize=8)

# Create subplot 2
ax2.set_xlim(TexR.min(),TexR.max())
ax2.set_ylim(EroR.min(),EroR.max())
# Calculate point density
ax2.hexbin(TexR,EroR, cmap='viridis', linewidths=0.1,rasterized=True)

# Plot best fit, mean and median models
ax2.plot(BestTex,BestEro, 'r', marker=(5,1)) 
ax2.plot(TexM,EroM, 'gd') 
ax2.plot(TexMed,EroMed, 'bo')
# Set Axes
ax2.locator_params(nbins=5)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.set_xlabel('Exposure Age (ka)', fontsize=8)
ax2.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)

# Create subplot 3
# Calculate point density
ax3.hexbin(TexR,InhR, cmap='viridis', linewidths=0.1,rasterized=True)
# Plot best fit, mean and median models
ax3.plot(BestTex,BestInh, 'r', marker=(5,1)) 
ax3.plot(TexM,InhM, 'gd') 
ax3.plot(TexMed,InhMed, 'bo')
# Set Axes
ax3.set_xlim(TexR.min(),TexR.max())
ax3.set_ylim(InhR.min(),InhR.max())
ax3.locator_params(nbins=5)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.set_xlabel('Exposure Age (ka)', fontsize=8)
ax3.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)

# Create subplot 4
# Calculate point density
ax4.hexbin(RdR,EroR, cmap='viridis', linewidths=0.1,rasterized=True)
# Plot best fit, mean and median models
ax4.plot(BestRd,BestEro, 'r', marker=(5,1)) 
ax4.plot(RdM,EroM, 'gd') 
ax4.plot(RdMed,EroMed, 'bo')
# Set Axes
ax4.set_xlim(RdR.min(),RdR.max())
ax4.set_ylim(EroR.min(),EroR.max())
ax4.locator_params(nbins=5)
ax4.tick_params(axis='both', which='major', labelsize=8)
ax4.set_xlabel('Rock Density (g/cm^3)', fontsize=8)
ax4.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)

# Create subplot 5
# Calculate point density
ax5.hexbin(RdR,InhR, cmap='viridis', linewidths=0.1,rasterized=True)
# Plot best fit, mean and median models
ax5.plot(BestRd,BestInh, 'r', marker=(5,1)) 
ax5.plot(RdM,InhM, 'gd') 
ax5.plot(RdMed,InhMed, 'bo')
# Set Axes
ax5.set_xlim(RdR.min(),RdR.max())
ax5.set_ylim(InhR.min(),InhR.max())
ax5.locator_params(nbins=5)
ax5.tick_params(axis='both', which='major', labelsize=8)
ax5.set_xlabel('Rock Density (g/cm^3)', fontsize=8)
ax5.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)

# Create subplot 6
# Calculate point density
ax6.hexbin(EroR,InhR, cmap='viridis', linewidths=0.1,rasterized=True)
# Plot best fit, mean and median models
ax6.plot(BestEro,BestInh, 'r', marker=(5,1)) 
ax6.plot(EroM,InhM, 'gd') 
ax6.plot(EroMed,InhMed, 'bo')
# Set Axes
ax6.set_xlim(EroR.min(),EroR.max())
ax6.set_ylim(InhR.min(),InhR.max())
ax6.locator_params(nbins=5)
ax6.tick_params(axis='both', which='major', labelsize=8)
ax6.set_xlabel('Erosion Rate (cm/kyr)', fontsize=8)
ax6.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)
plot.tight_layout()

# Build legend and save
BF = Line2D([0],[0], linestyle='none', marker=(5,1), markerfacecolor='red')
Mean = Line2D([0],[0], linestyle='none', marker='d', markerfacecolor='green') 
Med = Line2D([0],[0], linestyle='none', marker='o', markerfacecolor='blue')
f.legend((BF, Mean, Med), ("Best Fit", "Mean", "Median"), 'upper right', 
         numpoints=1, fontsize=6, ncol=3)
f.suptitle('Crossplots Colored by Density', x=.3, y=.98, fontsize=14)
f.subplots_adjust(top=.92)
plot.savefig('f4.pdf')
plot.close('all')

#%%
# Create figure 5
#
f, ax1 = plot.subplots()
# Plot and label interpolated density surface
ax1.hexbin(TexR,EroR, cmap='viridis', linewidths=0.1)

# Plot best fit, mean and median models
ax1.plot(BestTex,BestEro, 'r', marker=(5,1)) 
ax1.plot(TexM,EroM, 'gd') 
ax1.plot(TexMed,EroMed, 'bo')
# Set Axes
ax1.set_xlim(TexR.min(),TexR.max())
ax1.set_ylim(EroR.min(),EroR.max())
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_xlabel('Exposure Age (ka)', fontsize=8)
ax1.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)
BF = Line2D([0],[0], linestyle='none', marker=(5,1), markerfacecolor='red')
Mean = Line2D([0],[0], linestyle='none', marker='d', markerfacecolor='green') 
Med = Line2D([0],[0], linestyle='none', marker='o', markerfacecolor='blue')
f.legend((BF, Mean, Med), ("Best Fit", "Mean", "Median"), 'lower left', 
         numpoints=1, fontsize=6, ncol=1)
f.suptitle('Erosion vs Age Crossplot Colored by Density', fontsize=14)
f.subplots_adjust(top=.92)
plot.savefig('f5.pdf')
plot.close('all')

#%%
# Create figure 6
#
plot.title('Crossplots Colored by Likelihood')
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plot.subplots(2, 3, subplot_kw=dict(adjustable='datalim'))

for i in range(0,len(TexRPThin)):
    if LikNormPThin[i] <= LikNormMid: # Set color to plot based on likelihood
        color = [1, 1 , (200 * ((LikNormMid - LikNormPThin[i])/
                                (LikNormMid - LikNormMin)))/255]
        size=2
    elif LikNormPThin[i] > LikNormMid:
        color = [(255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # R
                 (255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # G
                0] #B
        size=3
    ax1.plot(TexRPThin[i], RdRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)
    ax2.plot(TexRPThin[i], EroRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)
    ax3.plot(TexRPThin[i], InhRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)
    ax4.plot(RdRPThin[i], EroRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)
    ax5.plot(RdRPThin[i], InhRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)
    ax6.plot(EroRPThin[i], InhRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)

# Set Axes
ax1.set_xlim(TexR.min(),TexR.max())
ax1.set_ylim(RdR.min(),RdR.max())
ax1.locator_params(nbins=5)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_xlabel('Exposure Age (ka)', fontsize=8)
ax1.set_ylabel('Rock Density (g/cm^3)', fontsize=8)
ax2.set_xlim(TexR.min(),TexR.max())
ax2.set_ylim(EroR.min(),EroR.max())
ax2.locator_params(nbins=5)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.set_xlabel('Exposure Age (ka)', fontsize=8)
ax2.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)
ax3.set_xlim(TexR.min(),TexR.max())
ax3.set_ylim(InhR.min(),InhR.max())
ax3.locator_params(nbins=5)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.set_xlabel('Exposure Age (ka)', fontsize=8)
ax3.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)
ax4.set_xlim(RdR.min(),RdR.max())
ax4.set_ylim(EroR.min(),EroR.max())
ax4.locator_params(nbins=5)
ax4.tick_params(axis='both', which='major', labelsize=8)
ax4.set_xlabel('Rock Density (g/cm^3)', fontsize=8)
ax4.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)
ax5.set_xlim(RdR.min(),RdR.max())
ax5.set_ylim(InhR.min(),InhR.max())
ax5.locator_params(nbins=5)
ax5.tick_params(axis='both', which='major', labelsize=8)
ax5.set_xlabel('Rock Density (g/cm^3)', fontsize=8)
ax5.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)
ax6.set_xlim(EroR.min(),EroR.max())
ax6.set_ylim(InhR.min(),InhR.max())
ax6.locator_params(nbins=5)
ax6.tick_params(axis='both', which='major', labelsize=8)
ax6.set_xlabel('Erosion Rate (cm/kyr)', fontsize=8)
ax6.set_ylabel('Inheritance\n(Atoms 36Cl/g x 10^6)', fontsize=8)

# Plot best fit, mean and median models
ax1.plot(BestTex,BestRd, 'r', marker=(5,1)) #Plot best fit value as a red star
ax1.plot(TexM,RdM, 'gd') #Plot mean as black square
ax1.plot(TexMed,RdMed, 'bo') #Plot median as cyan circle
ax2.plot(BestTex,BestEro, 'r', marker=(5,1)) 
ax2.plot(TexM,EroM, 'gd') 
ax2.plot(TexMed,EroMed, 'bo')
ax3.plot(BestTex,BestInh, 'r', marker=(5,1)) 
ax3.plot(TexM,InhM, 'gd') 
ax3.plot(TexMed,InhMed, 'bo')
ax4.plot(BestRd,BestEro, 'r', marker=(5,1)) 
ax4.plot(RdM,EroM, 'gd') 
ax4.plot(RdMed,EroMed, 'bo')
ax5.plot(BestRd,BestInh, 'r', marker=(5,1)) 
ax5.plot(RdM,InhM, 'gd') 
ax5.plot(RdMed,InhMed, 'bo')
ax6.plot(BestEro,BestInh, 'r', marker=(5,1)) 
ax6.plot(EroM,InhM, 'gd') 
ax6.plot(EroMed,InhMed, 'bo')

# Create legend and print
plot.tight_layout()
BF = Line2D([0],[0], linestyle='none', marker=(5,1), markerfacecolor='red')
Mean = Line2D([0],[0], linestyle='none', marker='d', markerfacecolor='green') 
Med = Line2D([0],[0], linestyle='none', marker='o', markerfacecolor='cyan')
f.legend((BF, Mean, Med), ("Best Fit", "Mean", "Median"), 'upper right', 
         numpoints=1, fontsize=6, ncol=3)
f.suptitle('Crossplots Colored by Likelihood', x=.3, y=.98, fontsize=14)
f.subplots_adjust(top=.92)
plot.savefig('f6.pdf')
plot.close('all')
#%%
# Create Figure 7
#
f, ax1 = plot.subplots()

for i in range(0,len(TexRPThin)):
    if LikNormPThin[i] <= LikNormMid: # Set color to plot based on likelihood
        color = [1, 1 , (200 * ((LikNormMid - LikNormPThin[i])/
                                (LikNormMid - LikNormMin)))/255]
        size=3
    elif LikNormPThin[i] > LikNormMid:
        color = [(255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # R
                 (255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # G
                0] #B
        size=6
    ax1.plot(TexRPThin[i], EroRPThin[i], 'o', markerfacecolor=color, 
             markeredgecolor='none', markersize=size)
    
# Plot best fit, mean and median models
ax1.plot(BestTex,BestEro, 'r', marker=(5,1)) 
ax1.plot(TexM,EroM, 'gd') 
ax1.plot(TexMed,EroMed, 'bo')
# Set Axes
ax1.set_xlim(TexR.min(),TexR.max())
ax1.set_ylim(EroR.min(),EroR.max())
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.set_xlabel('Exposure Age (ka)', fontsize=8)
ax1.set_ylabel('Erosion Rate (cm/kyr)', fontsize=8)
BF = Line2D([0],[0], linestyle='none', marker=(5,1), markerfacecolor='red')
Mean = Line2D([0],[0], linestyle='none', marker='d', markerfacecolor='green') 
Med = Line2D([0],[0], linestyle='none', marker='o', markerfacecolor='cyan')
f.legend((BF, Mean, Med), ("Best Fit", "Mean", "Median"), 'lower left', 
         numpoints=1, fontsize=6, ncol=1)
f.suptitle('Erosion vs Age Crossplot Colored by Likelihood', fontsize=14)
f.subplots_adjust(top=.92)
plot.savefig('f7.pdf')
plot.close('all')

#%%
# Create Figure 8
f, ax = plot.subplots()
ax.plot(M[Burn:Ran,5]*100)
ax.set_xlabel('Trial Number')
ax.set_ylabel('Retention Rate (%)')
f.suptitle('Model Retention Rate', fontsize=14)
plot.tight_layout(pad=1.5, w_pad=.5,h_pad=1)
plot.savefig('f8.pdf')
plot.close('all')
#%%
# Combine PDFs and delete old ones
pdfs = ['table.pdf','f1.pdf', 'f2.pdf', 'f3.pdf', 'f4.pdf', 'f5.pdf', 'f6.pdf', 'f7.pdf', 
        'f8.pdf']
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)
merger.write(Out_File+'.pdf')
os.remove('f1.pdf')
os.remove('f2.pdf')
os.remove('f3.pdf')
os.remove('f4.pdf')
os.remove('f5.pdf')
os.remove('f6.pdf')
os.remove('f7.pdf')
os.remove('f8.pdf')
