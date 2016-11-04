# -*- coding: utf-8 -*-
"""
Code to calculate theoretical 36Cl inventory using outputs of Schimmelpfenning 
et al 2009 spreadsheet calculator.
Written by Petr Yakovlev, Montana Bureau of Mines and Geology
@author Kalkberg

Required Packages:
    Numpy Matplotlib Glob Shutil Scipy

Usage:
    36Cl_Calc.py Data Priors Output PDFLatex_Path
    Data - File name of txt file containing sample information with columns:
        1-36Cl concentration, 2-Error on 36Cl, 3-Sampe depth in cm
    Priors - File name of txt file containing model priors, run parameters, and
        input information derived from outher sources.
    Output - Base file name of the output files desired by the user
    PDFLatex_Path - Path to pdflatex.exe from your LaTeX distribution
    Ex: 36Cl_Calc.py Kumkuli Params Results C:/Program Files/LaTeX

Outputs:
    Output.pdf - PDf of model statistics and charts.
    Retained_Models.txt - Contains models retained during the run, with columns
        1-Exposure Age, 2-Inheritance, 3-Density, 4-Erosion Rate
    Tested_models.txt - Contains models tested during the run, with columns
        1-Exposure Age, 2-Inheritance, 3-Density, 4-Erosion Rate, 5-Model
        likelihood, 6-Acceptance rate at that step

Note: 36_Calc.py must be located in the same directory as your input files!
        
"""
# Import packages
import numpy as np
import matplotlib.pyplot as plot
import os
import glob
import shutil
import sys
import scipy
from argparse import Namespace
import subprocess
import csv

# Translate inputs to variables
Data_File = sys.argv[1]+'.txt'
Prior_File = sys.argv[2]+'.txt'
Out_File = sys.argv[3]
PDFLatex = sys.argv[4]+'pdflatex.exe'

# Check number of input arguments, if not four (Python 5), print error, usage
# and terminate program
if len(sys.argv)==5:
    print("Data file set to %s" % Data_File)
    print("Priors file set to %s" % Prior_File)
    print("Output file set to %s" % Out_File)
else:
    print("Error: Wrong number of input arguments!")
    print("Usage:36Cl_Calc.py Data Priors Output PDFLatex_Path")
    print("Data - File name of txt file containing sample information")
    print("Priors - File name of txt file containing model priors, run parameters, and input information derived from outher sources.")
    print("Output - Base file name of the output files desired by the user")
    print("PDFLatex_Path - Path to pdflatex.exe from your LaTeX distribution")
    print("Ex: 36Cl_Calc.py Kumkuli Params Results C:/Program Files/LaTeX")
    sys.exit()

# Read input files to variables
with open(Data_File,'r') as infile:
    Cl, Clerr, depth = infile.read()
    infile = None
with open(Prior_File,'r') as infile:
   (Run, Ret, Burn, Thin, TexStep, InhStep, RdStep, EroStep, MaxTexTest, 
   MinTexTest, MaxInhTest, MinInhTest, MaxRdTest, MinRdTest, MaxEroTest, 
   MinEroTest, MaxTotE, MinTotE, TexStart, RdStart, InhStart, EroStart, Sn, 
   St, LCl, Nr, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm) = infile.read()
   infile = None

#Pre-allocate matrices
M = np.zeros((Run,6))
RM = np.zeros((Ret+Burn,6))    

# Set counters
s = 1 # Retained models
Ran = 1 # Tested models
q = 0 # Models plotted

PlotPoints=2000 # Set number of points to plot on scatterplots

# Create vector for plotting depth profiles
dp = np.matrix(np.linspace(0,np.max(depth)*1.2,num=201))

# Create vector of steps to take
Steps=np.matrix('%s %s %s %s' %(TexStep, InhStep, RdStep, EroStep))

# Set value of first model
M[0][0:3] = (TexStart,InhStart,RdStart,EroStart)

# Define function to estimate total 36Cl at range of depths defined by vector depth
def ClTot(Tex, Inh, Rd, Ero, LCl, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm, Nr, 
          depth):
    # Time factors with Erosion
    TcosmS = (1-np.exp(-(LCl+Rd*Ero/Af)*Tex))/(LCl+Rd*Ero/Af)
    TcosmEth = (1-np.exp(-(LCl+Rd*Ero/Leth)*Tex))/(LCl+Rd*Ero/Leth)
    TcosmTh = (1-np.exp(-(LCl+Rd*Ero/Af)*Lth))/(LCl+Rd*Ero/Lth)
    TcosmM = (1-np.exp(-(LCl+Rd*Ero/Am)*Tex))/(LCl+Rd*Ero/Am)
    
    # Number of 36Cl atoms produced by each pathway
    Ns = TcosmS*Js*np.exp((-depth*Rd)/Af) # Spallation
    Neth = TcosmEth*Jeth*np.exp((-depth*Rd)/Leth) # Epithermal neutrons
    Nth = TcosmTh*Jth*np.exp((-depth*Rd)/Lth) # Thermal neutrons
    Nm = TcosmM*Jm*np.exp((-depth*Rd)/Am) # Muons
    
    #Total 36Cl
    Ntot=Sn*St*(Ns+Neth+Nth+Nm)+Nr+Inh
    return Ntot
    
# Calculate likelihood of first model and store to model matrix
Ntot=ClTot(M[0,0], M[0,1], M[0,2], M[0,3], LCl, Af, Leth, Lth, Am, Js, Jeth, 
           Jth, Jm, Nr, depth)
L2=np.sum(np.square(Ntot-Cl/Clerr))
M[0,4]=np.exp(-.5*L2)

print("Running Model...")

# Run model
for j in range(1,Run):
    # If the number of retained models meets the required value, end the loop
    if s==(Ret*Thin+Burn):
        break
    # Keep track of how many models were tested
    Ran=Ran+1
    # Select next model to test
    trial=M[j][0:3]+np.matrix(np.random.normal(0,1,[1,4]))*Steps
    # Calculate total 36Cl for new parameters
    Ntot=ClTot(trial[0], trial[1], trial[2], trial[3], LCl, Af, Leth, Lth, Am, 
               Js, Jeth, Jth, Jm, Nr, depth)
    # Calculate L2 norm and likelihood
    L2=np.sum(np.square(Ntot-Cl/Clerr))
    M[j,4]=np.exp(-.5*L2)
    # Calculate ratio of likelihoods between current and previous model
    LRat=M[j,5]/M[j-1,5]
    # Retain models if likelihood ratio is greater than a random number, and 
    # parameters are within limits set by run parameters
    if (LRat > np.random.uniform(0,1) and 
        trial[0] >= MinTexTest and trial[1] <= MaxTexTest and
        trial[2] >= MinInhTest and trial[3] <= MaxInhTest and
        trial[4] >= MinRdTest and trial[5] <= MaxRdTest and
        trial[6] >= MinEroTest and trial[7] <= MaxEroTest and
        trial[0]*trial[3] >= MinTotE and trial[0]*trial[3] <= MaxTotE):
            s=s+1 # Add value to counter of retained models
            M[j][0:3]=trial # Add parameters to list of tested models 
            RM[s][0:3]=trial # Add parameters to list of retained models
            RM[s,4]=M[j,4] # Add likelihood to retained model
    else:
        M[j][0:3]=M[j-1][0:3] # Set coordinates of tested model to prev. value
        M[j,4]=M[j-1,4] # Update likelihood of current model to previous value
    M[j,5]=s/Ran # Record acceptance rate at this step

print("Generating statistics...")   
 
RM=RM[1:Burn,:] # Delete models from burn-in period
RMThin=RM[1::Thin,:] # Thin retained models for statistical analysis
RMThin=RMThin[np.argsort(-RMThin[:,4],0)] # Sort retained models by likelihood

# Split retained models into parameter vectors for easier interpretation
TexR=RMThin[:,0]    # Exposure Time (yr)
InhR=RMThin[:,1]    # Inheritance (atoms 36Cl)
RdR=RMThin[:,2]     # Rock Density (g/cm^3)
EroR=RMThin[:,3]    # Erosion Rate (cm/yr)
LikR=RMThin[:,4]    # Model likelihood

# Thin out model results for scatter plots
RandInd=np.sort(np.transpose(np.matrix(np.random.choice(len(RMThin),PlotPoints,
        replace=False))),0) # Sorted vertical vector of random indices
TexRPThin=np.transpose(np.squeeze(TexR[RandInd])) # Need to make 2D vert. vec.
InhRPThin=np.transpose(np.squeeze(InhR[RandInd]))
RdRPThin=np.transpose(np.squeeze(RdR[RandInd]))
EroRPThin=np.transpose(np.squeeze(EroR[RandInd]))
LikRPThin=np.transpose(np.squeeze(LikR[RandInd]))

# Find parameters of best fit (max likelihood) model
BestTex=np.amax(TexR(np.argmax(LikR))) # Use max of vector in case of repeats
BestInh=np.amax(InhR(np.argmax(LikR)))
BestRd=np.amax(RdR(np.argmax(LikR)))
BestEro=np.amax(EroR(np.argmax(LikR)))

# Calculate statistics of retained models
TexM=np.mean(TexR)
TexMed=np.median(TexR,0)[0,0] # Median returns a matrix, so need indices
TexSD=np.std(TexR)
InhM=np.mean(InhR)
InhMed=np.median(InhR,0)[0,0]
InhSD=np.std(InhR)
RdM=np.mean(RdR)
RdMed=np.median(RdR,0)[0,0]
RdSD=np.std(RdR)
EroM=np.mean(EroR)
EroMed=np.median(EroR,0)[0,0]
EroSD=np.std(EroR)

# Generate normalized likelihood vector and find stats of normalized values
LikNorm=LikR/np.max(LikR)
LikNormPThin=LikRPThin/np.max(LikR)
LikNormMax=np.max(LikNorm)
LikNormMin=np.min(LikNorm)
LikNormMid=(LikNormMax+LikNormMin)/2

print("Writing model output to csv files...")

# Save retained models to a csv
with open(Out_File+'Retained_Models.csv', 'w') as RMcsvfile:
    retainedmodels = csv.writer(RMcsvfile)
    for row in RMThin:
        retainedmodels.writerow(row)
        
# Savetested models to a csv
with open(Out_File+'Tested_Models.csv', 'w') as TMcsvfile:
    testedmodels = csv.writer(TMcsvfile)
    for row in M:
        testedmodels.writerow(row)
        
print("Generating Plots...")

# Generate arguments to plug into LaTeX code
Targs=Namespace(TexM=TexM, TexMed=TexMed, TexSD=TexSD, BestTex=BestTex,
               InhM=InhM, InhMed=InhMed, InhSD=InhSD, BestInh=BestInh,
               RdM=RdM, RdMed=RdMed, RdSD=RdSD, BestRd=BestRd,
               EroM=EroM, EroMed=EroMed, EroSD=EroSD, BestEro=BestEro)

# LaTeX code to generate table
content = r'''\documentclass[english]{article}
\usepackage[latin9]{inputenc}
\makeatletter
\providecommand{\tabularnewline}{\\}
\makeatother
\usepackage{babel}
\begin{document}
\begin{tabular}{|c|c|c|c|c|c|}
\cline{2-6} 
\multicolumn{1}{c|}{} & Mean & Median & StDev & Best Fit & Unit\tabularnewline
\hline 
w & %(TexM)s & %(TexMed)s & %(TexSD)s & %(BestTex)s & ka\tabularnewline
\hline 
x & %(InhM)s & %(InhMed)s & %(InhSD)s & %(BestInh)s & g/cm^3\tabularnewline
\hline 
y & %(RdM)s & %(RdMed)s & %(RdSD)s & %(BestRd)s & atoms/g\tabularnewline
\hline 
z & %(EroM)s & %(EroMed)s & %(EroSD)s & %(BestEro)s & cm/ka\tabularnewline
\hline 
\end{tabular}
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

# Calculate depth profiles for best fit, mean, and median models
NtotBest = ClTot(BestTex, BestInh, BestRd, BestEro, LCl, Af, Leth, Lth, Am, Js, 
               Jeth, Jth, Jm, Nr, dp)
NtotM = ClTot(TexM, InhM, RdM, EroM, LCl, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm, 
            Nr, dp)
NtotMed = ClTot(TexMed, InhMed, RdMed, EroMed, LCl, Af, Leth, Lth, Am, Js, 
                Jeth, Jth, Jm, Nr, dp)

# Pre-allocate matrix for depth profile plot and reset counter
MCl = np.zeros([len(TexRPThin)*len(dp),2]) 

#Set up plot
f1=plot.figure(1)
plot.xlabel('Atoms 36Cl/g')
plot.ylabel('Depth (cm)')
plot.gca().invert_yaxis()
plot.title('Depth Profile Colored by Likelihood')

# Calculate depth profiles for each retained model in set thinned for plotting
for i in range(0,len(TexRPThin)):
    NtotPlot = ClTot(TexRPThin[i], InhRPThin[i], RdRPThin[i], EroRPThin[i],
                     LCl, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm, Nr, dp)
    q = q + 1 # Increase counter by one
    MCl[q*len(dp)-(len(dp)-1):q*len(dp),:] = [np.transpose(NtotPlot),
        np.transpose(dp)] # Add to array of retained values
    if LikNormPThin[i] <= LikNormMid: # Set color to plot based on likelihood
        color = [1, 1 , (200 * ((LikNormMid - LikNormPThin[i])/
                                (LikNormMid - LikNormMin)))/255]
    elif LikNormPThin[i] > LikNormMid:
        color = [(255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # R
                 (255 - (205 * (1 - (LikNormMax - LikNormPThin[i])/
                                (LikNormMax - LikNormMid))))/255, # G
                1] #B
    plot.plot(Ntot,dp, color=color,linewidth=0.5)

plot.plot(NtotM, dp, color='g', linewidth=1.5, label='Mean') # Mean model as green line
plot.plot(NtotMed, dp, color='c', linewidth=1.5, label='Median') # Median model as cyan line
plot.plot(NtotBest, dp, color='r', linewidth=1.5, label='Best Fit') # Best fit model as red line
plot.errorbar(Cl, depth, xerr=Clerr, fmt='bs', markerfacecolor='none', label='data') # Data
plot.legend(loc=4)