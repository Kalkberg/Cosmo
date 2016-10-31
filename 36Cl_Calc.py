# -*- coding: utf-8 -*-
"""
Code to calculate theoretical 36Cl inventory using outputs of Schimmelpfenning 
et al 2009 spreadsheet calculator.
Written by Petr Yakovlev, Montana Bureau of Mines and Geology
@author Kalkberg

Required Packages:
    Numpy Matplotlib Glob Shutil

Usage:
    36Cl_Calc.py Data Priors Output
    Data - File name of txt file containing sample information with columns:
        1-36Cl concentration, 2-Error on 36Cl, 3-Sampe depth in cm
    Priors - File name of txt file containing model priors, run parameters, and
        input information derived from outher sources.
    Output - File name of the output PDF file desidred by the user
    Ex: 36Cl_Calc.py Kumkuli Params Results

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
import tarfile
import shutil
import sys

# Translate inputs to variables
Data_File = sys.argv[1]+'.txt'
Prior_File = sys.argv[2]+'.txt'
Out_File = sys.argv[3]+'.txt'

# Check number of input arguments, if not three (Python 4), print error, usage
# and terminate program
if len(sys.argv)==4:
    print("Data file set to %s" % Data_File)
    print("Priors file set to %s" % Prior_File)
    print("Output file set to %s" % Out_File)
else:
    print("Error: Wrong number of input arguments!")
    print("Usage:36Cl_Calc.py Data Priors Output")
    print("Data - File name of txt file containing sample information")
    print("Priors - File name of txt file containing model priors, run parameters, and input information derived from outher sources.")
    print("Output - File name of the output PDF file desidred by the user")
    print("Ex: 36Cl_Calc.py Kumkuli Params Results")
    sys.exit()

# Read input files to variables
with open(Data_File,'r') as infile:
    Cl, Clerr, depth = infile.read()
    infile = None
with open(Prior_File,'r') as infile:
    Run, Ret, Burn, Thin, TexStep, InhStep, RdStep, EroStep, MaxTexTest, 
    MinTexTest, MaxInhTest, MinInhTest, MaxRdTest, MinRdTest, MaxEroTest, 
    MinEroTest, MaxTotE, MinTotE, TexStart, RdStart, InhStart, EroStart, Sn, 
    St, LCl, Nr, Af, Leth, Lth, Am, Js, Jeth, Jth, Jm = infile.read()
    infile = None

#Pre-allocate matrices
M=np.zeros((Run,6))
RM=np.zeros((Ret+Burn,6))    

# Set value of first model
M[1,1:4]=(TexStart,InhStart,RdStart,EroStart)

# Set counters
s=1 # Retained models
Ran=1 # Tested models