import numpy as np
import os
os.chdir(r"C:\Users\nige\DTU Space\PhD\XRR_Fitting_tool")
import pandas as pd
import matplotlib.pyplot as plt
import math 
from differential_evolution import de, visualize_fit
from fresnel import fresnel
from unit_conversion import wavelength2energy, energy2wavelength
from optical_constants import oc
from figure_of_merit import fom
from save_output import save
from datetime import date
from matplotlib.ticker import MaxNLocator
import re

            
# ===============================================================
# Author: Nis C. Gellert
# DTU Space
# Last updated: 13/08/2021
# The following script plots and saves the desired model structure
# ===============================================================

# Define directories
optical_constants_dir = "oc_source/NIST/" # [CXRO, NIST, LLNL]
save_as_txt = [False, "C:/Users/nige/DTU Space/ds0079_BESSY_XFT_model_1p5deg.txt"] # Save fitted data as txt 
save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Multilayer Optimisation SPIE 2022/Python code verification/mot_ptc_bl.txt"] # Save fitted data as txt 

#save_as_png = [True, "Desktop/ds0079_BESSY_XFT_model_0p9deg.txt"] # Save fitted data as txt 

# Define Measurement Setup and Sample Structure
# If angle scan

energy_i = 8.048 # [keV] Incident energy 1.49 and 4.51 keV
#energy_i = 1.487 # [keV] Incident energy 1.49 and 4.51 keV
theta_i = np.arange(0.01,1.5,0.01) # [deg].Incident angle
xrange = theta_i

# If energy scan 
'''
energy_i = np.arange(1.,100.2,0.2) # [keV] Incident energy
theta_i = 0.128 # [deg].Incident angle
xrange = energy_i
'''
a_res = 0.0 # [degree] Instrument Angular resolution Note: Not sure if correct
e_res = 0.0 # [m]. Instrumental spectral resolution. Note: Not implemented
polarization_incident = 0.0 # Incident Polarization [-1:1].
polarization_analyzer = 1.0 # Polarization analyzer sensitivity
ang_off = 0.0 # [degree] Angular offset 
eng_off = 0.0 # [eV] Energy offset 

#           [material,  composition,    z,              sigma,      rho]
#cho_layer = [["C","H","O"],    [1,1,1],  [2],  [0.4], [1.0]] 
lowz_layer =  [["C"], [1], [10.0], [0.4], [2.267]] 
highz_layer = [["Pt"], [1],  [30.0],[0.4], [21.45]]  #
NiV_layer = [["Ni","V"],    [1,1],   [40.5], [0.4],[6.3]] 
#ad_layer =  [["Cr"],      [1],    [5.0], [0.6], [7.2]] 
si_layer = [["Si"],    [1],  [3.9], [0.4],[2.33]] 
substrate= [["Si","O"],    [1,2],   np.NaN, [0.4],[2.65]] 
#substrate = [["Si"],    [1],  np.NaN,    [0.4],[2.33]]


#si_layer =  [["Si"],      [1],    [4], [0.4], [2.33]] 
#pt_layer = [["Pt"],    [1],  [10.0], [0.4],[21.45]] 
#substrate = [["Si"],    [1],  np.NaN,     [0.4],[2.33]]


sample_structure = []
sample_structure = [
                    #lowz_layer,
                    #ad_layer,
                    NiV_layer,
                    #highz_layer,
                    substrate]


# Creates model and calls fresnell
theta_m = theta_i + ang_off # Theta model includes angle off set
energy_m = energy_i + eng_off # Energy model includes energy off set
# Critcal angle : math.degrees(np.sqrt(2*abs(n[1].real-1)))
if np.isscalar(energy_m) and xrange[1]>xrange[0] :
    # limit theta range to [min_theta, max_theta]
    angleScan = True
    energy_m = energy_m*1000
elif np.isscalar(theta_m) and xrange[1]>xrange[0] :
    energy_m = energy_m*1000 # I REMOVED THIS!!!
    angleScan = False

lambda_i = energy2wavelength(energy_m)
number_of_layers = len(sample_structure) # including substrate layer
number_of_atoms =[] 
elements = []
densities = [] 
thickness = [] 
roughness = [] 
for layer in sample_structure:
        # layer =  [material, composition, z, sigma, rho]
        elements.append(layer[0])
        number_of_atoms.append(layer[1])
        thickness.append(layer[2])
        roughness.append(layer[3])
        densities.append(layer[4])

z = thickness[:-1]
wavelength = lambda_i
rho = densities
sigma = roughness
oc_source = optical_constants_dir
theta = theta_m
'''
if angleScan:
    n = np.zeros(len(elements)+1, dtype=np.complex)+1  # + 1 due to top vacuum
    reflectance = np.zeros([len(theta)])
    print("Fitting angle scan")
else:
     # For energy scans, n is computed f or each wavelength and each layer material
     n = np.zeros((len(wavelength),len(elements)+1), dtype=np.complex)+1  # + 1 due to top vacuum
     reflectance = np.zeros([len(wavelength)])
     print("Fitting energy scan")
'''
if angleScan:  
    n = np.zeros(len(elements)+1, dtype=np.complex)+1  # + 1 due to top vacuum
    reflectance = np.zeros([len(theta)])
    print("Computing angle scan")
    for j in range(len(elements)): 
        n[j+1] = oc(wavelength,rho[j][0],number_of_atoms[j],elements[j], oc_source)                        
    reflectance[:] = fresnel(theta,wavelength,n,z,sigma, ares = a_res)        
else:
    n = np.zeros((len(wavelength),len(elements)+1), dtype=np.complex)+1  # + 1 due to top vacuum
    reflectance = np.zeros([len(wavelength)])
    print("Computing energy scan")     
    for j in range(len(elements)):
        n[:,j+1] = oc(wavelength,rho[j][0],number_of_atoms[j],elements[j], oc_source)    
    reflectance[:] = fresnel(theta,wavelength,n,z,sigma, ares = a_res)        
FOM = abs(sum(reflectance[:]))


# sigma symbol
if angleScan:
    fig, (ax3) = plt.subplots(nrows=1, ncols=1,figsize=(6,4.5))
    fig.tight_layout(pad=3.0)
    ax3.plot(theta, reflectance,marker='',linestyle= '-',label ='1.487 keV, Pt single layer',color='dimgray',linewidth = 4.0,markersize=3)
    #ax3.plot(theta, pt30,marker='',linestyle='-',label ='30 nm Pt',color='r',linewidth = 4.0)
    #ax3.plot(theta, pt30cr25p1,marker='',linestyle='solid',label ='2.5 nm Cr + 30 nm Pt',color='k',linewidth = 6.0)
    #ax3.plot(theta, pt30cr50p1,linestyle='--',label ='5.0 nm Cr + 30 nm Pt',color='r',linewidth = 3.0)

    #ax3.plot(theta, ptsi_1p5,linestyle= '--',label ='1.487 keV, Pt single layer',color='grey',linewidth = 4.0,markersize=3)
    #ax3.plot(theta, si_1p5,linestyle= '-',label ='1.487 keV, SiO$_2$ substrate',color='black',linewidth = 4.0,markersize=3)
    #ax3.plot(theta, ptsi_4p5,linestyle= 'dotted',label ='4.511 keV, Pt single layer',color='black',linewidth = 4.0,markersize=3)
    #ax3.plot(theta, si_4p5,linestyle='dashdot',label ='4.511 keV, SiO$_2$ substrate',color='grey',linewidth = 4.0,markersize=3)
    #ax3.plot(theta, reflectance_308,'b-',label ='z=30 nm, \u03C3=0.8 nm',linewidth = 4.0)
    #ax3.plot(theta, reflectance_3012,'m-',label ='z=30 nm, \u03C3=1.2 nm',linewidth = 4.0)
    #ax3.plot(theta, reflectance_244,'c--',label ='z=24 nm, \u03C3=0.4 nm',linewidth = 5.0)
    #ax3.plot(theta, reflectance_364,'k--',label ='z=36 nm, \u03C3=0.4 nm',linewidth = 4.0)
    #plt.text(0.801, 0.35, '4.511 keV, Pt single layer', fontsize = 15)
    #plt.text(4.1, 0.0000061, '1.487 keV', fontsize = 15)
    
    #ax3.plot(theta, reflectance_304,'p-',label ='\u03C3 sub = 2.5 nm')
    ax3.set_ylabel("Reflectance", fontsize=15)
    ax3.set_yscale("log")
    ax3.set_xlabel("Grazing angle (deg)", fontsize=15)

    #extraticks=[0.5,0.9,1.5,2.5]
    #ax3.set_xticks(list(ax3.get_xticks()) + extraticks)
    #ax3.set_xlim(0.,4.0)
    #ax3.set_xlim(0.3,2.6)
    #ax3.set_ylim(10**-5.,1.0)
    #ax3.set_ylim(2*10**-1,6*10**-1)
    ax3.grid()
    ax3.legend(handlelength=6)
    
else:
    fig, (ax3) = plt.subplots(nrows=1, ncols=1,figsize=(11,4.5))
    fig.tight_layout(pad=4.0)
    ax3.plot(energy_m/1000, reflectance,'k*')
    #ax3.plot(energy_m/1000, reflectance_1,'k-',label ='Z Pt = 30.0 nm, Z Cr = 4.0 nm')
    #ax3.plot(energy_m/1000, reflectance_2,'b-',label ='Z Pt = 21.9 nm, Z Cr = 3.9 nm')
    #ax3.plot(energy_m/1000, reflectance_3,'r-',label ='Z Pt = 30.0 nm, Z Cr = 2.0 nm')
    #ax3.plot(energy_m/1000, reflectance_4,'g-',label ='Z Pt = 21.9 nm, Z Cr = 2 nm')
    #ax3.set_xlabel("Energy (keV)")  
    ax3.set_ylabel("Reflectance")    
    ax3.set_xlabel("Incident energy (keV)") 
    #ax3.set_yscale("log")
    #ax3.set_xlim(0,10)
    ax3.legend()
    ax3.grid()



if save_as_txt[0]:   
    if angleScan:  
        file1 = open(save_as_txt[1],"w+") 
        today = date.today()
    
        for i in range(len(theta)):
             file1.write("%.4E %.9E\n" % (theta[i], reflectance[i]))
        file1.close() 
    else:
        file1 = open(save_as_txt[1],"w+") 
        today = date.today()
    
        for i in range(len(energy_m)):
             file1.write("%.4E %.9E\n" % (energy_m[i]/1000, reflectance[i]))
        file1.close()
    
    
    
#%%
