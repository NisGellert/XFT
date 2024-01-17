import numpy as np
import os
os.chdir(r"C:\Users\nige\DTU Space\PhD\XRR_Fitting_tool")
import pandas as pd
import matplotlib.pyplot as plt
from differential_evolution import de, visualize_fit
from fresnel import fresnel
from unit_conversion import wavelength2energy, energy2wavelength
from optical_constants import oc
from figure_of_merit import fom
from save_output import save
from datetime import date
from matplotlib.ticker import MaxNLocator
import sys
            
# ===============================================================
# Author: Nis C. Gellert
# DTU Space
# Last updated: 13/08/2021
# The following script plots and saves the desired multilayer model structure
# ===============================================================


optical_constants_dir = "oc_source/NIST/" # [CXRO, NIST, LLNL]
#save_as_txt = [False, "Data/Simulated Data/NuSTAR_XFT_layer_1_NIST.txt"] # Save fitted data as txt 
#save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Multilayer Optimisation SPIE 2022/Python code verification/recipe_5_powerlaw.txt"] # Save fitted data as txt 

# -- Define Measurement Setup --
#energy_i = 8.048 # [keV] Incident energy
#theta_i = np.arange(0,15.,0.01) # [deg] Incident angle

energy_i = np.arange(1.,200.2,0.2) # [keV] Incident energy
theta_i = 0.128 # [deg].Incident angle

a_res = 0.0 # [degree] Instrument Angular resolution Note: Not sure if correct
e_res = 0.0 # [m]. Instrumental spectral resolution. Note: Not implemented
polarization_incident = 0.0 # Incident Polarization [-1:1].
polarization_analyzer = 1.0 # Polarization analyzer sensitivity
ang_off = 0.0 # [degree] Angular offset 
eng_off = 0.0 # [keV] Energy offset 


# -- Define Model Space -- 
#             [material, composition, roughness[nm], density [g/cm^-3]]
lowz_layer =  [["C"], [1], [0.45], [2.267]] 
#lowz_layer =  [["Pt"], [1], [0.45], [2.33]] 
#lowz_layer =  [["SiC"], [1,1], [0.45], [3.21]] 

#highz_layer = [["Ni", "V"], [1,1], [0.45], [8.9]]  #Ni = 8.8, V = 6.11, NiO = 6.67, NiV=8,9
highz_layer = [["Pt"], [1], [0.45], [21.45]]  #Ni = 8.8, V = 6.11, NiO = 6.67 
substrate =   [["Si", "O"], [1,2], [0.45], [2.65]] 
#substrate =   [["Si"], [1], [0.45], [2.33]] 




N = 10 # Number of bi-layers 
Structure = "Power law" # "Power law", "Linear" or "Periodic"
dmin = 2.9 # [nm] Minimum bi-layer thickness
dmax = 10.95# [nm] Maximum bi-layer thickness. If Periodic structure, d=dmax.
c = 0.225 # Power law variable
gamma = 0.45 # Thickness ratio. We use Gamma high-Z. Gamma high-Z = z high-Z / d 
gamma_top = [True,0.7] 

# -- Define differentual evolution parameters --
#mutation_factor = 0.5 # [0.1:2.0]
#crossover_probability = 0.75 # [0:1]
#population_size = 30 # [10:50]
#iterations = 10 # [25 - 550] 
#mutation_scheme = "Best/1" # "Rand/1" or "Best/1"

plotFit = True # plots the best fit in every iteration
#logFitting = False
#weightsFitting = 'equal' #

# --- Define model space ---






# --- Flags ---
if N < 2:
    print("Flag: N must >= 2")
    sys.exit()
   
    
    

# -- Creates Sample Structure -- 
d = np.zeros(N) # Initialiser
z = np.zeros((N*2)) # Initialiser
if Structure == 'Power law': 
    b=np.exp((np.log(-(N-1)/(-1+np.exp(np.log(dmin/dmax)/c)))*c+np.log(dmin/dmax))/c) # Power law variable
    a=dmax*b**c # Power law variable
    for i in range(N): 
        d[i]=a/(b+i)**c # Bi-layer thickness
        z[2*i+1]=gamma*d[i]   # Thickness of high-Z
        z[2*i]=d[i]-z[2*i+1] #  Thickness of low-Z
elif Structure == 'Periodic':
    for i in range(N): 
        d[i] = dmax # Bi-layer thickness = dmax
        z[2*i+1]=gamma*d[i]  # Thickness of high-Z 
        z[2*i]=d[i]-z[2*i+1] #  Thickness of low-Z
elif Structure == 'Linear':
    for i in range(N): 
        d[i] = dmax-((dmax-dmin)/(N-1))*(i); # Bi-layer thickness
        z[2*i+1]=gamma*d[i]   # Thickness of high-Z
        z[2*i]=d[i]-z[2*i+1] #  Thickness of low-Z
if gamma_top[0]:
            z[1]=gamma_top[1]*d[0]
            z[0]=d[0]-[z[1]]
sample_structure = [] # Initialiser
for i in range(N): 
    sample_structure += [[a for a in lowz_layer]]
    sample_structure += [[a for a in highz_layer]] 
sample_structure += [substrate]
for i in range(N*2):
   sample_structure[i] += [[round(z[i],2)]]
    

# -- Title --
theta_m = theta_i + ang_off # Theta model includes angle off set
energy_m = energy_i + eng_off # Energy model includes energy off set
if np.isscalar(energy_m) and theta_i[1]>theta_i[0] :
    # limit theta range to [min_theta, max_theta]
    angleScan = True
    energy_m = energy_m*1000
elif np.isscalar(theta_m) and energy_i[1]>energy_i[0] :
    energy_m = energy_m*1000 # I REMOVED THIS!!!
    angleScan = False


lambda_i = energy2wavelength(energy_m)

# Create model_space input for d7e()
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
        roughness.append(layer[2])
        densities.append(layer[3])

thickness = densities[:-1]
for i in range(len(z)):
    thickness[i]=z[i]

model_space = densities + roughness + thickness 
wavelength = lambda_i
rho = densities
sigma = roughness
oc_source = optical_constants_dir
theta = theta_m


# --- Get reflectance ---
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






# --- Plot ---
if angleScan:
    fig, (ax3) = plt.subplots(nrows=1, ncols=1,figsize=(6,4.5))
    fig.tight_layout(pad=3.0)
    ax3.plot(theta, reflectance,'k-',label ='Model')
    #ax3.plot(theta, reflectance_3,'b-',label ='Model2')
    ax3.set_ylabel("Reflectance")
    ax3.set_yscale("log")
    ax3.set_xlabel("Grazing angle (deg)") 
    ax3.legend()
else:
    fig, (ax3) = plt.subplots(nrows=1, ncols=1,figsize=(11,4.5))
    fig.tight_layout(pad=4.0)
    ax3.plot(energy_m/1000, reflectance,'k-')
    ax3.set_xlabel("Energy (keV)")  
    ax3.set_ylabel("Reflectance")     





if save_as_txt[0]:   
    if angleScan:  
        #n_bf = np.zeros(len(elements)+1, dtype=np.complex)+1 
        # Calculate refractive indices of the layer materials
        #for j in range(len(elements)): 
            #n_bf[j+1] = oc(lambda_i,rho_bf[j],number_of_atoms[j],elements[j], optical_constants_dir)                         
        #reflectance_bf = fresnel(theta_m,lambda_i ,n_bf,z_bf,sigma_bf, ares = a_res)
        #save(save_as_txt[1],angleScan,sample_structure, theta_m, reflectance, data,rho_bf, sigma_bf, z_bf, minFOM, a_res, energy_m, logFitting, weightsFitting, ang_off, eng_off)
        bla=2
    else:
        
        fname = save_as_txt[1]
        file1 = open(fname,"w+") 
        today = date.today()
        file1.write("# Saved on %s\n\n" % today.strftime("%d/%m/%Y"))
        file1.write("# Sample structure:\n")
    

        file1.write("# N = %s \n" % N)
        file1.write("# Structure: %s \n" % Structure)
        file1.write("# dmin [nm] = %s \n" % dmin)
        file1.write("# dmax [nm] = %s \n" % dmax)
        file1.write("# Gamma = %s \n" % gamma)
        if Structure == 'Power law': file1.write("# c = %s \n" % c)
        if gamma_top[0]: file1.write("# Gamma top = %s \n" % gamma_top[1])
    
        for i in range(len(sample_structure)-1):
            file1.write("# %s rho = %s (g/cm^3), sigma = %s (nm), z = %s (nm) \n" % (sample_structure[i][0:2],  str(rho[i][0]),  str(sigma[i][0]),  str(round(z[i],2))   ))
        
        file1.write("# %s rho = %s (g/cm^3), sigma = %s (nm) \n\n" % (sample_structure[-1][0:2], str(rho[-1][0]), str(sigma[-1][0])))
        #file1.write("# FOM = %.2E \n" % minFOM)
        #file1.write("# Angular Offset on model (degree): %s \n" % ang_off)
        #file1.write("# Energy Offset on model (eV): %s \n" % eng_off)
        file1.write("# Incident angle (degree):  %s \n" % theta)
        #file1.write("# Angular resolution (degree): %s \n" % a_res )
        #file1.write("# Logaritmic fitting: %d \n" % (logFitting))
        #file1.write("# Weighting: %s \n\n"% weightsFitting)    
        file1.write("# Energy (keV), Reflectance\n" )
    
        for i in range(len(energy_m)):
             file1.write("%.4E %.9E\n" % (energy_m[i]/1000, reflectance[i]))
        file1.close() 
        
        
        
        
        
         