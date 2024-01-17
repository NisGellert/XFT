import numpy as np
import os
os.chdir(r"C:\Users\nige\DTU Space\PhD\XRR_Fitting_tool")


# ===============================================================
# The following is an example of how to set up 
# a 8.048 keV XRR measurement of an Ir/SiC bi-layer.

# Define directories
optical_constants_dir = "oc_source/CXRO/" # [CXRO, NIST, LLNL]

# Ir/SiC bilayer - 8.048 keV (simulated data in IMD GUI)
# Define meaured data
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0076_4_001.txt"
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Parabolic mirror/ds0079_2_001.txt"

#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Cal 2/ds0075_1_backup.txt"
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Cal 2/ds0075_1_backup.txt"
#
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0008_lexr_spec_20210826_3.txt"
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0021_lexr_spec_20210802.txt"
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0027_lexr_spec_20210802.txt"
#save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0008_lexr_spec_20210826_3_XFT.txt"] # Save fitted data as txt 
#save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0076_XFT_con.txt"] # Save fitted data as txt 

#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0021_lexr_spec_20210802.txt"
#save_as_txt = [True, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0021_lexr_spec_20210802_XFT.txt"] # Save fitted data as txt 



#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0077_1_001.txt"
#save_as_txt = [True, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0077_1_001_XFT.txt"] # Save fitted data as txt 

#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0078_1_001.txt"
#save_as_txt = [True, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0078_1_001_XFT.txt"] # Save fitted data as txt 


fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0079_9_001.txt"
save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0079_9_001_XFT.txt"] # Save fitted data as txt 
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0080_1_001.txt"
#save_as_txt = [True, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0080_1_001_XFT.txt"] # Save fitted data as txt 
#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0081_1_001.txt"
#save_as_txt = [True, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/ds0081_1_001_XFT.txt"] # Save fitted data as txt 

#save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Small test mirrors/Remeasure_3_sample_Apr21/Scatter/ds0027_lexr_spec_20210802_XFT.txt"] # Save fitted data as txt 


#fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/BESSY 21/ds0079_Round1_0.6_x=099.2.txt"
#save_as_txt = [False, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/BESSY 21/ds0079_Round1_0.6_x=099.2_XFT.txt"] # Save fitted data as txt 

#fname =  "Data/Measured data/ir_sic_ares_0p0_energy_1p54A.txt"
data = np.loadtxt(fname)[:,1] # Normalized measured reflectance
theta_i = np.loadtxt(fname)[:,0] # [deg] Incident angle 
xrange = [0.45, 2.5] # min_x and max_x

# Define Measurement Setup 
energy_i = 8048# [eV]. Reflectometer energy. Array or variable.
a_res = .007 # [degree] Instrument Angular resolution Note: Not sure if correct
e_res = 0.0 # [m]. Instrumental spectral resolution. Note: Not implemented
polarization_incident = 0.0 # Incident Polarization [-1:1].
polarization_analyzer = 1.0 # Polarization analyzer sensitivity
ang_off = 0.01 # [degree] Angular offset 
eng_off = 0.0 # [eV] Energy offset 

# Define differentual evolution parameters 
mutation_factor = 0.5 # [0.1:2.0]
crossover_probability = 0.75 # [0:1]
population_size = 30 # [10:50]
iterations = 100 # [25 - 550] 4
mutation_scheme = "Rand/1" # "Rand/1" or "Best/1"
gauss_range = [0.3, 2.1] # min_x and max_x. Important area in curve NOT INCLUDED!!


plotFit = True # plots the best fit in every iteration
logFitting = True
weightsFitting = 'equal' # 'equal' or 'statistical'
multilayer = False # Not finnoshed

# Define sample structure and  qmodel space  
#           [material,      composition,    z,      sigma,      rho]
cho_layer = [["C","H","O"],    [1,1,1],  [1., 4.5],     [0.05, 1.0],[1.0, 1.0]] 
pt_layer = [["Pt"], [1],  [10.1, 35.],     [0.05, 1.5], [21.45, 21.45]]  #  21.45
cr_layer =   [["Cr"],      [1],    [1.0, 5.0], [0.1, 1.5], [7.2, 7.2]]  # 7.19
sio_layer = [["Si","O"],    [1,2],  [2.0, 2.0],     [0.05, 1.5],[2.65, 2.65]] 
substrate = [["Si"],    [1],  np.NaN,     sio_layer[3],[2.33, 2.33]] 
#substrate = [["Si","O"],    [1,2],  np.NaN,     [0.1, 1.0],[2.65, 2.65]] 

sample_structure = [#cho_layer,
                    pt_layer,
                    cr_layer,
                    sio_layer,
                    substrate]

# Maybe couple Pt and CHO roughness. 



'''
# ===============================================================
# The following is an example of how to set up 
# a 1.0 deg XRR measurement of an Ir/SiC bi-layer.

# Define directories
optical_constants_dir = "oc_source/CXRO/" # [CXRO, NIST, LLNL]

# Ir/SiC bilayer - 8.048 keV (simulated data in IMD GUI)
# Define meaured data

fname = "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/BESSY/ds0079_Round1_0.6_x=099.2.dat"
save_as_txt = [True, "C:/Users/nige/DTU Space/PhD/Projects/Beatrix mirror/Parabolic mirror/Five samples for my project/BESSY/ds0079_Round1_0p6_XFT.txt"] # Save fitted data as txt 

data = np.loadtxt(fname)[:,1] # Normalized measured reflectance
energy_i = np.loadtxt(fname)[:,0] # [eV] Incident energy

xrange = [2000, 11000.0] # [eV] min_x and max_x

# Define Measurement Setup 
theta_i = 0.6 # [deg]. Reflectometer energy. Array or varable.
a_res = 0.0 # [deg] Instrument Angular resolution Note: Not sure if correct
e_res = 0.0 # [m]. Instrumental spectral resolution. Note: Not implemented
polarization_incident = 0.0 # Incident Polarization [-1:1].
polarization_analyzer = 1.0 # Polarization analyzer sensitivity
ang_off = 0.0 # [deg] Angular offset 
eng_off = 40.0 # [eV] Energy offset 

# Define differentual evolution parameters 
mutation_factor = 0.5 # [0.1:2.0]
crossover_probability = 0.75 # [0:1]
population_size = 50 # [10:50]
iterations = 15 # [25 - 550] 
mutation_scheme = "Best/1" # "Rand/1" or "Best/1"


plotFit = True # plots the best fit in every iteration
logFitting = True
weightsFitting = 'equal' # 'equal' or 'statistical'


# Define sample structure and model space  
#           [material,      composition,    z,      sigma,      rho]
cho_layer = [["C","H","O"],    [1,1,1],  [1.0, 4.5],     [0.1, 0.8],[1.0, 1.0]] 
pt_layer = [["Pt"], [1],  [19.1, 33.],     cho_layer[3], [21.45, 21.45]]  #  21.45
cr_layer =   [["Cr"],      [1],    [1., 5.0], [0.1, 1.7], [7.2, 7.2]]  # 7.19
sio_layer = [["Si","O"],    [1,2],  [2., 2.0],     [0.1, 1.6],[2.65, 2.65]] 
substrate = [["Si"],    [1],  np.NaN,     sio_layer[3],[2.33, 2.33]] 
#substrate = [["Si","O"],    [1,2],  np.NaN,     cr_layer[3],[2.65, 2.65]] 

sample_structure = [cho_layer,
                    pt_layer,
                    cr_layer,
                    sio_layer,
                    substrate]




'''
'''


# ===============================================================
# The following is an example of how to set up 
# a 8.048 keV XRR measurement of an Ir/SiC multi-layer.

# Define directories
optical_constants_dir = "oc_source/CXRO/" # [CXRO, NIST, LLNL]

# Ir/SiC bilayer - 8.048 keV (simulated data in IMD GUI)
# Define meaured data
fname =  "Data/Measured data/sonny_data.txt"
#fname =  "Data/Measured data/SiC_Ir_multilayer_simple_4A.txt"
data = np.loadtxt(fname)[:,1] # Normalized measured reflectance
theta_i = np.loadtxt(fname)[:,0] # [deg] Incident angle 
xrange = [0.2, 4.0] # min_x and max_x

# Define Measurement Setup 
energy_i = 8048 # [eV]. Reflectometer energy. Array or variable.
a_res = 0.0 # [degree] Instrument Angular resolution Note: Not sure if correct
e_res = 0.0 # [m]. Instrumental spectral resolution. Note: Not implemented
polarization_incident = 0.0 # Incident Polarization [-1:1].
polarization_analyzer = 1.0 # Polarization analyzer sensitivity
ang_off = 0.0 # [degree] Angular offset 
eng_off = 0.0 # [eV] Energy offset 

# Define differentual evolution parameters 
mutation_factor = 0.5 # [0.1:2.0]
crossover_probability = 0.75 # [0:1]
population_size = 30 # [10:50]
iterations = 350 # [25 - 550] 
mutation_scheme = "Best/1" # "Rand/1" or "Best/1"


plotFit = True # plots the best fit in every iteration
logFitting = False
weightsFitting = 'equal' # 'equal' or 'statistical'
save_as_txt = [True, "saved_data.txt"] # Save fitted data as txt 


# Define sample structure and model space  


N = 5

#%%

#           [material,      composition,    z,      sigma,      rho]
lowz_layer =   [["Si"],     [1],      [0.1, 0.5],   [0.01, 1.], [2.329,2.329]] 
highz_layer =   [["W"],     [1],      [0.1, 0.5],   [0.01, 1.], [19.3,19.3]] 
sio = [["Si","O"],     [1,2],    [0.2, 0.2] ,       [0.01, 1.], [2.65, 2.65]] 
substrate = [["Si"],     [1],    np.NaN,        [0.01, 1.], [2.329, 2.329]] 
sample_structure = []
for i in range(N): 
    sample_structure += [lowz_layer]
    sample_structure += [highz_layer]    
sample_structure += [sio]
sample_structure += [substrate]

#%%
sample_structure=[]

layer_0 = [["Si", "C"],   [1,1],  [2, 12.],     [0.40, 0.55], [3.21,3.21]] 
layer_1 =   [["Ir"],        [1],    layer_0[2], [0.40, 0.55], [22.65,22.65]] 
substrate = [["Si","O"],    [1,2],  np.NaN,     [0.33, 0.33],[2.65, 2.65]] 

#sample_structure = [0] * N*2
for i in range(N):
    newname = "layer_{}".format(2*i)
    exec(newname + "=layer_0" )
    newname = "layer_{}".format(2*i+1)
    exec(newname + "=layer_1" )


sample_structure = [layer_0,
                    layer_1, 
                    layer_2,
                    layer_3, 
                    layer_4,
                    layer_5, 
                    layer_6,
                    layer_7, 
                    layer_8,
                    layer_9, 
                    substrate]
print(sample_structure)   
    
#%%
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
    thickness.append(layer[2])
    roughness.append(layer[3])
    densities.append(layer[4])
        
        
model_space = densities + roughness + thickness[:-1] # excluding substrate "thickness"

#print(model_space[-1] is model_space[-2]) # True
#print(model_space[-4] is model_space[-5]) # False

print(model_space[-1] is model_space[-2]) # True
print(model_space[-1] is model_space[-3]) # True
print(model_space[-12] is model_space[-13]) # False
print(model_space[-12] is model_space[-14]) # IS true, but must be False
print(model_space[-12] is model_space[-15]) # False
print(model_space[-12] is model_space[-16]) # IS true, but must be False

#%%



layer_0 = [["Si", "C"],   [1,1],  [2, 12.],     [0.40, 0.55], [3.21,3.21]] 
layer_1 =   [["Ir"],        [1],    layer_0[2], [0.40, 0.55], [22.65,22.65]] 
layer_2 = [["Si", "C"],   [1,1],  layer_0[2],     [0.40, 0.55], [3.21,3.21]] 
layer_3 =   [["Ir"],        [1],    layer_0[2], [0.40, 0.55], [22.65,22.65]] 
layer_4 = [["Si", "C"],   [1,1],  layer_0[2],     [0.40, 0.55], [3.21,3.21]] 
layer_5 =   [["Ir"],        [1],    layer_0[2], [0.40, 0.55], [22.65,22.65]] 
layer_6 = [["Si", "C"],   [1,1],  layer_0[2],     [0.40, 0.55], [3.21,3.21]] 
layer_7 =   [["Ir"],        [1],    layer_0[2], [0.40, 0.55], [22.65,22.65]] 
layer_8 = [["Si", "C"],   [1,1],  layer_0[2],     [0.40, 0.55], [3.21,3.21]] 
layer_9 =   [["Ir"],        [1],    layer_0[2], [0.40, 0.55], [22.65,22.65]]
substrate = [["Si","O"],    [1,2],  np.NaN,     [0.33, 0.33],[2.65, 2.65]] 


sample_structure = [layer_0,
                    layer_1, 
                    layer_2,
                    layer_3, 
                    layer_4,
                    layer_5, 
                    layer_6,
                    layer_7, 
                    layer_8,
                    layer_9, 
                    substrate]

print(sample_structure)
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
    thickness.append(layer[2])
    roughness.append(layer[3])
    densities.append(layer[4])
        
        
model_space = densities + roughness + thickness[:-1] # excluding substrate "thickness"

#print(model_space[-1] is model_space[-2]) # True
#print(model_space[-4] is model_space[-5]) # False

print(model_space[-1] is model_space[-2]) # True
print(model_space[-1] is model_space[-3]) # True
print(model_space[-12] is model_space[-13]) # False
print(model_space[-12] is model_space[-14]) # Is true in for loop, but must be False like here
print(model_space[-12] is model_space[-15]) # False
print(model_space[-12] is model_space[-16]) # Is true in for loop, but must be False like here


#%%
sample_structure = []
for i in range(N): 
    sample_structure += [[a for a in lowz_layer]]
    sample_structure += [[a for a in highz_layer]] 
    
#%%
top_layer = [["Si", "C"],   [1,1],  [2, 12.],     [0.40, 0.55], [3.21,3.21]] 
layer_1 =   [["Ir"],        [1],    top_layer[2], [0.40, 0.55], [22.65,22.65]] 
substrate = [["Si","O"],    [1,2],  np.NaN,     [0.33, 0.33],[2.65, 2.65]] 

sample_structure = [top_layer,
                    layer_1, 
                    substrate]

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
    thickness.append(layer[2])
    roughness.append(layer[3])
    densities.append(layer[4])
        
        
model_space = densities + roughness + thickness[:-1] # excluding substrate "thickness"

print(model_space[-1] is model_space[-2]) # True
print(model_space[-4] is model_space[-5]) # False


#%%
#Hele problemet ligger i at se hvornår nogle værdier er coupled, som jeg bruger senere til at sætte dem lig med hinanden. 
#Med den manuelle opsætning virker det ved at tjekke om "a is b". Hvis den er coupled siger den true, og hvis de ikke er coupled siger den false (også selvom de har ens i værdier). 

#For exempel i python:
a =   [1]
b = a

c = [1]

a is b #giver true
a is c #giver false


Det virker. Men hvis det sættes ind i et for loop:
a = [1,2,3]
b = [a[2],3,4]
a1 = a
b1 = b

print(a1 is a) # True
print(b1 is b) # True
print(a is b) # False
print(a1 is b1) # False



for i in range(4):
    newname = "aa_{}".format(2*i)
    exec(newname + "=a" )
    newname = "bb_{}".format(2*i+1)
    exec(newname + "=b" )


print(a1 is a) # True
print(b1 is b) # True
print(a is b) # False
print(a1 is b1) # False



'''