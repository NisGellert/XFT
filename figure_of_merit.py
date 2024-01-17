
'''
Author: Nis C. Gellert
DTU Space
Last updated: 11/08/2021
Reference: IMD installation guide & user's manual. pg. 146-151, eq. 1-2
'''

import numpy as np
#from scipy.stats import linregress

def fom(data, model, logarithmic = False, weighting = 'equal'):
    # Compute Chi-square statistic
    
    if weighting == 'equal': 
        weights = np.ones(len(data)) # 1's
        
    elif weighting == 'statistical':
        weights = 1/data
        
    #elif weighting == 'instrumental':
        #if np.isscalar(R_error)== True:
            # print("Uncertainty on data not provided - using statistical weighting instead")
            #weights = 1/data # use statistical weighting if R_error of measured data has not been provided
        #else:
            #weights = 1/(R_error**2)
            # print("Instrumental weighting")
    
    if logarithmic: 
        chisq = np.nansum( weights*(np.log(model) - np.log(data))**2. )
        FOM = np.nansum(weights*(  np.log(model)-np.log(data))**2.) /np.nansum(weights)
    else: 
        chisq = np.nansum(weights*(model-data)**2.)
        FOM = np.nansum(weights*(model-data)**2.)/np.nansum(weights)
    
    # chisq = sum((model-data)**2./data) 
    

   
    return chisq



def fom_MO(x, reflectance, FOM_range, weighting = 'equal'): # x (energy/theta)
     
    if FOM_range[0]:
        for ij in range(len(FOM_range[1])):
            idx = (np.abs(x - FOM_range[1][ij])).argmin()
            reflectance[idx] = reflectance[idx]*FOM_range[2]
    
    if weighting == "equal":
        FOM = np.nansum(np.abs(reflectance))
    elif weighting == "statistical":
        FOM = np.nansum(np.abs(reflectance*x))
    elif weighting == "statistical2":
        FOM = np.nansum(np.abs(reflectance*x**2))
    elif weighting == "statistical3":
        FOM = np.nansum(np.abs(reflectance*x**3))
    elif weighting == "statistical3":
        FOM = np.nansum(np.abs(reflectance*x**4))
    elif weighting == "statistical5":
        FOM = np.nansum(np.abs(reflectance*x**5))
    elif weighting == "statistical6":
        FOM = np.nansum(np.abs(reflectance*x**6))
    elif weighting == "log":
        FOM = np.nansum(np.abs(np.log(reflectance)))
    #if weighting == "slope":
        
        #FOM = linregress(x, reflectance)[0]
    #if weighting == "Gaus":
         #FOM = 
        
    return FOM


