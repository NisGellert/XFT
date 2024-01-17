"""
Author: Nis C. Gellert
DTU Space
Last updated: 11/08/2021
"""
import numpy as np

def wavelength2energy(wavelength):
    # wavelength is in nm - conversion to energy in eV 
    eV = 1.60217662*1e-19 # [J] per eV
    h = 6.62607004*1e-34 # [m^2*kg/s]
    c = 299792458 # [m/s]
    
    E = h*c/(wavelength*1e-9)
    energy = E/eV 
    return energy
    
def energy2wavelength(energy):
    # energy is in eV - conversion to wavelength in nm 
    eV = 1.60217662*1e-19 # [J] per eV
    h = 6.62607004*1e-34 # [m^2*kg/s]
    c = 299792458 # [m/s]
    
    E = energy*eV # converting from [eV] to [J]
    wavelength = h*c/E*1e9 # [nm] Instrument wavelength / energy
    return wavelength



def thickness2energy(thickness,theta):
    # thickness (nm), energy (keV), wavelength (nm), theta (degree).   
    energy = wavelength2energy(2*thickness*np.sin(np.deg2rad(theta)))/1000
    return energy

def energy2thickness(energy,theta):
    # thickness (nm), energy (keV), wavelength (nm), theta (degree).   
    thickness = wavelength2energy(energy*1000)/(2*np.sin(np.deg2rad(theta)))
    return thickness


