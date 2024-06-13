import os
import pickle
import sys
import pynbody
import numpy as np


#generaize to work with any SimInfo
# Load simulation info from the specific RDZ pickle file
with open('SimulationInfo.RDZ.pickle', 'rb') as f:
    SimInfo = pickle.load(f)

verbose = True  # Print additional information if true

# Prepare a dictionary to hold mass data for all simulations
all_sim_masses = {}

for simulation in SimInfo:
    simpath = SimInfo[simulation]['path']
    halos = SimInfo[simulation]['goodhalos']
    
    # Check if simulation path exists
    if not os.path.exists(simpath):
        if verbose:
            print(f"Path does not exist: {simpath}")
        continue

    try:
        # Load the simulation
        sim = pynbody.load(simpath)
        sim.physical_units()
        if verbose:
            print(f"Simulation {simulation} loaded successfully.")
    except Exception as e:
        if verbose:
            print(f"Failed to load simulation {simulation}: {e}")
        continue

    # Prepare to calculate masses
    h = sim.halos()
    masses = {}  # This will hold mass data for the current simulation

    # Sequentially calculate masses
    for halo_id in halos:
        halo = h[halo_id]
        
        Mvir = halo['mass'].sum().in_units('Msol').view(np.ndarray)  # Total mass, converted to plain number
        Mstar = halo.star['mass'].sum().in_units('Msol').view(np.ndarray)  # Stellar mass, converted
        Mgas = halo.gas['mass'].sum().in_units('Msol').view(np.ndarray)  # Gas mass, converted
        HI = (halo.gas['HI'] * halo.gas['mass']).sum().in_units('Msol').view(np.ndarray)  # HI mass, converted
        Mb = Mgas + Mstar
        Mb_Mtot_ratio = Mb / Mvir if Mvir != 0 else np.nan

        # Store computed values in the masses dictionary
        masses[str(halo_id)] = {
            'Mvir': Mvir.tolist(),  # Converting np.ndarray to list for JSON compatibility
            'Mstar': Mstar.tolist(),
            'Mgas': Mgas.tolist(),
            'HI': HI.tolist(),
            'Mb/Mtot': float(Mb_Mtot_ratio)  
            #add mb/ mvir(<reff)
            #add mass-to-light-ratio
        }

    # Store the simulation's mass data in the all_sim_masses container
    all_sim_masses[simulation] = masses

print(all_sim_masses)

# Save the comprehensive results
with open("../Data/BasicData/RDZ.Masses.pickle", "wb") as f:
    pickle.dump(all_sim_masses, f)

if verbose:
    print("Finished calculating masses for all simulations. Results saved.")
