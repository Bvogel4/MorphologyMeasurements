import pynbody
import os
from matplotlib import pyplot as plt
import numpy as np

from akaxia import Sims  
from pynbody.plot.sph import image

import pynbody.plot.sph as sph

# Base directory to store images
base_image_dir = '../Figures/halo_check_images/'

# Ensure the base directory exists
if not os.path.exists(base_image_dir):
    os.makedirs(base_image_dir)

# Parameters for focused testing - specify the simulation and halo ID you want to test
test_sim_name = 'r431'  # Example: 'r431'
test_halo_id = 1  # Example: 1, adjust based on your needs

good_halos = []

# Set a mass threshold
mass_threshold = 1e4

#data_path = Sims['r431']['path']
for sims in Sims:
    data_path = Sims[sims]['path']
    print(data_path)
    sim = pynbody.load(data_path)
    sim.physical_units()
    h = sim.halos(write_fpos=False)
    print(len(h))
    for i in range(1, len(h)):
        try:
            halos = h[i]
            total_mass = halos['mass'].sum().in_units('Msol')
            if total_mass > mass_threshold:
                
                num_p = len(halos)
                #print(num_p)
                try:
                    pynbody.analysis.angmom.faceon(halos)
                    #Rhalf = pynbody.analysis.luminosity.half_light_r(halos)
                    Rhalf = pynbody.analysis.halo.virial_radius(sim, cen=None, overden=178, r_max=None, rho_def='matter')
                    width = Rhalf*6
                    ImageSpace = pynbody.filt.Sphere(width*np.sqrt(2)*1.01)
                    plt.figure(figsize=(8, 8))
                    sph_image = pynbody.plot.sph.image(halos.dm, qty='rho', units='Msol kpc^-3', width=width, cmap='viridis', resolution=500)
                    plt.colorbar(label='Dark Matter Density (Msol/kpc^3)')
                    image_path = base_image_dir + f'sim{sims}h{i}.png'
                    plt.savefig(image_path)
                    print(f'{image_path} saved')
                    plt.close()
                    good_halos.append(i)
                except Exception as e:
                    print(f'error occured plotting halo {i}: {e}')
                
            else:
                 print(f'halo {i} under mass threshold')
        except ValueError:
            print(f'halo {i} does not exist')
        except Exception as e:  # Catch errors
            print(f"Error processing halo {i} in simulation {sims}: {e}")


np.savetxt('akaxia_good_halos.txt', good_halos, delimiter = ',')
       