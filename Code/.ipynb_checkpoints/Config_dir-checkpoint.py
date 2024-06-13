import os
import pickle
from bwcdm_sims import Sims as BWCDMSims
from sbcdm_sims import Sims as SBCDMSims
from rdz_sims import Sims as RDZSims
from akaxia import Sims as AKSims
from stormCDM import Sims as ABSims

# Function to initialize directories
def init_directories(sim_dict, sim_type):
    for sim in sim_dict:
        os.system(f'mkdir ../Figures/Images/{sim}.{sim_type}')
        for hid in sim_dict[sim]['goodhalos']:
            os.system(f'mkdir ../Figures/Images/{sim}.{sim_type}/{hid}')

# Configuration for creating directories
create_dirs_config = {
    'BW': False,  # Set to False if you don't want to create directories for BW-CDM sims
    'SB': False,  # Set to False for SB-CDM sims
    'RDZ': True,  # Set to False for RDZ sims
    'AK': False,
    'AB': False
}

# Save simulation info to pickle
if create_dirs_config['BW']:
	pickle.dump(BWCDMSims, open('SimulationInfo.BW.pickle', 'wb'))
if create_dirs_config['SB']:
	pickle.dump(SBCDMSims, open('SimulationInfo.SB.pickle', 'wb'))
if create_dirs_config['RDZ']:
	pickle.dump(RDZSims, open('SimulationInfo.RDZ.pickle', 'wb'))
if create_dirs_config['AK']:
	pickle.dump(AKSims, open('SimulationInfo.AK.pickle', 'wb'))
if create_dirs_config['AB']:
	pickle.dump(ABSims, open('SimulationInfo.AB.pickle', 'wb'))
    
# Initialize image directory
init = input('Initialize Image Directory? (y,n): ')
if init == 'y':
    print('Initializing Image Directory...')
    path = '../Figures/Images'
    os.makedirs(path, exist_ok=True)
    if create_dirs_config['BW']:
        init_directories(BWCDMSims, "BW")
    if create_dirs_config['SB']:
        init_directories(SBCDMSims, "SB")
    if create_dirs_config['RDZ']:
        init_directories(RDZSims, "RDZ")
    if create_dirs_config['AK']:
        init_directories(AKSims, "AK")
    if create_dirs_config['AB']:
        init_directories(ABSims, "AB")
