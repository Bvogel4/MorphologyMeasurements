import sys
import os
import subprocess
import pickle

#from CollectAll import process_inputs
from SimInfoDicts.sim_type_name import sim_type_name
import logging
#setup log file to be written to ~/logs
#ensure log directory exists
log_dir = os.path.expanduser('~/logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, 'IntrinsicShapesAll.log')
logging.basicConfig(filename=log_file, level=logging.DEBUG)
logging.info('Started IntrinsicShapesAll.py')

def process_inputs(simulation):
    # get Images
    if verbose == '-v':
        logging.info(f'Getting Images for {simulation}')
    exit_code = os.system(
        f"{sys.executable} IsophoteImaging/ImageCollection.py -f {feedback} -s {simulation} {gen_im} -n {numproc} {verbose} {overwrite}")
    if exit_code != 0 and verbose == '-v':
        logging.info(f"Error: ImageCollection failed for {simulation}")
    # get Shapes
    logging.info(f'Getting Shapes for {simulation}')
    for stype in ['Stars', 'Dark']:
        exit_code = os.system(
            f"{sys.executable} IntrinsicShapes/3DShapeCollection.{stype}.py -f {feedback} -s {simulation} -n {numproc} {verbose} {overwrite}")
        # verify exit code for Stars before running Dark
        if exit_code != 0:
            logging.info(f"Error: 3DShapes.{stype} failed for {simulation}, skipping Dark Matter Shapes")
            break

    # get spherical harmonics
    # os.system(f"{sys.executable} IntrinsicShapes/GalaCollector.py -f {feedback} -s {simulation} -n {numproc} {gen_im} {verbose}")
    # get dynamical mass
    # os.system(f"{sys.executable} XuCorrelation/DynamicalMass.py -f {feedback} -s {simulation} -n {numproc}")
numproc = 24 # total number of cores to use
gen_im = '-i' # generate images
verbose = '-v' # verbose output
overwrite = '-o' # overwrite existing files

for feedback, use_sim in sim_type_name.items():
    if use_sim:
        try:
            pickle_path = f'PickleFiles/SimulationInfo.{feedback}.pickle'
        except:
            logging.info(f"Error: No pickle file found for {feedback}")
            continue
        logging.info(f'sim run: {feedback}')
        sims = pickle.load(open(pickle_path, 'rb'))
        for simulation in sims:
            process_inputs(simulation)


