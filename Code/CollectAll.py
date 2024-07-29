import argparse,os,pickle,sys
import os
import pickle
import multiprocessing

from SimInfoDicts.sim_type_name import sim_type_name
import pathlib

parser = argparse.ArgumentParser(description='Collect data from all simulations')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
args = parser.parse_args()

overwrite = '-o' if args.overwrite else ''
verbose = '-v' if args.verbose else ''

#describe each options and suboptions to the user at the start of the script
#Isophote imaging collects isophotes at different rotations, with an option to generate images
#Intrinsic shapes collects shapes of stars or dark matter particles
#gala fits spherical harmonics to the density profile, with an option to plot fitted density profiles
#mdyn calculates the dynamical mass of the galaxy
print('This script will collect data from all simulations for a given feedback type')
print('The following options are available:')
print('I: Collect isophote images and profiles')
print('S: Collect intrinsic shapes of stars or dark matter')
print('G: Collect density profiles using gala')
print('M: Calculate dynamical mass')
gen_im = ''
loop = True
while loop:
    type = input('Collect Images, Shapes, Gala, or Mdyn (I/S/G/M): ')
    if type in ['I','S','G','M']:
        loop = False
if type=='I':
    subdir = 'IsophoteImaging'
    loop = True
    while loop:
        im = input('Generate images in addition to Profiles? (y/n): ')
        if im in ['y','n']: loop = False
    gen_im = '-i' if im=='y' else ''
elif type=='S':
    subdir = 'IntrinsicShapes'
    loop = True
    while loop:
        im = input('Stellar Shapes or DM Shapes (S/D): ')
        if im in ['S','D']: loop = False
    stype = 'Stars' if im=='S' else 'Dark'
elif type=='G':
    subdir = 'IntrinsicShapes'
    loop = True
    while loop:
        im = input('Plot Density Profiles? (y/n): ')
        if im in ['y','n']: loop = False
    gen_im = '-i' if im=='y' else ''
elif type=='M':
    subdir='XuCorrelation'



def process_simulation(simulation, feedback, subdir, gen_im, numproc, verbose, overwrite):
    if type == 'I':
        os.system(f"{sys.executable} {subdir}/ImageCollection.py -f {feedback} -s {simulation} {gen_im} -n {numproc} {verbose} {overwrite}")
    elif type == 'S':
        os.system(f"{sys.executable} {subdir}/3DShapeCollection.{stype}.py -f {feedback} -s {simulation} -n {numproc} {verbose} {overwrite}")
    elif type=='G':
        os.system(f"{sys.executable} {subdir}/GalaCollector.py -f {feedback} -s {simulation} -n {numproc} {gen_im} {verbose}")
    elif type=='M':
        os.system(f"{sys.executable} {subdir}/DynamicalMass.py -f {feedback} -s {simulation} -n {numproc}")


def process_inputs(subdir, gen_im, verbose, overwrite, numproc):
    for feedback, use_sim in sim_type_name.items():
        if use_sim:
            pickle_path = f'PickleFiles/SimulationInfo.{feedback}.pickle'
            print('sim run',feedback)
            if os.path.exists(pickle_path):
                sims = pickle.load(open(pickle_path, 'rb'))
                with multiprocessing.Pool(1) as p:
                    p.starmap(process_simulation, [(s, feedback, subdir, gen_im, numproc, verbose, overwrite) for s in sims])
            else:
                print(f"No pickle file found for {feedback} feedback type.")

process_inputs(subdir, gen_im, verbose, overwrite,args.numproc)

# for feedback, use_sim in sim_type_name.items():
#     if use_sim:
#         pickle_path = f'PickleFiles/SimulationInfo.{feedback}.pickle'
#         if os.path.exists(pickle_path):
#             sims = pickle.load(open(pickle_path, 'rb'))
#             for s in sims:
#                 print('sims,feedback',s,feedback)
#                 if type == 'I':
#                     os.system(f"{sys.executable} {subdir}/ImageCollection.py -f {feedback} -s {s} {gen_im} -n {args.numproc} {verbose} {overwrite}")
#                 elif type == 'S':
#                     os.system(f"{sys.executable} {subdir}/3DShapeCollection.{stype}.py -f {feedback} -s {s} -n {args.numproc} {verbose} {overwrite}")
#                 elif type=='G':
#                     os.system(f"{sys.executable} {subdir}/GalaCollector.py -f {feedback} -s {s} -n {args.numproc} {gen_im} {verbose}")
#                 elif type=='M':
#                         os.system(f"{sys.executable} {subdir}/DynamicalMass.py -f {feedback} -s {s} -n {args.numproc}")
#                 # Add similar elif conditions for type 'G' and 'M' as needed
#         else:
#             print(f"No pickle file found for {feedback} feedback type.")
