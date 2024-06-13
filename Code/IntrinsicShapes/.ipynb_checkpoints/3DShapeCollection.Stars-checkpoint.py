import argparse, os, pickle, pymp, pynbody, sys, time, warnings
import numpy as np
import traceback
sys.path.append('..')
# from ShapeFunctions.Functions import halo_shape_stellar
from pathlib import Path
import sys

# Current script directory (3DShapesCollector.py's location)
current_dir = Path(__file__).parent

# Assuming your structure is as follows and you need to go up one level from current script,
# then enter the ShapeFunctions directory
shape_functions_dir = current_dir.parent / 'ShapeFunctions'
root_dir = current_dir.parent.parent
code_dir= root_dir / 'Code'

# Convert the path to a string and append it to sys.path
sys.path.append(str(shape_functions_dir))


#from Functions import halo_shape_stellar
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

parser = argparse.ArgumentParser(description='Collect 3D shapes of all resolved halos from a given simulation.')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
args = parser.parse_args()

#SimInfo = pickle.load(open(f'../SimulationInfo.{args.feedback}.pickle','rb'))
SimInfo = pickle.load(open(code_dir / f'SimulationInfo.{args.feedback}.pickle', 'rb'))
simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['halos']

print(f'Loading {args.simulation}')
tstart = time.time()
sim = pynbody.load(simpath)
sim.physical_units()
h = sim.halos()
ShapeData = pymp.shared.dict()
myprint(f'{args.simulation} loaded.',clear=True)

prog=pymp.shared.array((1,),dtype=int)
print(f'\tCalculating Shapes: 0.00%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
        t_start_current = time.time()
        if args.verbose: print(f'\tAnalyzing {halos[i]}...')
        hid = halos[i]
        halo = h[hid]
        current = {}

        try:
            #add subhalo removal
            pynbody.analysis.angmom.faceon(halo)
            rbins,ba,ca,angle,Es = pynbody.analysis.halo.halo_shape(halo,use_family = 'stars')
            current['rbins'] = rbins
            #current['a'] = a
            current['ba'] = ba
            current['ca'] = ca
            current['angle'] = angle
            current['Es'] = Es
            current['N_s'] = len(halo.s)

        except:
            print(f'Error in halo {hid}: {traceback.format_exc()}')
            current['rbins'] = np.array([])
            #current['a'] = np.array([])
            current['ba'] = np.array([])
            current['ca'] = np.array([])
            current['angle'] = 0
            current['Es'] = np.identity(3)
            current['N_s'] = np.nan

        ShapeData[str(hid)] = current
        prog[0]+=1
        if not args.verbose: myprint(f'\tCalculating Shapes: {round(prog[0]/len(halos)*100,2)}%',clear=True)
        t_end_current = time.time()
        if args.verbose: print(f'\t\t{hid} done in {round((t_end_current-t_start_current)/60,2)} minutes.')

data_dir = root_dir / 'Data'


ShapeFile = {}
for halo in halos:
    ShapeFile[str(halo)] = ShapeData[str(halo)]

file_path = data_dir / f'{args.simulation}.{args.feedback}.3DShapes.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(ShapeFile, file)
    
tstop = time.time()
myprint(f'\t{args.simulation} finished in {round((tstop-tstart)/60,2)} minutes.',clear=True)
