import argparse, os, pickle, pymp, pynbody, sys, time, warnings
import numpy as np
import traceback
# from ShapeFunctions.Functions import halo_shape_stellar
from pathlib import Path
import sys

# Current script directory (3DShapesCollector.py's location)
current_dir = Path(__file__).parent

shape_functions_dir = current_dir.parent / 'ShapeFunctions'
root_dir = current_dir.parent.parent
code_dir = root_dir / 'Code'
pickle_dir = code_dir / 'PickleFiles'

# Convert the path to a string and append it to sys.path
sys.path.append(str(shape_functions_dir))
from Functions import halo_shape_stellar

def adjust_rin(halo, initial_rin, target_particles, max_iterations=100, tolerance=0.2):
    rin = initial_rin
    for _ in range(max_iterations):
        particles_within = halo[halo['r'] < rin]
        N = len(particles_within)
        # Check if we're within the tolerance
        if abs(N - target_particles) / target_particles <= tolerance:
            #print warning if rin is greater than 0.25kpc
            # if rin > 0.25:
            #     print(f'Warning: rin for halo {hid} is greater than 0.25kpc.')
            #limit lowst value of rin to 0.15kpc
            if rin < 0.15:
                rin = 0.15
            return rin

        # Use cube root for 3D space, allows for both increasing and decreasing
        rin = rin * ((target_particles / N) ** (1 / 3))
    raise ValueError(
        f"Could not find suitable rin with {target_particles} ± {tolerance * 100}% particles after {max_iterations} iterations")

#from Functions import halo_shape_stellar
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

parser = argparse.ArgumentParser(description='Collect 3D shapes of all resolved halos from a given simulation.')
parser.add_argument('-f','--feedback',default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing data file')
args = parser.parse_args()

#custom to rerun specific simulations r431, r615 ,r618, r634 even without overwrite
# if args.simulation in ['r615','r618','r634']:
#     args.overwrite = True

#try and load output file if it already exists
#ignore if overwrite is set

if not args.overwrite:
    if os.path.exists(root_dir / 'Data' / f'{args.simulation}.{args.feedback}.3DShapes.pickle'):
        with open(root_dir / 'Data' / f'{args.simulation}.{args.feedback}.3DShapes.pickle', 'rb') as file:
            ShapeData = pickle.load(file)
        print(f'Output file found for {args.simulation}. Exiting...')
        sys.exit()


#SimInfo = pickle.load(open(f'../SimulationInfo.{args.feedback}.pickle','rb'))
try:
    SimInfo = pickle.load(open(pickle_dir / f'SimulationInfo.{args.feedback}.pickle', 'rb'))
except:
    print(f'Could not find SimulationInfo.{args.feedback}.pickle in {code_dir}.')
    sys.exit()




simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['goodhalos']
print(f'Loading {args.simulation}')
tstart = time.time()
sim = pynbody.load(simpath)
sim.physical_units()
h = sim.halos()
ShapeData = pymp.shared.dict()
myprint(f'{args.simulation} loaded.',clear=False)




prog=pymp.shared.array((1,),dtype=int)
#print(f'\tCalculating Shapes: 0.00%')
with pymp.Parallel(args.numproc) as pl:
    for i in pl.xrange(len(halos)):
        t_start_current = time.time()
        if args.verbose: print(f'\tAnalyzing {halos[i]}...')
        hid = halos[i]
        halo = h[hid]
        current = {}

        try:
            pynbody.analysis.angmom.faceon(halo)
            rout = halo.s['r'].max()

            #calculate number of bins, aim to have 200 particles per bin
            n = len(halo.s)
            #check that there at at least 10^4 particles
            if n < 10**4:
                # display warnimg
                print(f"Warning: {n} particles in halo {hid}")
                bins = 20
                n_i = int(n/bins)
            else:
                bins = int((np.log10(n)-3)**2 * 20)
                #set inner radius to .25kpc
                n_i = int(n/bins)
                #cap n_i to 200
                if n_i > 250:
                    n_i = 250
            rin = pynbody.array.SimArray(0.25, 'kpc')
            try:
                rin = adjust_rin(halo.s, initial_rin=rin, target_particles=1.5*n_i)
                rin = pynbody.array.SimArray(rin, 'kpc')
                #print particles within rin, and number of bins
                #print(f"Particles within rin: {len(halo.s[halo.s['r'] < rin])}")
                print(f"Number of bins: {bins}")
                print(f"Adjusted rin: {rin}")
            except ValueError as e:
                print(f"Error: {e}")
            except RuntimeError as e:
                print(f"Error: {e}")

            rout = halo.s['r'].max()
            print(f'rin: {rin}, rout: {rout}, bins: {bins}')
            rbins, a, ba, ca, angle, Es, n, n_i = halo_shape_stellar(halo.s,N=bins,rin=rin,rout=rout)

            print(rbins[0],rbins[-1])
            #rbins, axes, N, rotations, weights = pynbody.analysis.halo.shape(halo.s,nbins = 100, max_iterations = 20, tol=1e-2, justify = True)
            # a = axes[:, 0]
            # b = axes[:, 1]
            # c = axes[:, 2]
            # ba = b / a
            # ca = c / a
            # Es = Rotations
            current['rbins'] = rbins
            current['a'] = a
            current['ba'] = ba
            current['ca'] = ca
            z = np.array((0,0,1))
            #rotations is rotation matrix as a function of rbin, apply rotations matrix to z
            #to get the new z axis
            #along the axis of rbins

            current['angle'] = angle
            current['Es'] = Es
            current['N_s'] = len(halo.s)

        except:
            print(f'Error in halo {hid}: {traceback.format_exc()}')
            current['rbins'] = np.array([])
            current['a'] = np.array([])
            current['ba'] = np.array([])
            current['ca'] = np.array([])
            current['angle'] = 0
            current['Es'] = np.identity(3)
            current['N_s'] = np.nan

        ShapeData[str(hid)] = current
        prog[0]+=1
        #if not args.verbose: myprint(f'\tCalculating Shapes: {round(prog[0]/len(halos)*100,2)}%',clear=True)
        t_end_current = time.time()
        if args.verbose: print(f'\t\t{hid} done in {round((t_end_current-t_start_current)/60,2)} minutes.')

data_dir = root_dir / 'Data'

#convert shared dict to normal dict
ShapeData = dict(ShapeData)

ShapeFile = {}
for halo in halos:
    #check if halo was analyzed
    if str(halo) in ShapeData:
        ShapeFile[str(halo)] = ShapeData[str(halo)]
    else:
        print(f'Warning: Halo {halo} not analyzed')

file_path = data_dir / f'{args.simulation}.{args.feedback}.3DShapes.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(ShapeFile, file)
    
tstop = time.time()
#myprint(f'\t{args.simulation} finished in {round((tstop-tstart)/60,2)} minutes.',clear=True)
print(f'\t{args.simulation} finished in {round((tstop-tstart)/60,2)} minutes.')