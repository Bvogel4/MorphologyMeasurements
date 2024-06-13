import argparse,os,pickle,pymp,pynbody,sys,time,warnings
import numpy as np
from pathlib import Path
import sys
warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

parser = argparse.ArgumentParser(description='Collect DM shapes of all resolved halos from a given simulation.')
parser.add_argument('-f','--feedback',choices=['BW','SB','RDZ','AK','AB'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
parser.add_argument('-o', '--override', action='store_true', help='Override existing data if found')
args = parser.parse_args()

current_dir = Path(__file__).parent
data_dir = current_dir.parent.parent / 'Data'
shapefile_path = data_dir / f'{args.simulation}.{args.feedback}.DMShapes.pickle'
progress_path = data_dir / f'{args.simulation}.{args.feedback}.progress.pickle'

# Load or initialize progress
#if progress_path.exists() and not args.override:
    #with open(progress_path, 'rb') as file:
        #ShapeData = pickle.load(file)
#else:
    #ShapeData = pymp.shared.dict()

shape_functions_dir = current_dir.parent / 'ShapeFunctions'
root_dir = current_dir.parent.parent
code_dir= root_dir / 'Code'
sys.path.append(str(shape_functions_dir))
data_dir = root_dir / 'Data'

sim_info_file_path = code_dir / f'SimulationInfo.{args.feedback}.pickle'
with open(sim_info_file_path, 'rb') as file:
    SimInfo = pickle.load(file)
stellar_shape_file_path = data_dir / f'{args.simulation}.{args.feedback}.3DShapes.pickle'
dark_only = False
try:
    with open(stellar_shape_file_path, 'rb') as file:
        StellarShape = pickle.load(file)
except:
    dark_only = True
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
        #if str(hid) in ShapeData:
            #continue 
        current = {}

        try:
            if dark_only == True:
                #print(str(hid))
                #print('DM only')
                pynbody.analysis.angmom.faceon(halo)
                r_vir = pynbody.analysis.halo.virial_radius(halo, cen=None, overden=200, r_max=None, rho_def='critical')
                current['r_vir'] = r_vir
                rout = r_vir * 1.1
                rbins, ba, ca, angle, Es = pynbody.analysis.halo.halo_shape(halo, N=100, rin=None, rout=None, bins='equal')
                current['rbins'] = rbins
                #current['a'] = a
                current['ba'] = ba
                current['ca'] = ca
                current['angle'] = angle
                current['Es'] = Es
                current['N_d'] = len(halo.dm)   
                #current['reff'] = halo['r'].sum() / len(halo) #average radius of the particles in the halo, not weighted by mass
                current['M'] = halo['mass'].sum()
                # num particles per bin
                radii = np.sqrt((halo['x']**2 + halo['y']**2 + halo['z']**2))
                particle_counts = np.zeros(len(rbins))                
                # Count particles in each bin
                for i in range(1, len(rbins)):
                    particle_counts[i-1] = ((radii > rbins[i-1]) & (radii <= rbins[i])).sum()                
                if hid == 1:
                    print(particle_counts)                
                # Store particle counts in the current dictionary
                current['particle_counts'] = particle_counts
            else:
                pynbody.analysis.angmom.faceon(halo)
                rin = StellarShape[str(hid)]['rbins'][0]
                rout = StellarShape[str(hid)]['rbins'][-1]
                rbins,ba,ca,angle,Es = pynbody.analysis.halo.halo_shape(halo,rin=rin,rout=rout)
                current['rbins'] = rbins
                current['ba'] = ba
                current['ca'] = ca
                current['angle'] = angle
                current['Es'] = Es
                current['N_d'] = len(halo.dm) 
        except Exception as e:
            print(f"An error occurred for halo {hid}: {e}")
            current['rbins'] = np.array([])
            current['ba'] = np.array([])
            current['ca'] = np.array([])
            current['angle'] = 0
            current['Es'] = np.identity(3)
            current['N_d'] = (np.nan) 

        ShapeData[str(hid)] = current
        prog[0]+=1
        if not args.verbose: myprint(f'\tCalculating Shapes: {round(prog[0]/len(halos)*100,2)}%',clear=True)
        t_end_current = time.time()
        if args.verbose: print(f'\t\t{hid} done in {round((t_end_current-t_start_current)/60,2)} minutes.')
        # Save progress
        with open(progress_path, 'wb') as file:
            pickle.dump(dict(ShapeData), file)




ShapeFile = {}
for halo in halos:
    ShapeFile[str(halo)] = ShapeData[str(halo)]

file_path = data_dir / f'{args.simulation}.{args.feedback}.DMShapes.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(ShapeFile, file)

tstop = time.time()
myprint(f'\t{args.simulation} finished in {round((tstop-tstart)/60,2)} minutes.',clear=True)