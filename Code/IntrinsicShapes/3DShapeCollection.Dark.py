import argparse,os,pickle,pymp,pynbody,sys,time,warnings, traceback
import numpy as np
from pathlib import Path
import sys

from pynbody import array
from pynbody.analysis import profile
import numpy as np
import logging
logger = logging.getLogger('pynbody.analysis.halo')
from numpy import matmul as X
from numpy import sin,cos
from skimage.measure import EllipseModel
from math import pi



#explicitly show depreciation warnings
warnings.filterwarnings("always",category=DeprecationWarning)

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
data_dir = root_dir/ 'Data'
pickle_dir = root_dir/ 'Code' / 'PickleFiles'
code_dir= root_dir / 'Code'

#warnings.filterwarnings("ignore")
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

parser = argparse.ArgumentParser(description='Collect DM shapes of all resolved halos from a given simulation.')
parser.add_argument('-f','--feedback',default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
parser.add_argument('-o', '--overwrite', action='store_true', help='Override existing data if found')
args = parser.parse_args()

#can I still make a main function?


#custom to rerun specific simulations r431, r615 ,r618, r634 even without overwrite
# if args.simulation in ['r615','r618','r634']:
#     args.overwrite = True

def halo_shape(sim, N=100, rin=None, rout=None, bins='equal'):
    """
    Returns radii in units of ``sim['pos']``, axis ratios b/a and c/a,
    the alignment angle of axis a in radians, and the rotation matrix
    for homeoidal shells over a range of N halo radii.

    **Keyword arguments:**

    *N* (100): The number of homeoidal shells to consider. Shells with
    few particles will take longer to fit.

    *rin* (None): The minimum radial bin in units of ``sim['pos']``.
    Note that this applies to axis a, so particles within this radius
    may still be included within homeoidal shells. By default this is
    taken as rout/1000.

    *rout* (None): The maximum radial bin in units of ``sim['pos']``.
    By default this is taken as the largest radial value in the halo
    particle distribution.

    *bins* (equal): The spacing scheme for the homeoidal shell bins.
    ``equal`` initialises radial bins with equal numbers of particles,
    with the exception of the final bin which will accomodate remainders.
    This number is not necessarily maintained during fitting.
    ``log`` and ``lin`` initialise bins with logarithmic and linear
    radial spacing.

    Halo must be in a centered frame.
    Caution is advised when assigning large number of bins and radial
    ranges with many particles, as the algorithm becomes very slow.
    """

    #-----------------------------FUNCTIONS-----------------------------
    # Define an ellipsoid shell with lengths a,b,c and orientation E:
    def Ellipsoid(r, a,b,c, E):
        x,y,z = np.dot(np.transpose(E),[r[:,0],r[:,1],r[:,2]])
        return (x/a)**2 + (y/b)**2 + (z/c)**2

    # Define moment of inertia tensor:
    MoI = lambda r,m: np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(3)]\
                               for i in range(3)])

    # Splits 'r' array into N groups containing equal numbers of particles.
    # An array is returned with the radial bins that contain these groups.
    sn = lambda r,N: np.append([r[i*int(len(r)/N):(1+i)*int(len(r)/N)][0]\
                               for i in range(N)],r[-1])

    # Retrieves alignment angle:
    almnt = lambda E: np.arccos(np.dot(np.dot(E,[1.,0.,0.]),[1.,0.,0.]))
    #-----------------------------FUNCTIONS-----------------------------

    if (rout == None): rout = sim.dm['r'].max()
    if (rin == None): rin = rout/1E3

    posr = np.array(sim.dm['r'])[np.where(sim.dm['r'] < rout)]
    pos = np.array(sim.dm['pos'])[np.where(sim.dm['r'] < rout)]
    mass = np.array(sim.dm['mass'])[np.where(sim.dm['r'] < rout)]

    rx = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    ry = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    rz = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]

    # Define bins:
    if (bins == 'equal'): # Each bin contains equal number of particles
        mid = sn(np.sort(posr[np.where((posr >= rin) & (posr <= rout))]),N*2)
        rbin = mid[1:N*2+1:2]
        mid = mid[0:N*2+1:2]

    elif (bins == 'log'): # Bins are logarithmically spaced
        mid = profile.Profile(sim.dm, type='log', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = np.sqrt(mid[0:N]*mid[1:N+1])

    elif (bins == 'lin'): # Bins are linearly spaced
        mid = profile.Profile(sim.dm, type='lin', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = 0.5*(mid[0:N]+mid[1:N+1])

    # Define b/a and c/a ratios and angle arrays:
    ba,ca,angle = np.zeros(N),np.zeros(N),np.zeros(N)
    Es = [0]*N

    # Begin loop through radii:
    for i in range(0,N):

        # Initialise convergence criterion:
        tol = 1E-3
        count = 0

        # Define initial spherical shell:
        a=b=c = rbin[i]
        E = np.identity(3)
        L1,L2 = rbin[i]-mid[i],mid[i+1]-rbin[i]

        # Begin iterative procedure to fit data to shell:
        while True:
            count += 1

            # Collect all particle positions and masses within shell:
            r = pos[np.where((posr < a+L2) & (posr > c-L1*c/a))]
            inner = Ellipsoid(r, a-L1,b-L1*b/a,c-L1*c/a, E)
            outer = Ellipsoid(r, a+L2,b+L2*b/a,c+L2*c/a, E)
            r = r[np.where((inner > 1.) & (outer < 1.))]
            m = mass[np.where((inner > 1.) & (outer < 1.))]

            # End iterations if there is no data in range:
            if (len(r) == 0):
                ba[i],ca[i],angle[i],Es[i] = b/a,c/a,almnt(E),E
                logger.info('No data in range after %i iterations' %count)
                break

            # Calculate shape tensor & diagonalise:
            D = list(np.linalg.eig(MoI(r,m)/np.sum(m)))

            # Purge complex numbers:
            if isinstance(D[1][0,0],complex):
                D[0] = D[0].real ; D[1] = D[1].real
                logger.info('Complex numbers in D removed...')

            # Compute ratios a,b,c from moment of intertia principles:
            anew,bnew,cnew = np.sqrt(abs(D[0])*3.0)

            # The rotation matrix must be reoriented:
            E = D[1]
            if ((bnew > anew) & (anew >= cnew)): E = np.dot(E,rz)
            if ((cnew > anew) & (anew >= bnew)): E = np.dot(np.dot(E,ry),rx)
            if ((bnew > cnew) & (cnew >= anew)): E = np.dot(np.dot(E,rz),rx)
            if ((anew > cnew) & (cnew >= bnew)): E = np.dot(E,rx)
            if ((cnew > bnew) & (bnew >= anew)): E = np.dot(E,ry)
            cnew,bnew,anew = np.sort(np.sqrt(abs(D[0])*3.0))

            # Keep a as semi-major axis and distort b,c by b/a and c/a:
            div = rbin[i]/anew
            anew *= div
            bnew *= div
            cnew *= div

            # Convergence criterion:
            if (np.abs(b/a-bnew/anew) < tol) & (np.abs(c/a-cnew/anew) < tol):
                if (almnt(-E) < almnt(E)): E = -E
                ba[i],ca[i],angle[i],Es[i] = bnew/anew,cnew/anew,almnt(E),E
                break

            # Increase tolerance if convergence has stagnated:
            elif (count%15 == 0): tol *= 5.
            #print warning if tol is greater than 0.05

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew

    return [array.SimArray(rbin, sim.d['pos'].units), ba, ca, angle, Es]

if not args.overwrite:
    if os.path.exists(root_dir / 'Data' / f'{args.simulation}.{args.feedback}.DMShapes.pickle'):
        with open(root_dir / 'Data' / f'{args.simulation}.{args.feedback}.DMShapes.pickle', 'rb') as file:
            ShapeData = pickle.load(file)
        print(f'Output file found for {args.simulation}. Exiting...')
        sys.exit()

sim_info_file_path = code_dir / f'SimulationInfo.{args.feedback}.pickle'
try:
    SimInfo = pickle.load(open(pickle_dir / f'SimulationInfo.{args.feedback}.pickle', 'rb'))
except:
    print(f'Could not find SimulationInfo.{args.feedback}.pickle in {code_dir}.')
    sys.exit()



stellar_shape_file_path = data_dir / f'{args.simulation}.{args.feedback}.3DShapes.pickle'
dark_only = False
try:
    with open(stellar_shape_file_path, 'rb') as file:
        StellarShape = pickle.load(file)
except:
    print(f'Could not find {stellar_shape_file_path}.')
    print(f'Running in dark matter only mode.')
    dark_only = True


def adjust_rin(halo, initial_rin, target_particles, max_iterations=100, tolerance=0.2):
    rin = initial_rin
    for _ in range(max_iterations):
        particles_within = halo[halo['r'] < rin]
        N = len(particles_within)

        # Check if we're within the tolerance
        if abs(N - target_particles) / target_particles <= tolerance:
            #print warning if rin is greater than 0.25kpc
            if rin <0.15:
                rin = 0.15
            return rin

        # Use cube root for 3D space, allows for both increasing and decreasing
        if N == 0:
            #assume bad guess of rin and return last rin
            return rin
        
        rin = rin * ((target_particles / N) ** (1 / 3))

    raise ValueError(
        f"Could not find suitable rin with {target_particles} ± {tolerance * 100}% particles after {max_iterations} iterations")

simpath = SimInfo[args.simulation]['path']
halos = SimInfo[args.simulation]['goodhalos']


print(f'Loading {args.simulation}')
tstart = time.time()
sim = pynbody.load(simpath)
sim.physical_units()
h = sim.halos()
ShapeData = pymp.shared.dict()
myprint(f'{args.simulation} loaded.',clear=True)

prog=pymp.shared.array((1,),dtype=int)
print(f'\tCalculating Shapes: 0.00%')
with (pymp.Parallel(args.numproc) as pl):
    for i in pl.xrange(len(halos)):
        t_start_current = time.time()
        if args.verbose: print(f'\tAnalyzing {halos[i]}...')
        hid = halos[i]
        halo = h[hid]
        current = {}
        try:
            if dark_only == True:
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
                #get rout from stars in data_dir / f'{args.simulation}.{args.feedback}.3DShapes.pickle'
                try:
                    rbins = StellarShape[str(hid)]['rbins']
                    rin = rbins[0]
                    rout = rbins[-1]
                except:
                    print(f'Could not find rout for halo {hid}. Using default value.')
                    rout = sim.s['r'].max()

                #change bins for dark matter though
                n = len(halo.dm[halo.dm['r'] < rout])
                # check that there at at least 10^4 particles
                if n < 10 ** 4:
                    # display warnimg
                    print(f"Warning: {n} particles in halo {hid}")
                    bins = 20
                    n_i = int(n / bins)
                else:
                    bins = int((np.log10(n) - 3) ** 2 * 20)
                    # set inner radius to .25kpc
                    n_i = int(n / bins)





                print(f'rin: {rin}, rout: {rout}, bins: {bins}')
                rbins, ba, ca, angle, Es = halo_shape(halo.dm,N=bins,rin=rin/2,rout=rout,bins='equal')
                print(rbins[0], rbins[-1])
                current['rbins'] = rbins
                current['ba'] = ba
                current['ca'] = ca
                current['angle'] = angle
                current['Es'] = Es
                #get number of dm pariticles in the region with stars halo.dm['r'] < rout
                
                current['N_d'] = len(halo.dm[halo.dm['r'] < rout])
                #print warning if this number is less than 1000
                if current['N_d'] < 1000:
                    print(f'Warning: {current["N_d"]} DM particles found in region with stars {hid}.')

        except Exception as e:
            print(f"An error occurred for halo {hid}: {e}")
            print(traceback.format_exc())
            current['rbins'] = np.array([])
            current['ba'] = np.array([])
            current['ca'] = np.array([])
            current['angle'] = 0
            current['Es'] = np.identity(3)
            current['N_d'] = (np.nan) 

        ShapeData[str(hid)] = current
        prog[0]+=1
        if not args.verbose: myprint(f'\tCalculating Shapes: {round(prog[0]/len(halos)*100,2)}%',clear=False)
        t_end_current = time.time()
        if args.verbose: print(f'\t\t{hid} done in {round((t_end_current-t_start_current)/60,2)} minutes.')



#convert to regular dictionary
ShapeData = dict(ShapeData)

ShapeFile = {}
for halo in halos:
    ShapeFile[str(halo)] = ShapeData[str(halo)]

file_path = data_dir / f'{args.simulation}.{args.feedback}.DMShapes.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(ShapeFile, file)

tstop = time.time()
myprint(f'\t{args.simulation} finished in {round((tstop-tstart)/60,2)} minutes.',clear=True)