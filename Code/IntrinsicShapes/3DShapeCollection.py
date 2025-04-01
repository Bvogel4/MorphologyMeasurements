
import argparse, os, pickle, pymp, pynbody, sys, time, warnings
import numpy as np
import traceback
# from ShapeFunctions.Functions import halo_shape_stellar
from pathlib import Path
import sys


from pymp.config import num_threads

# Current script directory (3DShapesCollector.py's location)
current_dir = Path(__file__).parent

shape_functions_dir = current_dir.parent / 'ShapeFunctions'
root_dir = current_dir.parent.parent
code_dir = root_dir / 'Code'
pickle_dir = code_dir / 'PickleFiles'
data_dir = root_dir / 'Data'

# Convert the path to a string and append it to sys.path
sys.path.append(str(shape_functions_dir))


import argparse, os, pickle, pymp, pynbody, sys, time, warnings, traceback
import numpy as np
from pathlib import Path
import sys

from pynbody import array
from pynbody.analysis import profile
import numpy as np
import logging
import multiprocessing as mp
from functools import partial

logger = logging.getLogger('pynbody.analysis.halo')
from numpy import matmul as X
from numpy import sin, cos
from skimage.measure import EllipseModel
from math import pi


def shape(sim, nbins=100, rmin=None, rmax=None, bins='equal',
          ndim=3, max_iterations=100, tol=1e-3, justify=False):
    """Calculates the shape of the provided particles in homeoidal shells, over a range of nbins radii.

    Homeoidal shells maintain a fixed area (ndim=2) or volume (ndim=3). Note that all provided particles are used in
    calculating the shape, so e.g. to measure dark matter halo shape from a halo with baryons, you should pass
    only the dark matter particles.

    The simulation must be pre-centred, e.g. using :func:`center`.

    The algorithm is sensitive to substructure, which should ideally be removed.

    Caution is advised when assigning large number of bins and radial ranges with many particles, as the
    algorithm becomes very slow.

    Parameters
    ----------

      nbins : int
          The number of homeoidal shells to consider. Shells with few particles will take longer to fit.

      rmin : float
          The minimum radial bin in units of sim['pos']. By default this is taken as rout/1000.
          Note that this applies to axis a, so particles within this radius may still be included within
          homeoidal shells.

      rmax : float
          The maximum radial bin in units of sim['pos']. By default this is taken as the largest radial value
          in the halo particle distribution.

      bins : str
          The spacing scheme for the homeoidal shell bins. 'equal' initialises radial bins with equal numbers
          of particles, with the exception of the final bin which will accomodate remainders. This
          number is not necessarily maintained during fitting. 'log' and 'lin' initialise bins
          with logarithmic and linear radial spacing.

      ndim : int
          The number of dimensions to consider; either 2 or 3 (default). If ndim=2, the shape is calculated
          in the x-y plane. If using ndim=2, you may wish to make a cut in the z direction before
          passing the particles to this routine (e.g. using :class:`pynbody.filt.BandPass`).

      max_iterations : int
          The maximum number of shape calculations (default 10). Fewer iterations will result in a speed-up,
          but with a bias towards spheroidal results.

      tol : float
          Convergence criterion for the shape calculation. Convergence is achieved when the axial ratios have
          a fractional change <=tol between iterations.

      justify : bool
          Align the rotation matrix directions such that they point in a single consistent direction
          aligned with the overall halo shape. This can be useful if working with slerps.

    Returns
    -------

      rbin : SimArray
          The radial bins used for the fitting

      axis_lengths : SimArray
          A nbins x ndim array containing the axis lengths of the ellipsoids in each shell

      num_particles : np.ndarray
          The number of particles within each bin

      rotation_matrices : np.ndarray
          The rotation matrices for each shell

    """

    # Sanitise inputs:
    if (rmax == None): rmax = sim['r'].max()
    if (rmin == None): rmin = rmax / 1E3
    assert ndim in [2, 3]
    assert max_iterations > 0
    assert tol > 0
    assert rmin >= 0
    assert rmax > rmin
    assert nbins > 0
    if ndim == 2:
        assert np.sum((sim['rxy'] >= rmin) & (sim['rxy'] < rmax)) > nbins * 2
    elif ndim == 3:
        assert np.sum((sim['r'] >= rmin) & (sim['r'] < rmax)) > nbins * 2
    if bins not in ['equal', 'log', 'lin']: bins = 'equal'

    # Handy 90 degree rotation matrices:
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # -----------------------------FUNCTIONS-----------------------------
    sn = lambda r, N: np.append([r[i * int(len(r) / N):(1 + i) * int(len(r) / N)][0] \
                                 for i in range(N)], r[-1])

    # General equation for an ellipse/ellipsoid:
    def Ellipsoid(pos, a, R):
        x = np.dot(R.T, pos.T)
        return np.sum(np.divide(x.T, a) ** 2, axis=1)

    # Define moment of inertia tensor:
    def MoI(r, m, ndim=3):
        return np.array([[np.sum(m * r[:, i] * r[:, j]) for j in range(ndim)] for i in range(ndim)])

    # Calculate the shape in a single shell:
    def shell_shape(r, pos, mass, a, R, r_range, ndim=3):

        # Find contents of homoeoidal shell:
        mult = r_range / np.mean(a)
        in_shell = (r > min(a) * mult[0]) & (r < max(a) * mult[1])
        pos, mass = pos[in_shell], mass[in_shell]
        inner = Ellipsoid(pos, a * mult[0], R)
        outer = Ellipsoid(pos, a * mult[1], R)
        in_ellipse = (inner > 1) & (outer < 1)
        ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

        # End if there is no data in range:
        if not len(ellipse_mass):
            return a, R, np.sum(in_ellipse)

        # Calculate shape tensor & diagonalise:
        D = list(np.linalg.eigh(MoI(ellipse_pos, ellipse_mass, ndim) / np.sum(ellipse_mass)))

        # Rescale axis ratios to maintain constant ellipsoidal volume:
        R2 = np.array(D[1])
        a2 = np.sqrt(abs(D[0]) * ndim)
        div = (np.prod(a) / np.prod(a2)) ** (1 / float(ndim))
        a2 *= div

        return a2, R2, np.sum(in_ellipse)

    # Re-align rotation matrix:
    def realign(R, a, ndim):
        if ndim == 3:
            if a[0] > a[1] > a[2] < a[0]:
                pass  # abc
            elif a[0] > a[1] < a[2] < a[0]:
                R = np.dot(R, Rx)  # acb
            elif a[0] < a[1] > a[2] < a[0]:
                R = np.dot(R, Rz)  # bac
            elif a[0] < a[1] > a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Ry)  # bca
            elif a[0] > a[1] < a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Rz)  # cab
            elif a[0] < a[1] < a[2] > a[0]:
                R = np.dot(R, Ry)  # cba
        elif ndim == 2:
            if a[0] > a[1]:
                pass  # ab
            elif a[0] < a[1]:
                R = np.dot(R, Rz[:2, :2])  # ba
        return R

    # Calculate the angle between two vectors:
    def angle(a, b):
        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Flip x,y,z axes of R2 if they provide a better alignment with R1.
    def flip_axes(R1, R2):
        for i in range(len(R1)):
            if angle(R1[:, i], -R2[:, i]) < angle(R1[:, i], R2[:, i]):
                R2[:, i] *= -1
        return R2

    def process_bin(i, r, pos, mass, rbins, bin_edges, ndim):
        a = np.ones(ndim) * rbins[i]
        R = np.identity(ndim)

        for j in range(max_iterations):
            a2 = a.copy()
            a, R, N = shell_shape(r, pos, mass, a, R, bin_edges[[i, i + 1]], ndim)

            convergence_criterion = np.all(np.isclose(np.sort(a), np.sort(a2), rtol=tol))
            if convergence_criterion:
                R = realign(R, a, ndim)
                if np.sign(np.linalg.det(R)) == -1:
                    R[:, 1] *= -1
                a = np.flip(np.sort(a))
                return i, a, R, N

        return i, np.ones(ndim) * np.nan, np.identity(ndim) * np.nan, 0


    # -----------------------------FUNCTIONS-----------------------------

    # Set up binning:
    r = np.array(sim['r']) if ndim == 3 else np.array(sim['rxy'])
    pos = np.array(sim['pos'])[:, :ndim]
    mass = np.array(sim['mass'])

    if (bins == 'equal'):  # Bins contain equal number of particles
        full_bins = sn(np.sort(r[(r >= rmin) & (r <= rmax)]), nbins * 2)
        bin_edges = full_bins[0:nbins * 2 + 1:2]
        rbins = full_bins[1:nbins * 2 + 1:2]
    elif (bins == 'log'):  # Bins are logarithmically spaced
        bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
        rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif (bins == 'lin'):  # Bins are linearly spaced
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialise the shape arrays:
    rbins = array.SimArray(rbins, sim['pos'].units)
    axis_lengths = array.SimArray(np.zeros([nbins, ndim]), sim['pos'].units) * np.nan
    N_in_bin = np.zeros(nbins).astype('int')
    #create an array with n R = np.identity(ndim)
    rotations = np.array([np.identity(ndim)] * nbins) * np.nan

    # Calculate the shape in each bin:
    threads = num_threads
    #create shared objects axis_lengths, N_in_bin, rotations
    shared_results = pymp.shared.dict()


    with pymp.Parallel(threads) as p:
        for i in p.range(nbins):
            #p.print(f'Processing bin {i}')
            i, a, R, N = process_bin(i, r, pos, mass, rbins, bin_edges, ndim)
            #store in results
            with p.lock:
                shared_results[i] = (a, R, N)
    #unpack results
    results = dict(shared_results)
    for i in range(nbins):
        a, R, N = results[i]
        axis_lengths[i] = a
        N_in_bin[i] = N
        rotations[i] = R


    # Ensure the axis vectors point in a consistent direction:
    if justify:
        _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
        rotations = np.array([flip_axes(R_global, i) for i in rotations])
    #print(rotations[0])
    axis_lengths = np.squeeze(axis_lengths.T).T
    rotations = np.squeeze(rotations)

    return rbins, axis_lengths, N_in_bin, rotations

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


def get_bins(n):
    n = n*.9
    if n < 1e4:
        return int(20 * n/1e4 /2 + 10)
    elif n >= 1e4:
        return int( (np.log10(n) - 3) ** 2 * 20)



#custom to rerun specific simulations r431, r615 ,r618, r634 even without overwrite
# if args.simulation in ['r615','r618','r634']:
#     args.overwrite = True

#try and load output file if it already exists
#ignore if overwrite is set
        # Function to process shape for given parameters
def process_shape(particles, rin, rout, bins):
    shape_data = {}
    rbins, axis_lengths, num_particles, rotations = shape(particles, nbins=bins, ndim=3, rmin=rin, rmax=rout, max_iterations=125, tol=1e-2, justify=False)
    #for testing 1e-2, as it's faster
    ba = axis_lengths[:, 1] / axis_lengths[:, 0]
    ca = axis_lengths[:, 2] / axis_lengths[:, 0]
    shape_data['rbins'] = rbins
    shape_data['ba'] = ba
    shape_data['ca'] = ca
    shape_data['rotations'] = rotations
    shape_data['N'] = num_particles
    return shape_data

#noinspection PyUnreachableCode
def process_halo(hid):
    #center halo
    #pynbody.analysis.halo.center(hid)
    #maybe in the future change this to face-on
    pynbody.analysis.angmom.faceon(hid)
    print(f'Analyzing halo {hid}')

    try:
        # Find the stellar shape of the halo (original run)
        try:
            # Find the number of particles in the halo
            N_star = len(hid.s)
            rin = 0.1
            if N_star  == 0:
                print(f'No star particles in halo {hid}.')
                rout = hid.dm['r'].max() * 0.3
                starshape = None
            else:

                # Find the number of bins to use

                # get radius that contains 80% of star particles
                rsort = hid.s['r'][np.argsort(hid.s['r'])]
                r_8 = rsort[int(0.8 * N_star)]
                rout = rsort[-1]
                #rout = r_8*2 # this one is pretty close with 5e-3 tol
                #print(hid.s['r'].max().in_units('kpc'),hid.s['r'].min().in_units('kpc'))
                # if rout < rin:
                #     rout = None
                #     rin = None #let the function decide

                bins = get_bins(N_star)
                starshape = process_shape(hid.s, rin, rout, bins)
                starshape['r_80'] = r_8
                r_star_max = starshape['rbins'][-1]
                rout = r_star_max

            N_dark = len(hid.d['r'][hid.d['r'] < rout])
            bins = get_bins(N_dark)
            darkshape = process_shape(hid.d, rin, rout, bins)


        except Exception as e:
            print(f'Error in halo {hid}: {e}')
            traceback.print_exc()
            return None, None

        return starshape, darkshape

    except Exception as e:
        print(f'Error in halo {hid}: {e}')
        traceback.print_exc()
        return None, None

# def process_halo(hid):
#     #center halo
#     pynbody.analysis.halo.center(hid)
#     print(f'Analyzing halo {hid}')
#
#     try:
#         # Find the number of particles in the halo
#         N_star = len(hid.s)
#
#         # Find the number of bins to use
#
#         # get radius that contains 80% of star particles
#         rsort =  hid.s['r'][np.argsort(hid.s['r'])]
#         r_8 = rsort[int(0.8*N_star)]
#         rout = rsort[-1]
#         bins = get_bins(N_star)
#         rin = 0.1
#
#         # Find the stellar shape of the halo
#         try:
#             starshape = {}
#             rbins, axis_lengths, num_particles, rotations = shape(hid.s, nbins=bins, ndim=3, rmin=rin, rmax=rout, max_iterations=125, tol=1e-2, justify=False)
#             r_star_max = rbins[-3]
#             ba = axis_lengths[:, 1] / axis_lengths[:, 0]
#             ca = axis_lengths[:, 2] / axis_lengths[:, 0]
#
#
#
#             starshape['rbins'] = rbins
#             starshape['ba'] = ba
#             starshape['ca'] = ca
#             starshape['rotations'] = rotations
#             starshape['N_s'] = num_particles
#             starshape['r_80'] = r_8
#
#             rout = r_star_max
#             # get the dark matter shape of the halo
#             N_dark = len(hid.d['r'][hid.d['r'] < rout])
#             bins = get_bins(N_dark)
#
#
#             darkshape = {}
#
#             rbins, axis_lengths, num_particles, rotations = shape(hid.d, nbins=bins, ndim=3, rmin=rin, rmax=rout, max_iterations=50, tol=1e-2, justify=False)
#             ba = axis_lengths[:, 1] / axis_lengths[:, 0]
#             ca = axis_lengths[:, 2] / axis_lengths[:, 0]
#
#
#
#             darkshape['rbins'] = rbins
#             darkshape['ba'] = ba
#             darkshape['ca'] = ca
#             darkshape['rotations'] = rotations
#             darkshape['N_d'] = num_particles
#
#         except Exception as e:
#             print(f'Error in halo {hid}: {e}')
#             traceback.print_exc()
#             return None, None
#
#         return starshape, darkshape
#
#     except Exception as e:
#         print(f'Error in halo {hid}: {e}')
#         traceback.print_exc()
#         return None, None

num_threads = 1
def main():
    global num_threads
    parser = argparse.ArgumentParser(description='Collect 3D shapes of all resolved halos from a given simulation.')
    parser.add_argument('-f','--feedback',default='BW',help='Feedback Model')
    parser.add_argument('-s','--simulation',required=True,help='Simulation to analyze')
    parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
    parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing data file')
    args = parser.parse_args()

    if not args.overwrite:
        if os.path.exists(root_dir / 'Data' / f'{args.simulation}.{args.feedback}.3DShapes.pickle'):
            print(f'Output file found for {args.simulation}. Exiting...')
            sys.exit()

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
    print(f'{args.simulation} loaded.')
    num_threads = args.numproc


    # Process halos
    results = {}
    for hid in halos:
        try:
            results[hid] = process_halo(h[hid])
        except Exception as e:
            print(f'Error in halo {hid}: {e}')
            print(traceback.format_exc())
            traceback.print_exc()
            results[hid] = None

    # Combine results
    StarShapeData = {}
    DarkShapeData = {}
    #print(results)
    for hid, (starshape, darkshape) in results.items():
        if starshape is not None:
            StarShapeData[hid] = starshape
        else:
            print(f'Error in halo {hid} (stellar) or no star particles.')
        if darkshape is not None:
            DarkShapeData[hid] = darkshape
        else:
            print(f'Error in halo {hid} (dark matter).')

    #Save results
    file_path = data_dir / f'{args.simulation}.{args.feedback}.3DShapes.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(StarShapeData, file)
    file_path = data_dir / f'{args.simulation}.{args.feedback}.DMShapes.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(DarkShapeData, file)

    print(f'Analysis completed in {(time.time() - tstart)/60:.2f} minutes.')

    

if __name__ == '__main__':
    main()
