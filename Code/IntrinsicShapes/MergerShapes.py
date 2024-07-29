
import os,pickle
import matplotlib.pylab as plt
from scipy.interpolate import UnivariateSpline as Smooth
from pynbody import array
import argparse
import traceback
import pathlib
import sys
import pandas as pd
import pynbody
import numpy as np

def shape(sim, nbins=100, rmin=None, rmax=None, bins='equal',
          ndim=3, max_iterations=10, tol=1e-3, justify=False):
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
    axis_lengths = array.SimArray(np.zeros([nbins, ndim]), sim['pos'].units)
    N_in_bin = np.zeros(nbins).astype('int')
    rotations = [0] * nbins

    # Loop over all radial bins:
    for i in range(nbins):

        # Initial spherical shell:
        a = np.ones(ndim) * rbins[i]
        a2 = np.zeros(ndim)
        a2[0] = np.inf
        R = np.identity(ndim)

        # Iterate shape estimate until a convergence criterion is met:
        iteration_counter = 0
        while ((np.abs(a[1] / a[0] - np.sort(a2)[-2] / max(a2)) > tol) & \
               (np.abs(a[-1] / a[0] - min(a2) / max(a2)) > tol)) & \
                (iteration_counter < max_iterations):
            a2 = a.copy()
            a, R, N = shell_shape(r, pos, mass, a, R, bin_edges[[i, i + 1]], ndim)
            iteration_counter += 1

        # Adjust orientation to match axis ratio order:
        R = realign(R, a, ndim)

        # Ensure consistent coordinate system:
        if np.sign(np.linalg.det(R)) == -1:
            R[:, 1] *= -1

        # Update profile arrays:
        a = np.flip(np.sort(a))
        axis_lengths[i], rotations[i], N_in_bin[i] = a, R, N

    # Ensure the axis vectors point in a consistent direction:
    if justify:
        _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
        rotations = np.array([flip_axes(R_global, i) for i in rotations])

    return rbins, np.squeeze(axis_lengths.T).T, N_in_bin, np.squeeze(rotations)


def get_shapes(current, sims, hid, snapshot, SimInfo):
    simpath = SimInfo[sims]['path']  # this is only good for the '004096' snapshot
    # if not the '004096' snapshot, need to change the path by replacing instances of '004096' with the snapshot number
    simpath = simpath.replace('004096', snapshot)

    print(f'Loading {simpath}')

    simSnap = pynbody.load(simpath)
    simSnap.physical_units()
    halos = simSnap.halos()
    halo = halos[hid]
    pynbody.analysis.angmom.faceon(halo)
    print(f'Loaded {simpath}')
    print(f'Calculating stellar shapes for {sims}.{hid}.{snapshot}')
    # calculate stellar shape
    r_s, axis_lengths, num_particles, rotation_matrices = shape(halo.s, nbins=100, max_iterations=20, tol=1e-2,
                                                                justify=True)
    ba_s = axis_lengths[:, 1] / axis_lengths[:, 0]
    ca_s = axis_lengths[:, 2] / axis_lengths[:, 0]
    rot_s = rotation_matrices
    rin = r_s[0]
    rout = r_s[-1]
    # calculate dark matter shape
    print(f'Calculating dark matter shapes for {sims}.{hid}.{snapshot}')
    r_d, axis_lengths, num_particles, rotation_matrices = shape(halo.d, rmin = rin, rmax = rout,
                                                                nbins=100, max_iterations=20, tol=1e-2,
                                                                justify=True)
    ba_d = axis_lengths[:, 1] / axis_lengths[:, 0]
    ca_d = axis_lengths[:, 2] / axis_lengths[:, 0]
    rot_d = rotation_matrices
    print(f'Calculating smoothed shapes for {sims}.{hid}.{snapshot}')

    ba_s_smoothed, ca_s_smoothed = Smooth(r_s, ba_s, k=3), Smooth(r_s, ca_s, k=3)
    ba_d_smoothed, ca_d_smoothed = Smooth(r_d, ba_d, k=3), Smooth(r_d, ca_d, k=3)


    # save data
    current['ba_s'] = ba_s
    current['ca_s'] = ca_s
    current['rot_s'] = rot_s
    current['ba_d'] = ba_d
    current['ca_d'] = ca_d
    current['rot_d'] = rot_d
    current['r_s'] = r_s
    current['r_d'] = r_d
    current['ba_s_smoothed'] = ba_s_smoothed
    current['ca_s_smoothed'] = ca_s_smoothed
    current['ba_d_smoothed'] = ba_d_smoothed
    current['ca_d_smoothed'] = ca_d_smoothed
    current['ba_s_smoothed_at_Reff'] = ba_s_smoothed(Reff)
    current['ca_s_smoothed_at_Reff'] = ca_s_smoothed(Reff)
    current['ba_d_smoothed_at_Reff'] = ba_d_smoothed(Reff)
    current['ca_d_smoothed_at_Reff'] = ca_d_smoothed(Reff)

    return current


def plot(ax, current, sims, hid, snapshot,Reff):
    r_s = current['r_s']
    ba_s = current['ba_s']
    ca_s = current['ca_s']

    r_d = current['r_d']
    ba_d = current['ba_d']
    ca_d = current['ca_d']

    ba_s_smoothed = current['ba_s_smoothed']
    ca_s_smoothed = current['ca_s_smoothed']
    ba_d_smoothed = current['ba_d_smoothed']
    ca_d_smoothed = current['ca_d_smoothed']

    # r vs ba, r vs ca
    ax[0].plot(r_s, ba_s, label='Stellar', c='k')
    ax[0].plot(r_d, ba_d, label='Dark Matter', c='r')
    ax[0].plot(r_s, ba_s_smoothed(r_s), label='Stellar Smoothed', c='k', ls='--')
    ax[0].plot(r_d, ba_d_smoothed(r_d), label='Dark Matter Smoothed', c='r', ls='--')
    ax[0].axvline(Reff, c='gray', ls='--')
    ax[0].set_ylabel('b/a')
    ax[0].legend()
    ax[0].set_ylim(0, 1)



    ax[1].plot(r_s, ca_s, label='Stellar', c='k')
    ax[1].plot(r_d, ca_d, label='Dark Matter', c='b')
    ax[1].plot(r_s, ca_s_smoothed(r_s), label='Stellar Smoothed', c='k', ls='--')
    ax[1].plot(r_d, ca_d_smoothed(r_d), label='Dark Matter Smoothed', c='b', ls='--')
    ax[1].axvline(Reff, c='gray', ls='--')
    ax[1].set_ylabel('c/a')
    ax[1].legend()
    ax[1].set_ylim(0, 1)


    # make directory if it doesn't exist



feedbacks = ['BWMDC', 'MerianCDM']

sim_repo = '/data/REPOSITORY/'




for feedback in feedbacks:
    try:
        merger_shapes = pickle.load(open(f'../../Data/MergerShapes.{feedback}.pickle', 'rb'))
    except:
        merger_shapes = {}
    #mergers is a dataframe with columns: sim  halo    004096    004032
    #where the numbers are the snapshot numbers of the merger
    #sim is the simulation name
    #halo is the halo number
    # the values are merger rations, nan if no merger
    merger_ratios = pd.read_pickle(f'../../Data/BasicData/Mergers.{feedback}.pickle')
    # idenitfy what sim and halo mergers are in
    SimInfo = pickle.load(open(f'../PickleFiles/SimulationInfo.{feedback}.pickle', 'rb'))


    for sims in SimInfo:

        #check if the sim has non nan mergers
        sim_mergers = merger_ratios[merger_ratios['sim'] == sims]
        #remove nans
        sim_mergers = sim_mergers.dropna()
        if len(sim_mergers) == 0:
            continue
        Profiles = pickle.load(open(f'../../Data/{sims}.{feedback}.Profiles.pickle', 'rb'))
        merger_shapes[sims] = {}
        for hid in SimInfo[sims]['goodhalos']:

            #check if halo has mergers
            halo_mergers = sim_mergers[sim_mergers['halo'] == hid]
            if len(halo_mergers) == 0:
                continue
            merger_shapes[sims][hid] = {}
            fig,axes = plt.subplots(3,2,figsize=(10,10))
            #print(Profiles[str(hid)]['x000y000'].keys())
            Reff = Profiles[str(hid)]['x000y000']['Reff']  # just using Reff from the 4096 snapshot
            
            i=0
            for snapshot in ['004096', '004032', '003936']:
                merger_shapes[sims][hid][snapshot] = {}
                current = {}
                #check if halo has mergers in snapshot
                #snap_mergers = halo_mergers[snapshot]
                #if np.isnan(snap_mergers):
                #    continue
                try:
                    current = get_shapes(current,sims, hid, snapshot, SimInfo)
                    #plot the shapes on the ith row of the axes
                    try:
                        plot(axes[i], current, sims, hid, snapshot, Reff)
                    except Exception as e:
                        print(f'Error in plotting {sims}.{hid}.{snapshot}')
                        print(e)
                        print(traceback.format_exc())
                    #label row with merger ratio if not nan
                    #if not np.isnan(snap_mergers):
                        #axes[i][0].set_title(f'Merger Ratio: {snap_mergers}')
                    axes[i][0].set_title(f'{snapshot}')
                    #save the data
                    merger_shapes[sims][hid][snapshot] = current
                    try:
                        pickle.dump(merger_shapes, open(f'../../Data/MergerShapes.{feedback}.pickle', 'wb'))
                    except Exception as e:
                        print(f'Error in saving {sims}.{hid}.{snapshot}')
                        print(e)
                        print(traceback.format_exc())
                except Exception as e:
                    print(f'Error in {sims}.{hid}.{snapshot}')
                    print(e)
                    print(traceback.format_exc())
                i += 1


            #add shared x axis label
            axes[2][0].set_xlabel('r [kpc]')
            axes[2][1].set_xlabel('r [kpc]')
            filepath = f'../../Figures/MergerShapes/{sims}.{hid}.RecentMergerShape.png'
            pathlib.Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath)
            plt.close()







