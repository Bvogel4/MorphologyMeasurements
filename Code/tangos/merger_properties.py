from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties import LivePropertyCalculation
from tangos.properties.pynbody.centring import centred_calculation
from tangos.properties import LivePropertyCalculationInheritingMetaProperties
import pynbody
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from pynbody import array
import traceback
from pynbody.analysis import profile
import logging
logger = logging.getLogger('pynbody.analysis.halo')

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


#Pynbody Shape functions for stellar morphology
def halo_shape_stellar(sim, N=100, rin=None, rout=None, bins='equal', ret_pos=False):
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

    if (rout == None): rout = sim.s['r'].max()
    if (rin == None): rin = rout/1E3

    posr = np.array(sim.s['r'])[np.where(sim.s['r'] < rout)]
    pos = np.array(sim.s['pos'])[np.where(sim.s['r'] < rout)]
    mass = np.array(sim.s['mass'])[np.where(sim.s['r'] < rout)]

    rx = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    ry = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    rz = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]

    # Define bins:
    if (bins == 'equal'): # Each bin contains equal number of particles
        mid = sn(np.sort(posr[np.where((posr >= rin) & (posr <= rout))]),N*2)
        rbin = mid[1:N*2+1:2]
        mid = mid[0:N*2+1:2]

    elif (bins == 'log'): # Bins are logarithmically spaced
        mid = profile.Profile(sim.s, type='log', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = np.sqrt(mid[0:N]*mid[1:N+1])

    elif (bins == 'lin'): # Bins are linearly spaced
        mid = profile.Profile(sim.s, type='lin', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = 0.5*(mid[0:N]+mid[1:N+1])

    # Define b/a and c/a ratios and angle arrays:
    ba,ca,angle,aout,n,n_i,pos_out,pos_in = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),[],[]
    Es = [0]*N

    # Begin loop through radii:
    for i in range(0,N):

        # Initialise convergence criterion:
        tol = .1
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
            num = len(r)
            if (count == 1):
                num_i = len(r)
                pos_in.append(r)

            # End iterations if there is no data in range:
            if (len(r) == 0):
                ba[i],ca[i],angle[i],Es[i],n_i[i] = b/a,c/a,almnt(E),E,num_i
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
                aout[i],ba[i],ca[i],angle[i],Es[i],n[i],n_i[i] = anew,bnew/anew,cnew/anew,almnt(E),E,num,num_i
                pos_out.append(r)
                break

            # Increase tolerance if convergence has stagnated:
            elif (count%10 == 0): tol *= 5.

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew
    if ret_pos:
        return [array.SimArray(rbin,sim.d['pos'].units),aout,ba,ca,angle,Es,n,n_i,np.array(pos_out),np.array(pos_in)]
    else:
        return [array.SimArray(rbin,sim.d['pos'].units),aout,ba,ca,angle,Es,n,n_i]

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
            elif (count%10 == 0): tol *= 5.

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew

    return [array.SimArray(rbin, sim.d['pos'].units), ba, ca, angle, Es]


class SmoothAxisRatio(LivePropertyCalculation):
    names = ['ba_s_smoothed', 'ba_d_smoothed', 'ca_s_smoothed', 'ca_d_smoothed', 'ba_s_at_reff', 'ba_d_at_reff', 'ca_s_at_reff', 'ca_d_at_reff']

    def requires_property(self):
        return ['ba_s', 'ba_d', 'ca_s', 'ca_d', 'rbins', 'reff']
    @staticmethod
    def smooth_shape(x, y, k=3):
        return UnivariateSpline(x, y, k=k)

    def calculate(self, halo, existing_properties):
        rbins = existing_properties['rbins']
        ba_s_smoothed = self.smooth_shape(rbins, existing_properties['ba_s'])
        ba_d_smoothed = self.smooth_shape(rbins, existing_properties['ba_d'])
        ca_s_smoothed = self.smooth_shape(rbins, existing_properties['ca_s'])
        ca_d_smoothed = self.smooth_shape(rbins, existing_properties['ca_d'])
        reff = existing_properties['reff']
        ba_s_reff = ba_s_smoothed(reff)
        ba_d_reff = ba_d_smoothed(reff)
        ca_s_reff = ca_s_smoothed(reff)
        ca_d_reff = ca_d_smoothed(reff)
        return ba_s_smoothed(rbins), ba_d_smoothed(rbins), ca_s_smoothed(rbins), ca_d_smoothed(rbins), ba_s_reff, ba_d_reff, ca_s_reff, ca_d_reff




class Shape(PynbodyPropertyCalculation):
    """
    Calculate the axis ratio of the stellar and dark matter components of a halo, only for where stars and dark matter both exist.
    """
    names = ['ba_s', 'ba_d', 'ca_s', 'ca_d', 'rbins']
    def __init__(self, simulation):
        super().__init__(simulation)



    def _get_shape(self, halo):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)

        nbins = 100
        #get stellar shapes
        stars = halo.s
        #rbins, axis_s, D_in_bin,rot_s = shape(stars,nbins=nbins,rmin=None,rmax=None,ndim=3,bins='equal',max_iterations=20,tol=1e-3)
        rbins, a, ba_s, ca_s, angle, Es, n, ni =  halo_shape_stellar(stars, N=nbins, rin=None, rout=None, bins='equal')

        #self.rbins = rbins
        #get dark matter shapes
        rmin = rbins[0]
        rmax = rbins[-1]
        dm = halo.dm
        #r_d, axis_d, D_in_bin,rot_d = shape(dm,nbins=nbins,rmin=rmin,rmax=rmax,ndim=3,bins='equal',max_iterations=20,tol=1e-3)
        r_d, ba_d, ca_d, angle, Es = halo_shape(dm, N=nbins, rin=rmin, rout=rmax, bins='equal')
        # a_s = axis_s[:,0]
        # b_s = axis_s[:,1]
        # c_s = axis_s[:,2]
        # ba_s = b_s/a_s
        # ca_s = c_s/a_s


        # a_d = axis_d[:,0]
        # b_d = axis_d[:,1]
        # c_d = axis_d[:,2]
        # ba_d = b_d/a_d
        # ca_d = c_d/a_d

        #assert that all axis ratios are between 0 and 1
        assert np.all(ba_s <= 1) and np.all(ba_s >= 0)
        assert np.all(ca_s <= 1) and np.all(ca_s >= 0)
        assert np.all(ba_d <= 1) and np.all(ba_d >= 0)
        assert np.all(ca_d <= 1) and np.all(ca_d >= 0)


        return ba_s, ba_d, ca_s, ca_d, rbins.in_units('kpc')
    def calculate(self, halo, existing_properties):
        ba_s, ba_d, ca_s, ca_d, rbins = self._get_shape(halo)
        return ba_s, ba_d, ca_s, ca_d , rbins


    def plot_xlog(self):
        return False

    def plot_ylog(self):
        return False


class SersicFit(PynbodyPropertyCalculation):
    names = ['reff','rhalf']

    @staticmethod
    def sersic(r, mueff, reff, n):
        return mueff + 2.5 * (0.868 * n - 0.142) * ((r / reff) ** (1. / n) - 1)
    

    def calculate(self, halo, existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        # Get the surface density profile
        try:
            Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        except:
            Rhalf = np.nan
        try:
            prof = pynbody.analysis.profile.Profile(halo.s,type='lin',min=.25,max=5*Rhalf,ndim=2,nbins=int((5*Rhalf)/0.1))
            vband = prof['sb,V']
            smooth = np.nanmean(
                np.pad(vband.astype(float), (0, 3 - vband.size % 3), mode='constant', constant_values=np.nan).reshape(
                    -1, 3), axis=1)
            x = np.arange(len(smooth)) * 0.3 + 0.15
            x[0] = .05
            if True in np.isnan(smooth):
                x = np.delete(x, np.where(np.isnan(smooth) == True))
                y = np.delete(smooth, np.where(np.isnan(smooth) == True))
            else:
                y = smooth
            r0 = x[int(len(x) / 2)]
            m0 = np.mean(y[:3])
            par, ign = curve_fit( self.sersic, x, y, p0=(m0, r0, 1), bounds=([10, 0, 0.5], [40, 100, 16.5]) )
            reff = pynbody.array.SimArray(par[1], 'kpc')
        except:
            print("Sersic fit failed")
            print(traceback.format_exc())
            #set reff to value of later halo
            try:
                reff = halo.calculate('later(1).reff')
            except:
                reff = np.nan
        return reff, Rhalf
        # except:
        #     print("Sersic fit failed")
        #     print(traceback.format_exc())
        #     return np.nan



class dynamical_time(PynbodyPropertyCalculation):
    names = ['tdyn']

    def requires_property(self):
        return ['rbins']

    def calculate(self, halo, existing_properties):
        pynbody.analysis.angmom.faceon(halo)
        rbins = existing_properties['rbins']
        prof = pynbody.analysis.profile.Profile(halo,bins=rbins,ndim=2)
        mass_enc = prof['mass_enc']
        dyntime = (rbins ** 3 / (2 * pynbody.units.G * mass_enc)) ** (1 / 2)
        return dyntime





class BaryonicFractionReff(PynbodyPropertyCalculation):

    names = ['Mvir_within_reff', 'Mstar_within_reff', 'Mgas_within_reff', 'Mb_mvir_within_reff']

    def requires_property(self):
        return ['reff','max_radius']
    @staticmethod
    def mass_properties_within_r(halo, r):
        # halo should be in physcial units, but just in case
        halo.physical_units()

        sphere_filter = pynbody.filt.Sphere(r)
        sphere = halo[sphere_filter]

        m_tot = (sphere['mass'].sum().in_units('Msol'))
        m_gas = (sphere.gas['mass'].sum().in_units('Msol'))
        m_star = (sphere.star['mass'].sum().in_units('Msol'))
        m_dm = (sphere.dm['mass'].sum().in_units('Msol'))
        m_vir_within_r = m_gas + m_star + m_dm
        # assert that all of these values are positive, and not close to 0 they are stored as pynbody SimArrays in units of solar masses
        # assert that m_tot is the sum of the other masses within floating point error
        assert np.isclose(m_tot, m_vir_within_r,
                          rtol=1e-10), f"Total mass is {m_tot}, sum of components is {m_gas + m_star + m_dm}"
        Mb_within_r = m_gas + m_star
        mb_mvir_within_r = Mb_within_r / m_vir_within_r

        return m_vir_within_r, m_star, m_gas, mb_mvir_within_r

    def calculate(self, halo, existing_properties):
        reff = existing_properties['reff']
        Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff = self.mass_properties_within_r(
            halo, reff)
        return Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff



class BaryonicFractionVirial(PynbodyPropertyCalculation):

    names = ['Mvir', 'Mstar', 'Mgas', 'Mb_mvir']

    def calculate(self, halo, existing_properties):
        m_gas = halo.gas['mass'].sum().in_units('Msol').view(np.ndarray)
        m_star = halo.star['mass'].sum().in_units('Msol').view(np.ndarray)
        m_dm = halo.dm['mass'].sum().in_units('Msol').view(np.ndarray)
        m_vir = halo['mass'].sum().in_units('Msol').view(np.ndarray)
        try:
            Mb = m_gas + m_star
            mb_mvir = Mb / m_vir
        except ZeroDivisionError:
            mb_mvir = np.nan
        

        return m_vir, m_star, m_gas, mb_mvir




class DMDensityProfile(PynbodyPropertyCalculation):
    names = ['rho_dm', 'rho_dm_rbins']

    def calculate(self, halo, existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        prof = pynbody.analysis.profile.Profile(halo.dm, type='log', min=0.1, ndim=3)
        rbins= prof['rbins']
        rho_dm = prof['density']
        return rho_dm, rbins



