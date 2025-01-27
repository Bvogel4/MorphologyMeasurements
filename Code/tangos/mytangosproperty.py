from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties import LivePropertyCalculation
from tangos.properties.pynbody.centring import centred_calculation
from tangos.properties import LivePropertyCalculationInheritingMetaProperties
import pynbody
import numpy as np
import pymp
import scipy
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from pynbody import array
import traceback
import logging


logger = logging.getLogger('pynbody.analysis.halo')

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
    threads = 40
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

def get_bins(n):
    if n < 1e4:
        return int(20 * n/1e4 /2 + 10)
    elif n >= 1e4:
        return int( (np.log10(n) - 3) ** 2 * 20)








class r_80(PynbodyPropertyCalculation):
    # get the radius that contains 80% of the mass of the stars
    names = ['r_80']
    def __init__(self, simulation):
        super().__init__(simulation)

    def calculate(self, halo, existing_properties):
        N_star = len(halo.s)
        # get radius that contains 80% of star particles
        rsort = halo.s['r'][np.argsort(halo.s['r'])]
        r_8 = rsort[int(0.8 * N_star)]
        return r_8






class Shape_Profile(PynbodyPropertyCalculation):
    """
    Calculate the axis ratio of the stellar and dark matter components of a halo, only for where stars and dark matter both exist.
    """
    names = ['ba_s', 'ca_s', 'rbins_s', 'ba_d', 'ca_d', 'rbins_d']

    def __init__(self, simulation):
        super().__init__(simulation)

    @staticmethod
    def process_shape(particles, rin, rout, bins):
        rbins, axis_lengths, num_particles, rotations = shape(particles,
                                                              nbins=bins,
                                                              ndim=3, rmin=rin,
                                                              rmax=rout,
                                                              max_iterations=125,
                                                              tol=1e-2,
                                                              justify=False)
        ba = axis_lengths[:, 1] / axis_lengths[:, 0]
        ca = axis_lengths[:, 2] / axis_lengths[:, 0]
        shape_dict = {'ba': ba, 'ca': ca, 'rbins': rbins}
        return shape_dict

    def _get_shape(self, halo):
        nan_array= np.array([np.nan]*100)
        nan_dict  = {'ba': nan_array, 'ca': nan_array, 'rbins': nan_array}
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        rin = 0.1
        # Find the stellar shape of the halo (original run)
        try:
            # Find the number of particles in the halo
            N_star = len(halo.s)
            if N_star == 0:
                rout = halo.dm['r'].max() * 0.3
                starshape = nan_dict
            else:
                # Find the number of bins to use
                rsort = halo.s['r'][np.argsort(halo.s['r'])]
                r_8 = rsort[int(0.8 * N_star)]
                rout = rsort[-1]
                bins = get_bins(N_star)
                starshape = self.process_shape(halo.s, rin, rout, bins)
                r_star_max = starshape['rbins'][-3]
                rout = r_star_max
        except Exception as e:
            #print(f'Error in halo {halo}: {e}')
            #raise error
            raise e
            traceback.print_exc()
            starshape = nan_dict
        try:
            N_dm = len(halo.dm)
            bins = get_bins(N_dm)
            if rout == None:
                rout = halo.dm['r'].max() * 0.3
            darkshape = self.process_shape(halo.dm, rin, rout, bins)
            # probably not resolved
            # try:
            #
            #     rout_2 = darkshape['rbins'][3]
            #     N_dark_1kpc = len(halo.d['r'][halo.d['r'] < rout_2])
            #
            #     # Second run for dark matter shape (0.1 to 1 kpc, 500 particles per bin)
            #     bins_2 = N_dark_1kpc / 500
            #     if bins_2 >= 1:
            #         darkshape_2 = self.process_shape(halo.d, rin, rout_2, int(bins_2))
            #
            #         # combine the two dark matter shape profiles
            #         # Convert to numpy arrays, concatenate, then convert back to pynbody array
            #         units = darkshape['rbins'].units  # Store original units
            #         rbins = pynbody.array.SimArray(
            #             np.concatenate((
            #                 darkshape['rbins'].view(np.ndarray),
            #                 darkshape_2['rbins'].view(np.ndarray)
            #             ))
            #         )
            #         rbins.units = units  # Restore the original units
            #         ba = np.concatenate((darkshape['ba'], darkshape_2['ba']))
            #         ca = np.concatenate((darkshape['ca'], darkshape_2['ca']))
            #         #sort the arrays by rbins
            #         sort_index = np.argsort(rbins)
            #         rbins = rbins[sort_index]
            #         ba = ba[sort_index]
            #         ca = ca[sort_index]
            #         darkshape = {'ba': ba, 'ca': ca, 'rbins': rbins}
            # except Exception as e:
            #     print(f'Error in low radius dark matter shape: {e}')
            #     print(traceback.format_exc())
            #     print(darkshape_2)
            #     raise e
                
        except Exception as e:
            #print(f'Error in halo {halo}: {e}')
            traceback.print_exc()
            darkshape = nan_dict
            raise e
        return starshape, darkshape

    def calculate(self, halo, existing_properties):
        starshape,darkshape = self._get_shape(halo)
        ba_s, ca_s, rbins_s = starshape['ba'], starshape['ca'], starshape['rbins']
        ba_d, ca_d, rbins_d = darkshape['ba'], darkshape['ca'], darkshape['rbins']
        return ba_s, ca_s, rbins_s, ba_d, ca_d, rbins_d


    def plot_xlog(self):
        return False

    def plot_ylog(self):
        return False

class SmoothAxisRatio(LivePropertyCalculation):
    names = ['ba_s_smoothed', 'ba_d_smoothed', 'ca_s_smoothed', 'ca_d_smoothed',
             'ba_s_at_reff', 'ba_d_at_reff', 'ca_s_at_reff', 'ca_d_at_reff']

    def requires_property(self):
        return ['ba_s', 'ba_d', 'ca_s', 'ca_d', 'rbins_s','rbins_d', 'reff']

    @staticmethod
    def nan_func(x):
        return np.nan
    @staticmethod
    def smooth_shape(rbins, ba, ca):
        k = 4
        s_factor = 1
        """
        Smooth and filter data, handling a few NaN values gracefully.

        Parameters:
        rbins, ba, ca: array-like, input data
        k: int, degree of the smoothing spline (default 3)
        s_factor: float, smoothing factor as a fraction of len(rbins) (default 0.01)
        residual_threshold, jump_threshold, jump_percentage: unused in this version

        Returns:
        rbins, ba, ca: filtered arrays
        ba_s, ca_s: smoothed spline functions
        """
        import numpy as np
        from scipy.interpolate import UnivariateSpline

        # Remove rows where either ba or ca is NaN
        mask = ~np.isnan(ba) & ~np.isnan(ca)
        rbins_filtered = rbins[mask]
        ba_filtered = ba[mask]
        ca_filtered = ca[mask]

        #if there are no values, return the original values
        if len(rbins_filtered) == 0:
            return rbins, ba, ca, SmoothAxisRatio.nan_func, SmoothAxisRatio.nan_func

        # Calculate smoothing parameter
        s = s_factor * len(rbins_filtered)

        # Create splines
        #print(rbins_filtered, ba_filtered)
        ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
        ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)

        # Print some diagnostic information
        # print(f"Total data points: {len(rbins)}")
        # print(f"Data points after NaN removal: {len(rbins_filtered)}")
        # print(f"NaN percentage: {(1 - len(rbins_filtered)/len(rbins))*100:.2f}%")

        n = len(rbins_filtered)
        # calculate residuals and remove outliers
        ba_residuals = ba_filtered - ba_s(rbins_filtered)
        ca_residuals = ca_filtered - ca_s(rbins_filtered)
        # calculate the standard deviation of the residuals
        ba_std = np.std(ba_residuals)
        ca_std = np.std(ca_residuals)
        # remove outliers
        d = 3

        mask = np.abs(ba_residuals) < d * ba_std

        rbins_filtered = rbins_filtered[mask]
        ba_filtered = ba_filtered[mask]
        ca_filtered = ca_filtered[mask]
        mask = np.abs(ca_residuals[mask]) < d * ca_std
        rbins_filtered = rbins_filtered[mask]
        ba_filtered = ba_filtered[mask]
        ca_filtered = ca_filtered[mask]
        # Recreate splines
        ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
        ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)

        # remove any points that are isolated in space
        # calculate the difference between each point

        diff = np.diff(rbins_filtered, prepend=0)
        # print(diff)
        # mask isolated points
        mask = diff > 1
        # print(mask)
        # print(rbins_filtered[mask])
        # print(diff[mask])
        rbins_filtered = rbins_filtered[~mask]
        ba_filtered = ba_filtered[~mask]
        ca_filtered = ca_filtered[~mask]
        # Recreate splines
        ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
        ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)
        # Print some diagnostic information
        # print(f"Data points after outlier removal: {len(rbins_filtered)}")
        # print(f"Outlier percentage: {(1 - len(rbins_filtered)/len(rbins))*100:.2f}%")

        # def clip_function(func):
        #     def clipped(x):
        #         return np.clip(func(x), 0, 1)
        #
        #     return clipped
        #
        # # clip the function to 0,1
        # ba_s_c = clip_function(ba_s)
        # ca_s_c = clip_function(ca_s)

        return rbins_filtered, ba_filtered, ca_filtered, ba_s, ca_s

    def calculate(self, halo, existing_properties):
        rbins_s = existing_properties['rbins_s']
        rbins_d = existing_properties['rbins_d']
        #print(rbins_s,rbins_d)
        rbins_s, ba_s, ca_s, ba_s_spline, ca_s_spline = self.smooth_shape(rbins_s, existing_properties['ba_s'],
                                                                            existing_properties['ca_s'])
        rbins_d, ba_d, ca_d, ba_d_spline, ca_d_spline = self.smooth_shape(rbins_d, existing_properties['ba_d'],
                                                                            existing_properties['ca_d'])
        reff = existing_properties['reff']
        ba_s_at_reff = ba_s_spline(reff)
        ca_s_at_reff = ca_s_spline(reff)
        ba_d_at_reff = ba_d_spline(reff)
        ca_d_at_reff = ca_d_spline(reff)
        return ba_s_spline, ca_s_spline, ba_d_spline, ca_d_spline, ba_s_at_reff, ca_s_at_reff, ba_d_at_reff, ca_d_at_reff
    

class SersicFit(PynbodyPropertyCalculation):
    names = ['reff', 'rhalf']

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
            prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25,
                                                    max=5 * Rhalf, ndim=2,
                                                    nbins=int(
                                                        (5 * Rhalf) / 0.1))
            vband = prof['sb,V']
            smooth = np.nanmean(
                np.pad(vband.astype(float), (0, 3 - vband.size % 3),
                       mode='constant', constant_values=np.nan).reshape(
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
            par, ign = curve_fit(self.sersic, x, y, p0=(m0, r0, 1),
                                 bounds=([10, 0, 0.5], [40, 100, 16.5]))
            reff = pynbody.array.SimArray(par[1], 'kpc')
        except:
            print("Sersic fit failed")
            print(traceback.format_exc())
            # set reff to value of later halo
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
        prof = pynbody.analysis.profile.Profile(halo, bins=rbins, ndim=2)
        mass_enc = prof['mass_enc']
        dyntime = (rbins ** 3 / (2 * pynbody.units.G * mass_enc)) ** (1 / 2)
        return dyntime


class BaryonicFractionReff(PynbodyPropertyCalculation):
    names = ['Mvir_within_reff', 'Mstar_within_reff', 'Mgas_within_reff',
             'Mb_mvir_within_reff']

    def requires_property(self):
        return ['reff', 'max_radius']

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
        prof = pynbody.analysis.profile.Profile(halo.dm, type='log', min=0.1,
                                                ndim=3)
        rbins = prof['rbins']
        rho_dm = prof['density']
        return rho_dm, rbins



