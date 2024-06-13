from pynbody import array
from pynbody.analysis import profile
import numpy as np
import logging
logger = logging.getLogger('pynbody.analysis.halo')
from numpy import matmul as X
from numpy import sin,cos
from skimage.measure import EllipseModel
from math import pi

#Imagae coordinate conversion
def pix2kpc(pix,width):
    return(pix/1000.*width)
def kpc2pix(kpc,width):
    return(kpc/width*1000)

#Projection Functions
def Rx(theta):
    #X-axis rotation matrix
    a = np.radians(theta)
    return( [[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]] )

def Ry(theta):
    #Y-axis rotation matrix
    a = np.radians(theta)
    return( [[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]] )

def Ellipsoid(a,ba,ca,Es,xrot,yrot):
    #Generate 3D ellispoid from halo_shape_stellar outputs (a,ba,ca,Es)
    #and rotate it along x-axis and y-axis (xrot,yrot)
    resolution = 50
    phi = np.linspace(0,2*np.pi,resolution)
    theta = np.linspace(0,np.pi,resolution)
    #Get x,y,z of original ellispoid
    x,y,z=[[],[],[]]
    for i in np.arange(resolution):
        for j in np.arange(resolution):
            x.append(a*sin(theta[i])*cos(phi[j]))
            y.append(a*ba*sin(theta[i])*sin(phi[j]))
            z.append(a*ca*cos(theta[i]))
    x,y,z = np.array(x),np.array(y),np.array(z)
    #Rotate original ellipsoid by Es
    rx,ry,rz = np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))
    for j in np.arange(len(x)):
        rv = X(Es,np.array([[x[j]],[y[j]],[z[j]]]))
        rx[j] = rv[0][0]
        ry[j] = rv[1][0]
        rz[j] = rv[2][0]
    #Rotate ellipsoid around x-axis then y-axis
    for i in np.arange(len(rx)):
        vec = [rx[i],ry[i],rz[i]]
        rx[i],ry[i],rz[i] = X(Ry(yrot),X(Rx(xrot),vec))
    return(rx,ry,rz)

def Project(x,y,z):
    edge_x,edge_y = x[(z>-.1)&(z<.1)],y[(z>-.1)&(z<.1)]
    xy=np.zeros((len(edge_x),2))
    for i in np.arange(len(edge_x)):
        xy[i] = [edge_x[i],edge_y[i]]
    E = EllipseModel()
    E.estimate(np.array(xy))
    params = E.params
    cen = np.array([params[0],params[1]])
    phi = params[4]
    a,b = params[2],params[3]#np.max([params[2],params[3]]),np.min([params[2],params[3]])
    return(a,b,cen,phi)

def cart2pol(x, y):
    #Cartesian to Polar coordinate conversion
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
def pol2cart(rho, phi):
    #Polar to Cartesian coordinate conversion
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def Project_OLD(x,y,z):
    #Return the b/a axis ratios of a flattened 3D rotated ellipsoid
    phi_bins = np.linspace(-pi,pi,50)
    r,phi = cart2pol(np.array(x),np.array(y))
    r_edge,phi_edge = [[],[]]
    for i in np.arange(len(phi_bins)-1):
        binr,binphi = [[],[]]
        for j in np.arange(len(r)):
            if phi_bins[i]<phi[j]<phi_bins[i+1]:
                binr.append(r[j])
                binphi.append(phi[j])
        if len(binr)>0:
            r_edge.append(max(binr))
            phi_edge.append(binphi[binr.index(max(binr))])
    edge_x,edge_y=pol2cart(r_edge,phi_edge)
    xy=np.zeros((len(edge_x),2))
    for i in np.arange(len(edge_x)):
        xy[i] = [edge_x[i],edge_y[i]]
    E = EllipseModel()
    E.estimate(np.array(xy))
    params = E.params
        
    cen = np.array([params[0], params[1]])
    phi = params[4]
    a, b = params[2], params[3]  # You could also directly assign without checking for max or min as commented


    
    return(a,b,cen,phi)

def myround(x, base=5):
    return base * round(x/base)

def RandomRotation(res=15):
    phi = np.degrees(np.random.uniform(0,np.pi*2))
    y = int(myround(phi,base=res))
    theta = np.degrees(np.arccos(np.random.uniform(-1,1)))
    x = int(myround(theta,base=res))
    if x>165:
        x = x%180
        y = (180-y)
        if y<0:
            y = y+360
    if y == 360:
        y = 0
    return(x,y)


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











#RGI From Scipy 1.9 (conda won't update past 1.7???)
import itertools

import numpy as np

from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.interpolate._cubic import PchipInterpolator
#from scipy.interpolate._rgi_cython import evaluate_linear_2d, find_indices
from scipy.interpolate._bsplines import make_interp_spline
#from scipy.interpolate._fitpack2 import RectBivariateSpline


def _check_points(points):
    descending_dimensions = []
    grid = []
    for i, p in enumerate(points):
        # early make points float
        # see https://github.com/scipy/scipy/pull/17230
        p = np.asarray(p, dtype=float)
        if not np.all(p[1:] > p[:-1]):
            if np.all(p[1:] < p[:-1]):
                # input is descending, so make it ascending
                descending_dimensions.append(i)
                p = np.flip(p)
            else:
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending or descending" % i)
        # see https://github.com/scipy/scipy/issues/17716
        p = np.ascontiguousarray(p)
        grid.append(p)
    return tuple(grid), tuple(descending_dimensions)


def _check_dimensionality(points, values):
    if len(points) > values.ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), values.ndim))
    for i, p in enumerate(points):
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))


class RegularGridInterpolator:
    """
    Interpolation on a regular or rectilinear grid in arbitrary dimensions.

    The data must be defined on a rectilinear grid; that is, a rectangular
    grid with even or uneven spacing. Linear, nearest-neighbor, spline
    interpolations are supported. After setting up the interpolator object,
    the interpolation method may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip". This
        parameter will become the default for the object's ``__call__``
        method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True.

    fill_value : float or None, optional
        The value to use for points outside of the interpolation domain.
        If None, values outside the domain are extrapolated.
        Default is ``np.nan``.

    Methods
    -------
    __call__

    Attributes
    ----------
    grid : tuple of ndarrays
        The points defining the regular grid in n dimensions.
        This tuple defines the full grid via
        ``np.meshgrid(*grid, indexing='ij')``
    values : ndarray
        Data values at the grid.
    method : str
        Interpolation method.
    fill_value : float or ``None``
        Use this value for out-of-bounds arguments to `__call__`.
    bounds_error : bool
        If ``True``, out-of-bounds argument raise a ``ValueError``.

    Notes
    -----
    Contrary to `LinearNDInterpolator` and `NearestNDInterpolator`, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    In other words, this class assumes that the data is defined on a
    *rectilinear* grid.

    .. versionadded:: 0.14

    The 'slinear'(k=1), 'cubic'(k=3), and 'quintic'(k=5) methods are
    tensor-product spline interpolators, where `k` is the spline degree,
    If any dimension has fewer points than `k` + 1, an error will be raised.

    .. versionadded:: 1.9

    If the input data is such that dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    Examples
    --------
    **Evaluate a function on the points of a 3-D grid**

    As a first example, we evaluate a simple example function on the points of
    a 3-D grid:

    >>> from scipy.interpolate import RegularGridInterpolator
    >>> import numpy as np
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = np.linspace(1, 4, 11)
    >>> y = np.linspace(4, 7, 22)
    >>> z = np.linspace(7, 9, 33)
    >>> xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    >>> data = f(xg, yg, zg)

    ``data`` is now a 3-D array with ``data[i, j, k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> interp = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = np.array([[2.1, 6.2, 8.3],
    ...                 [3.3, 5.2, 7.1]])
    >>> interp(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to

    >>> f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)
    (125.54200000000002, 145.894)

    **Interpolate and extrapolate a 2D dataset**

    As a second example, we interpolate and extrapolate a 2D data set:

    >>> x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
    >>> def ff(x, y):
    ...     return x**2 + y**2

    >>> xg, yg = np.meshgrid(x, y, indexing='ij')
    >>> data = ff(xg, yg)
    >>> interp = RegularGridInterpolator((x, y), data,
    ...                                  bounds_error=False, fill_value=None)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
    ...            s=60, c='k', label='data')

    Evaluate and plot the interpolator on a finer grid

    >>> xx = np.linspace(-4, 9, 31)
    >>> yy = np.linspace(-4, 9, 31)
    >>> X, Y = np.meshgrid(xx, yy, indexing='ij')

    >>> # interpolator
    >>> ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
    ...                   alpha=0.4, color='m', label='linear interp')

    >>> # ground truth
    >>> ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
    ...                   alpha=0.4, label='ground truth')
    >>> plt.legend()
    >>> plt.show()

    Other examples are given
    :ref:`in the tutorial <tutorial-interpolate_regular_grid_interpolator>`.

    See Also
    --------
    NearestNDInterpolator : Nearest neighbor interpolation on *unstructured*
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on *unstructured* data
                           in N dimensions

    interpn : a convenience function which wraps `RegularGridInterpolator`

    scipy.ndimage.map_coordinates : interpolation on grids with equal spacing
                                    (suitable for e.g., N-D image resampling)

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
           :doi:`10.1090/S0025-5718-1988-0917826-0`

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}
    _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
    _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan):
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)
        elif method in self._SPLINE_METHODS:
            self._validate_grid_dimensions(points, method)
        self.method = method
        self.bounds_error = bounds_error
        self.grid, self._descending_dimensions = _check_points(points)
        self.values = self._check_values(values)
        self._check_dimensionality(self.grid, self.values)
        self.fill_value = self._check_fill_value(self.values, fill_value)
        if self._descending_dimensions:
            self.values = np.flip(values, axis=self._descending_dimensions)

    def _check_dimensionality(self, grid, values):
        _check_dimensionality(grid, values)

    def _check_points(self, points):
        return _check_points(points)

    def _check_values(self, values):
        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        return values

    def _check_fill_value(self, values, fill_value):
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
        return fill_value

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to evaluate the interpolator at.

        method : str, optional
            The method of interpolation to perform. Supported are "linear",
            "nearest", "slinear", "cubic", "quintic" and "pchip". Default is
            the method chosen when the interpolator was created.

        Returns
        -------
        values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at `xi`. See notes for behaviour when
            ``xi.ndim == 1``.

        Notes
        -----
        In the case that ``xi.ndim == 1`` a new axis is inserted into
        the 0 position of the returned array, values_x, so its shape is
        instead ``(1,) + values.shape[ndim:]``.

        Examples
        --------
        Here we define a nearest-neighbor interpolator of a simple function

        >>> import numpy as np
        >>> x, y = np.array([0, 1, 2]), np.array([1, 3, 7])
        >>> def f(x, y):
        ...     return x**2 + y**2
        >>> data = f(*np.meshgrid(x, y, indexing='ij', sparse=True))
        >>> from scipy.interpolate import RegularGridInterpolator
        >>> interp = RegularGridInterpolator((x, y), data, method='nearest')

        By construction, the interpolator uses the nearest-neighbor
        interpolation

        >>> interp([[1.5, 1.3], [0.3, 4.5]])
        array([2., 9.])

        We can however evaluate the linear interpolant by overriding the
        `method` parameter

        >>> interp([[1.5, 1.3], [0.3, 4.5]], method='linear')
        array([ 4.7, 24.3])
        """
        is_method_changed = self.method != method
        method = self.method if method is None else method
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)

        xi, xi_shape, ndim, nans, out_of_bounds = self._prepare_xi(xi)

        if method == "linear":
            indices, norm_distances = self._find_indices(xi.T)
            if (ndim == 2 and hasattr(self.values, 'dtype') and
                    self.values.ndim == 2 and self.values.flags.writeable and
                    self.values.dtype in (np.float64, np.complex128) and
                    self.values.dtype.byteorder == '='):
                # until cython supports const fused types, the fast path
                # cannot support non-writeable values
                # a fast path
                out = np.empty(indices.shape[1], dtype=self.values.dtype)
                # result = evaluate_linear_2d(self.values,
                #                             indices,
                #                             norm_distances,
                #                             self.grid,
                #                             out)
            else:
                result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            indices, norm_distances = self._find_indices(xi.T)
            result = self._evaluate_nearest(indices, norm_distances)
        elif method in self._SPLINE_METHODS:
            if is_method_changed:
                self._validate_grid_dimensions(self.grid, method)
            result = self._evaluate_spline(xi, method)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        # f(nan) = nan, if any
        if np.any(nans):
            result[nans] = np.nan
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _prepare_xi(self, xi):
        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             f"{xi.shape[-1]} but this "
                             f"RegularGridInterpolator has dimension {ndim}")

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = np.asarray(xi, dtype=float)

        # find nans in input
        nans = np.any(np.isnan(xi), axis=-1)

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)
            out_of_bounds = None
        else:
            out_of_bounds = self._find_out_of_bounds(xi.T)

        return xi, xi_shape, ndim, nans, out_of_bounds

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # Compute shifting up front before zipping everything together
        shift_norm_distances = [1 - yi for yi in norm_distances]
        shift_indices = [i + 1 for i in indices]

        # The formula for linear interpolation in 2d takes the form:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
        zipped1 = zip(indices, shift_norm_distances)
        zipped2 = zip(shift_indices, norm_distances)

        # Take all products of zipped1 and zipped2 and iterate over them
        # to get the terms in the above formula. This corresponds to iterating
        # over the vertices of a hypercube.
        hypercube = itertools.product(*zip(zipped1, zipped2))
        value = np.array([0.])
        for h in hypercube:
            edge_indices, weights = zip(*h)
            weight = np.array([1.])
            for w in weights:
                weight = weight * w
            term = np.asarray(self.values[edge_indices]) * weight[vslice]
            value = value + term   # cannot use += because broadcasting
        return value

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [np.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _validate_grid_dimensions(self, points, method):
        k = self._SPLINE_DEGREE_MAP[method]
        for i, point in enumerate(points):
            ndim = len(np.atleast_1d(point))
            if ndim <= k:
                raise ValueError(f"There are {ndim} points in dimension {i},"
                                 f" but method {method} requires at least "
                                 f" {k+1} points per dimension.")

    def _evaluate_spline(self, xi, method):
        # ensure xi is 2D list of points to evaluate (`m` is the number of
        # points and `n` is the number of interpolation dimensions,
        # ``n == len(self.grid)``.)
        if xi.ndim == 1:
            xi = xi.reshape((1, xi.size))
        m, n = xi.shape

        # Reorder the axes: n-dimensional process iterates over the
        # interpolation axes from the last axis downwards: E.g. for a 4D grid
        # the order of axes is 3, 2, 1, 0. Each 1D interpolation works along
        # the 0th axis of its argument array (for 1D routine it's its ``y``
        # array). Thus permute the interpolation axes of `values` *and keep
        # trailing dimensions trailing*.
        axes = tuple(range(self.values.ndim))
        axx = axes[:n][::-1] + axes[n:]
        values = self.values.transpose(axx)

        if method == 'pchip':
            _eval_func = self._do_pchip
        else:
            _eval_func = self._do_spline_fit
        k = self._SPLINE_DEGREE_MAP[method]

        # Non-stationary procedure: difficult to vectorize this part entirely
        # into numpy-level operations. Unfortunately this requires explicit
        # looping over each point in xi.

        # can at least vectorize the first pass across all points in the
        # last variable of xi.
        last_dim = n - 1
        first_values = _eval_func(self.grid[last_dim],
                                  values,
                                  xi[:, last_dim],
                                  k)

        # the rest of the dimensions have to be on a per point-in-xi basis
        shape = (m, *self.values.shape[n:])
        result = np.empty(shape, dtype=self.values.dtype)
        for j in range(m):
            # Main process: Apply 1D interpolate in each dimension
            # sequentially, starting with the last dimension.
            # These are then "folded" into the next dimension in-place.
            folded_values = first_values[j, ...]
            for i in range(last_dim-1, -1, -1):
                # Interpolate for each 1D from the last dimensions.
                # This collapses each 1D sequence into a scalar.
                folded_values = _eval_func(self.grid[i],
                                           folded_values,
                                           xi[j, i],
                                           k)
            result[j, ...] = folded_values

        return result

    @staticmethod
    def _do_spline_fit(x, y, pt, k):
        local_interp = make_interp_spline(x, y, k=k, axis=0)
        values = local_interp(pt)
        return values

    @staticmethod
    def _do_pchip(x, y, pt, k):
        local_interp = PchipInterpolator(x, y, axis=0)
        values = local_interp(pt)
        return values

    def _find_indices(self, xi):
        return np.NaN#find_indices(self.grid, xi)

    def _find_out_of_bounds(self, xi):
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        return out_of_bounds