import scipy
import numpy as np

from ..__h5__ import LoadableModel

class dataSq (LoadableModel, role='structurefactor',
              fields=['_q_data','_sq_data'],attrs=['rho']):
    """Structure factor interpolated from data.

    Parameters
    ----------
    q : array_like
        Grid points where the data points are defined. Needs to be
        in strictly increasing order.
    sq : array_like
        S(q) values corresponding to the q grid points.
        Must have dimension matching q.
    density : float, default=1.0
        Number density to which the structure factor corresponds.
    """
    def __init__ (self, q, sq, density=1.0):
        super().__init__()
        self.rho = density
        self._q_data, self._sq_data = q, sq
        self.reinit()
    def reinit (self):
        self._interpolator = scipy.interpolate.CubicSpline(self._q_data, self._sq_data)
    def density (self):
        return self.rho
    def interpolate_sq (self, q, interpolator, minval=None, maxval=1.0):
        #return scipy.interpolate.interpn(self._q_data, self._sq_data, q)
        if isinstance(q,np.ndarray):
            sqint = interpolator(q)
            sqint[q > np.max(self._q_data)] = maxval
            sqint[q < np.min(self._q_data)] = minval if minval is not None else sqint[0]
        else:
            if q > np.max(self._q_data):
                return 1.0
            if q < np.min(self._q_data):
                return interpolator(np.min(self._q_data))
            sqint = interpolator(q)
        return sqint
    def _cq_from_sq (self, sq):
        return ((1.-1./sq)/self.density())
    def cq(self, q):
        return self._cq_from_sq (self.interpolate_sq(q, self._interpolator))
    def Sq(self, q):
        sq_ = self.interpolate_sq(q, self._interpolator)
        return sq_, self._cq_from_sq (sq_)
    def dcq_dq (self, q):
        sq_ = self.interpolate_sq(q, self._interpolator)
        sqder_ = self.interpolate_sq(q, self._interpolator.derivative(), minval=0.0, maxval=0.0)
        return 1./self.density()/sq_**2 * sqder_

class dataMixtureSq (LoadableModel, role='structurefactor',
                     fields=['_q_data','_sq_data','densities']):
    """Mixture structure factor interpolated from data.

    Parameters
    ----------
    q : array_like
        Grid points where the data points are defined. Needs to be
        in strictly increasing order.
    sq : array_like
        Structure factor data on the q grid. If q has shape (M,)
        sq should have shape (M,S,S) or (M,S(S+1)/2) for a mixture of S
        species, depending on the data format (see below).
    density : array_like
        Partial number densities, should have shape (S,)
    matrix_format : string
        'full': sq should have shape (M,S,S) and gives the full matrix
        for each q.
        'upper_triangular' : sq should have shape (M,S(S+1)/2) and gives
        the upper triangular matrix for each q.
        'lower_triangular' : sq should have shape (M,S(S+1)/2) and gives
        the lower triangular matrix for each q.
        The difference between the last two is in the ordering of indices:
        for example, in a 3x3 matrix

            ================        ================
            upper_triangular        lower_triangular
            0 1 2                   0 _ _
            _ 3 4                   1 2 _
            _ _ 5                   3 4 5
            ================        ================

        (For 2x2 matrices it does not matter.)
    normalization : string

        'densities': sq data approaches the unit matrix set by the
            number densities at large q.
        'unity': sq data approaches the unit matrix at large q.

        The assumption here is that the matrix elements represent
        the correlation of species-density fluctuations; these
        structure factors are known as the Ashcroft-Langreth
        structure factors.

        The internal data format is always corresponding to 'unity'.

        Additionally, the following formats are understood and will
        be converted to the internal representation:

        'FaberZinman':  Faber-Zinman structure factors.
        'BathiaThornton': Bathia-Thornton structure factors for
          binary mixtures (density and charge fluctuations).
    """
    def __init__ (self, q, sq, density, matrix_format='full',
                  normalization='densities'):
        super().__init__()
        self._q_data = q
        self.densities = density
        # conversion of _sq_data into full-matrix form
        # is performed here and not upon reinit():
        # we save/restore always the full-matrix form
        qshape = q.shape[0]
        shape = sq.shape
        d = shape[-1]
        if matrix_format == 'full':
            assert shape == (qshape,d,d)
            assert density.shape == (d,)
            irange = range(d)
            jrange = lambda i : range(d)
            ij = lambda i,j,k : (i,j)
            symmetry = False
        elif matrix_format == 'lower_triangular':
            D = int((np.sqrt(8*shape[-1]+1)-1)/2)
            assert shape == (qshape,D*(D+1)/2)
            assert density.shape == (d,)
            irange = range(d)
            jrange = lambda i : range(i+1)
            ij = lambda i,j,k : (k,)
            symmetry = True
        elif matrix_format == 'upper_triangular':
            D = int((np.sqrt(8*shape[-1]+1)-1)/2)
            assert shape == (qshape,D*(D+1)/2)
            assert density.shape == (d,)
            irange = range(d)
            jrange = lambda i : range(i,d)
            ij = lambda i,j,k : (k,)
            symmetry = True
        else:
            raise f"unknown matrix format {matrix_format}"
        self._sq_data = np.zeros((qshape,d,d))
        if normalization == 'densities':
            xij = np.outer(np.sqrt(self.densities))/np.sum(self.densities)
        else:
            xij = 1.0
        k = 0
        for i in irange:
            for j in jrange(i):
                self._sq_data[:,i,j] = sq[:,*ij(i,j)] / xij[i,j]
                if symmetry and not j==i:
                    self._sq_data[:,j,i] = sq[:,*ij(i,j)] / xij[i,j]
                k += 1
        if normalization == 'FaberZinman':
            # FIXME untested
            xij = np.outer(np.sqrt(self.densities))/np.sum(self.densities)
            for i in range(d):
                for j in range(d):
                    self._sq_data[:,i,j] = xij[i,j] * (self._sq_data[:,i,j]-1)
                self._sq_data[:,i,i] += 1
        elif normalization == 'Bathia-Thornton':
            # FIXME untested
            x = self.densities/np.sum(self.densities)
            assert d==2 # currently only implemented for binary mixtures
            xA, xB = x[0], x[1]
            SNN = self._sq_data[:,0,0].copy()
            SCC = self._sq_data[:,1,1].copy()
            SNC = self._sq_data[:,0,1].copy()
            self._sq_data[:,0,0] = xA * SNN + 2*SNC + SCC/np.sqrt(xA*xB)
            self._sq_data[:,1,1] = xB * SNN - 2*SNC + SCC/np.sqrt(xA*xB)
            self._sq_data[:,0,1] = np.sqrt(xA*xB) * SNN \
                    + (xB-xA)/np.sqrt(xA*xB) * SNC - SCC/np.sqrt(xA*xB)
            self._sq_data[:,1,0] = self._sq_data[:,0,1]
        self.reinit()
    def reinit (self):
        self._interpolator = scipy.interpolate.CubicSpline(self._q_data, self._sq_data)
    def density (self):
        return np.sum(self.densities)
    def partial_density (self, a):
        return self.densities[a]
    def interpolate_sq (self, q, interpolator, minval=None, maxval=1.0):
        #return scipy.interpolate.interpn(self._q_data, self._sq_data, q)
        if isinstance(q,np.ndarray):
            sqint = interpolator(q)
            sqint[q > np.max(self._q_data)] = maxval*np.diag(np.ones_like(self.densities))
            sqint[q < np.min(self._q_data)] = minval if minval is not None else sqint[0]
        else:
            if q > np.max(self._q_data):
                return 1.0
            if q < np.min(self._q_data):
                return interpolator(np.min(self._q_data))
            sqint = interpolator(q)
        return sqint
    def _cq_from_sq(self, sq):
        rhoa, rhob = np.meshgrid(self.densities, self.densities)
        # FIXME: if q is not ndarray?
        return (np.diag(np.ones_like(self.densities)) - np.linalg.inv(sq))/np.sqrt(rhoa*rhob)
        # this would be the form to use if the internal sq would be
        # normalized to partial-density matrix:
        #return np.diag(1./self.densities) - np.linalg.inv(np.sqrt(rhoa*rhob)[None,:,:]*sq)
    def cq(self, q):
        return self._cq_from_sq (self.interpolate_sq(q, self._interpolator))
    def Sq(self, q):
        sq_ = self.interpolate_sq(q, self._interpolator)
        return sq_, self._cq_from_sq(sq_)
    def dcq_dq (self,q):
        sqinv_ = np.linalg.inv(self.interpolate_sq(q, self._interpolator))
        sqder_ = self.interpolate_sq(q, self._interpolator.derivative(), minval=0.0, maxval=0.0)
        # FIXME
        rhoa, rhob = np.meshgrid(self.densities, self.densities)
        return 1./np.sqrt(rhoa*rhob) * np.einsum("...ij,...jk,...kl->...il",sqinv_,sqder_,sqinv_)


class interpolatedSq (LoadableModel, role='structurefactor',
                      fields=['gammalist','sqlist'],attrs=['gamma']):
    """Interpolation between structure factors.

    Performs linear interpolation between the two nearest given static
    structure factors; intended for the case where a set of S(q) data
    has been obtained from simulation or experiment and read into objects
    of the type `mctpy.structurefactors.dataSq` or
    `mctpy.structurefactors.dataMixtureSq`, but should work with any
    structure factor object, even interpolating between different
    potentials or closures.

    Parameters
    ----------
    gammalist : array_list
        A list of real-valued parameters to which the given structure
        factors are associated. The interpolation will then work by calling
        `set_gamma` with a value somewhere in between these points.
        Needs to be in strictly increasing order.
    sqlist : list
        List of corresponding structure-factor objects. They should match
        in the shape of `sq` and `cq` arrays they return, but otherwise
        there is no restriction.
    """
    def __init__ (self, gammalist, sqlist):
        self.gammalist = np.array(gammalist)
        self.sqlist = sqlist
        assert len(sqlist) == len(gammalist)
        self.set_gamma(self.gammalist[0])
    def reinit (self):
        self.set_gamma(self.gamma)
    def set_gamma (self, gamma):
        """Set the interpolation point.

        After setting a `gamma` value, the usual structure-factor methods
        like `cq`, `Sq`, and `dcq_dq` will return interpolated values.
        If not called, the interpolation returns the values corresponding
        to the first structure factor given."""
        gi0 = np.sum(self.gammalist <= gamma) - 1
        M = len(self.gammalist)
        if gi0 < 0:
            gi0 = 0
        gi1 = M - np.sum(self.gammalist > gamma)
        if gi1 >= M:
            gi1 = M - 1
        self.gamma = gamma
        self.gi0 = gi0
        self.gi1 = gi1
        self.g0 = self.gammalist[gi0]
        self.g1 = self.gammalist[gi1]
        return self
    def cq (self, q):
        _cq0 = self.sqlist[self.gi0].cq(q)
        _cq1 = self.sqlist[self.gi1].cq(q)
        return _cq0 + (self.gamma-self.g0)/(self.g1-self.g0) * (_cq1-_cq0)
    def Sq (self, q):
        _sq0, _cq0 = self.sqlist[self.gi0].Sq(q)
        _sq1, _cq1 = self.sqlist[self.gi1].Sq(q)
        return _sq0 + (self.gamma-self.g0)/(self.g1-self.g0) * (_sq1-_sq0), \
               _cq0 + (self.gamma-self.g0)/(self.g1-self.g0) * (_cq1-_cq0)
    def dcq_dq (self, q):
        _dcq0 = self.sqlist[self.gi0].dcq_dq(q)
        _dcq1 = self.sqlist[self.gi1].dcq_dq(q)
        return _dcq0 + (self.gamma-self.g0)/(self.g1-self.g0) * (_dcq1-_dcq0)

