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
        number densities at large q
        'unity': sq data approaches the unit matrix at large q.
    """
    def __init__ (self, q, sq, density, matrix_format='full', normalization='densities'):
        super().__init__()
        self._q_data = q
        self.densities = density
        # conversion of _sq_data into full-matrix form
        # is performed here and not upon reinit():
        # we save/restore always the full-matrix form
        qshape = q.shape[0]
        shape = sq.shape
        if matrix_format == 'full':
            d = shape[-1]
            assert (shape == (qshape,d,d))
            assert (density.shape == (d,))
            self._sq_data = np.zeros((qshape,d,d))
            for i in range(d):
                for j in range(d):
                    if normalization == 'densities':
                        xij = np.sqrt(self.densities[i]*self.densities[j])/np.sum(self.densities)
                    else:
                        xij = 1.0
                    self._sq_data[:,i,j] = sq[:,i,j]/xij
        else:
            # FIXME needs to be tested
            d = int(np.sqrt(2*shape[-1]+0.25)-0.5)
            assert (shape == (qshape,d*(d+1)//2))
            assert (density.shape == (d,))
            self._sq_data = np.zeros((qshape,d,d))
            if matrix_format == 'lower_triangular':
                k = 0
                for i in range(d):
                    for j in range(i+1):
                        if normalization == 'densities':
                            xij = np.sqrt(self.densities[i]*self.densities[j])/np.sum(self.densities)
                        else:
                            xij = 1.0
                        self._sq_data[:,i,j] = sq[:,k]/xij
                        if not j==i:
                            self._sq_data[:,j,i] = sq[:,k]/xij
                        k += 1
            else:
                k = 0
                for i in range(d):
                    for j in range(i,d):
                        if normalization == 'densities':
                            xij = np.sqrt(self.densities[i]*self.densities[j])/np.sum(self.densities)
                        else:
                            xij = 1.0
                        self._sq_data[:,i,j] = sq[:,k]/xij
                        if not j==i:
                            self._sq_data[:,j,i] = sq[:,k]/xij
                        k += 1
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

