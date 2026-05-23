import scipy
import numpy as np

class dataSq:
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
        self.rho = density
        self._q_data, self._sq_data = q, sq
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

class dataMixtureSq:
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
            upper_triangular        lower_triangular
            0 1 2                   0
              3 4                   1 2
                5                   3 4 5
        (For 2x2 matrices it does not matter.)
    normalization : string
        'densities': sq data approaches the unit matrix set by the
        number densities at large q
        'unity': sq data approaches the unit matrix at large q.
    """
    def __init__ (self, q, sq, density, matrix_format='full', normalization='densities'):
        self._q_data = q
        qshape = q.shape[0]
        shape = sq.shape
        self.densities = density
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
        return np.diag(1./self.densities) - np.linalg.inv(np.sqrt(rhoa*rhob)[None,:,:]*sq)
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

