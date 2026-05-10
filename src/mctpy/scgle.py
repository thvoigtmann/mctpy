import numpy as np

from numba import njit
from .__util__ import model_base

# TODO mark dates for initial implementation // (tv) 2016-02-10, 2016-02-16
# TODO also used elsewhere, make it a __util__ function?
def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

class scgle_model (model_base):
    """The simple-liquid model of SCGLE.

    Parameters
    ----------
    Sq : object
        The static structure factor object
    q : array_like
        Wave number grid.
    D0 : float, default = 1.0
        Short-term diffusion coefficient for Brownian dynamics.
    alpha_min : float, default = 1.725
        Fraction of structure-factor maximum to set `k_min` to.
    k_min : float, optional
        Value to set `k_min` to directly, overriding `alpha_min`.

    Notes
    -----
    If only `alpha_min` is set, initialization will search for
    the global maximum of the structure factor on the given q grid,
    and set `k_min` to `alpha_min` times the maximum value obtained
    from a parabolic fit around the three points closest to the
    maximum. If a value is given for `k_min`, this supersedes the
    automatic determination and takes the value directly.
    """
    def __init__ (self, Sq, q, D0=1.0, alpha_min=1.725, k_min=None):
        model_base.__init__(self)
        self.rho = Sq.density()
        self.q = q
        self.Sq = Sq
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        if k_min is not None:
            self.k_min = k_min
        else:
            self.k_min = self._calc_kmin(alpha_min)
        self.__init_vertices__()
        self.D0 = D0
    def __len__ (self):
        return self.M
    def vector_dimension (self):
        return 2
    def Wq (self):
        return np.repeat(self.q*self.q,2)
    def Aq (self):
        return np.transpose([self.sq,np.ones_like(self.sq)]).flatten()/self.vth**2
    def Bq (self):
        return np.transpose([self.sq,np.ones_like(self.sq)]).flatten()/self.D0
    def dq (self):
        return _dq(self.q)
    def __init_vertices__ (self):
        pre = 1./(6*np.pi**2.) * self.rho
        lambda_q = self.lambdak(self.q)
        q, k = np.meshgrid(self.q, self.q, indexing='ij')
        Sq, Sk = np.meshgrid(self.sq, self.sq, indexing='ij')
        cq, ck = np.meshgrid(self.cq, self.cq, indexing='ij')
        lq, lk = np.meshgrid(lambda_q, lambda_q, indexing='ij')
        self.V = pre * lq * k**4 * ck**2 * Sk
    def _calc_kmin (self, alpha_min):
        """Calculate kmin parameter of SCGLE as `alpha_min`
        times the position of the first maximum in the static
        structure factor.

        The function here simply looks for the maximum of the
        S(q) data, as given by `numpy.argmax`. If the first peak
        in S(q) happens to be not the overall maximum, you should
        investigate yourself and initialize the model with a given
        `k_min` directly."""
        # figure out argmax, get three points around it for quadratic fit
        i1 = np.argmax(self.sq)
        if i1 == 0 or i1 == self.M-1:
            return self.q[i1]
        x0, x1, x2 = self.q[i1-1], self.q[i1], self.q[i1+1]
        y0, y1, y2 = self.sq[i1-1], self.sq[i1], self.sq[i1+1]
        # quadratic fit with three known points, select maximum
        ptmp = (y2-y0)/(y2-y1)*(x2-x1)/(x2-x0)
        qmax = 0.5/(ptmp-1)*(ptmp*(x2+x1) - (x2+x0))
        return alpha_min * qmax
    def lambdak (self, q):
        """SCGLE cutoff function lambda_k(q)."""
        return 1./(1. + (q/self.k_min)**2)
    def make_kernel (self):
        V = self.V
        M = self.M
        dq = self.dq()
        sq = self.sq
        @njit
        def ker (m, phi, i, t):
            for qi in range(M):
                mq = 0.
                for ki in range(M):
                    mq += V[qi,ki] * phi[2*ki] * phi[2*ki+1]
                m[2*qi] = mq * dq[qi] * sq[qi]
                m[2*qi+1] = mq * dq[qi]
        return ker
    def set_C (self, f):
        # TODO calculate zeta(F) ??
        self._f_ = f
    def make_dm (self):
        V = self.V
        M = self.M
        q = self.q
        dq = self.dq()
        sq = self.sq
        fc = self._f_
        @njit
        def dm(m, phi, dphi):
            for qi in range(M):
                mq = 0.
                pre = sq[qi] * (1-fc[2*qi])**2
                pre = sq[qi]
                for ki in range(M):
                    mq += pre * V[qi,ki] * fc[2*ki+1] * (1-fc[2*ki])**2 * dphi[2*ki]
                    mq += pre * V[qi,ki] * fc[2*ki] * (1-fc[2*ki+1])**2 * dphi[2*ki+1]
                m[2*qi] = mq * dq[qi] / q[qi]**2
                mq = 0.
                pre = (1-fc[2*qi+1])**2
                pre = 1.
                for ki in range(M):
                    mq += pre * V[qi,ki] * fc[2*ki+1] * (1-fc[2*ki])**2 * dphi[2*ki]
                    mq += pre * V[qi,ki] * fc[2*ki] * (1-fc[2*ki+1])**2 * dphi[2*ki+1]
                m[2*qi+1] = mq * dq[qi] / q[qi]**2
        return dm
    def make_dmhat (self):
        V = self.V
        M = self.M
        q = self.q
        dq = self.dq()
        sq = self.sq
        fc = self._f_
        @njit
        def dmhat(m, f, ehat):
            for ki in range(M):
                mk = 0.
                for qi in range(M):
                    mk += sq[qi] * V[qi,ki] * fc[2*ki+1] * (1-fc[2*ki])**2 * ehat[2*qi] / q[qi]**2
                    mk += V[qi,ki] * fc[2*ki+1] * (1-fc[2*ki])**2 * ehat[2*qi+1] / q[qi]**2
                m[2*ki] = mk * dq[ki]
                mk = 0.
                for qi in range(M):
                    mk += sq[qi] * V[qi,ki] * fc[2*ki] * (1-fc[2*ki+1])**2 * ehat[2*qi] / q[qi]**2
                    mk += V[qi,ki] * fc[2*ki] * (1-fc[2*ki+1])**2 * ehat[2*qi+1] / q[qi]**2
                m[2*ki+1] = mk * dq[ki]
        return dmhat
    def make_dm2 (self):
        V = self.V
        M = self.M
        q = self.q
        dq = self.dq()
        sq = self.sq
        fc = self._f_
        @njit
        def dm2 (m, phi, dphi):
            for qi in range(M):
                mq = 0.
                for ki in range(M):
                    mq += V[qi,ki] * (1-fc[2*ki])**2 * dphi[2*ki] * (1-fc[2*ki+1])**2 * dphi[2*ki+1]
                    #mq += V[qi,ki] * dphi[2*ki] * dphi[2*ki+1]
                m[2*qi] = sq[qi] * mq * dq[qi] / q[qi]**2
                #m[2*qi] = mq * dq[qi] / q[qi]**2
                m[2*qi+1] = mq * dq[qi] / q[qi]**2
        return dm2
    #def shear_modulus: TODO
    def h5save (self, fh):
        grp = fh.create_group("model")
        grp.attrs['type'] = 'scgle'
        grp.attrs['M'] = self.M
        grp.attrs['dynamics'] = 'BD' # TODO FIXME
        grp.attrs['D0'] = self.D0
        grp.attrs['rho'] = self.rho
        grp.create_dataset("q",data=self.q)
        grp.create_dataset("sq",data=self.sq)
        grp.create_dataset("cq",data=self.cq)

