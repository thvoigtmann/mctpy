import numpy as np

from numba import njit
from .__util__ import model_base, void, nparray, np_gradient, _dq
from .__h5__ import LoadableModel

import scipy.integrate

## simple_liquid_model kernels for numba
@njit
def simple_liquid_model_kernel (m, phi, i, t, a, b, dq):
    apre = np.array([1,1,1,2,2,2,2,2,2])
    M = dq.shape[0]
    for qi in range(M):
        mq = 0.
        for n in range(6):
            pi, ki = qi, 0
            zqk = b[n,qi,qi] * phi[pi]
            mq += a[n,0] * apre[n] * zqk * phi[ki]
            for ki in range(1, M):
                if ki <= qi:
                    pi = qi - ki
                    zqk += b[n,qi,pi] * phi[pi]
                else:
                    pi = ki - qi - 1
                    zqk -= b[n,qi,pi] * phi[pi]
                pi = qi + ki
                if pi < M:
                    zqk += b[n,qi,pi] * phi[pi]
                mq += a[n,ki] * apre[n] * zqk * phi[ki]
        m[qi] = mq * dq[qi]**2
@njit
def simple_liquid_model_calc_C (C, f, a, b):
    apre = np.array([1,1,1,2,2,2,2,2,2])
    M = C.shape[0]
    for qi in range(M):
        for n in range(9):
            pi, ki = qi, 0
            zqk = b[n,qi,qi] * f[pi]
            if n==0: C[qi,ki] = 0.
            C[qi,ki] += a[n,0] * zqk * (1-f[ki])**2
            for ki in range(1, M):
                if ki <= qi:
                    pi = qi - ki
                    zqk += b[n,qi,pi] * f[pi]
                else:
                    pi = ki - qi - 1
                    zqk -= b[n,qi,pi] * f[pi]
                pi = qi + ki
                if pi < M:
                    zqk += b[n,qi,pi] * f[pi]
                if n==0: C[qi,ki] = 0.
                C[qi,ki] += a[n,ki] * zqk * (1-f[ki])**2
@njit
def simple_liquid_model_dm(m, phi, dphi, V, dq, q):
    m[:] = 2 * dq**2/q**2 * np.dot(V,dphi)
@njit
def simple_liquid_model_dmhat(m, f, ehat, V, dq, q):
    m[:] = np.dot(ehat/q**2, V) * 2 * dq**2
@njit
def simple_liquid_model_dm2 (m, phi, dphi, a, b, dq, q):
    apre = np.array([1,1,1,2,2,2,2,2,2])
    M = dq.shape[0]
    for qi in range(M):
        mq = 0.
        for n in range(6):
            pi, ki = qi, 0
            zqk = b[n,qi,qi] * dphi[pi] * (1-phi[pi])**2
            mq += a[n,0] * apre[n] * zqk * dphi[ki] * (1-phi[ki])**2
            for ki in range(1, M):
                if ki <= qi:
                    pi = qi - ki
                    zqk += b[n,qi,pi] * dphi[pi] * (1-phi[pi])**2
                else:
                    pi = ki - qi - 1
                    zqk -= b[n,qi,pi] * dphi[pi] * (1-phi[pi])**2
                pi = qi + ki
                if pi < M:
                    zqk += b[n,qi,pi] * dphi[pi] * (1-phi[pi])**2
                mq += a[n,ki] *apre[n]* zqk * dphi[ki] * (1-phi[ki])**2
        m[qi] = mq * dq[qi]**2 / q[qi]**2
class simple_liquid_model (model_base, LoadableModel,
                           kernel=simple_liquid_model_kernel,
                           dm=simple_liquid_model_dm,
                           dmhat=simple_liquid_model_dmhat,
                           dm2=simple_liquid_model_dm2,
                           fields=['q','sq','cq'], attrs=['rho','D0'],
                           objects=['Sq'], resetable=['q','Sq','D0','vth'],
                           safe_reset=['D0','vth']):
    """The simple-liquid standard model of MCT.

    Parameters
    ----------
    Sq : object
        The static structure factor object.
    q : array_like
        Wave number grid.
    D0 : float, default: 1.0
        Short-term diffusion coefficient, used for Brownian dynamics.
    vth : float, default: 1.0
        Thermal velocity, used for Newtonian dynamics.

    Notes
    -----
    The wave number grid *must* be regular with step size dq and
    have its first point at dq/2.
    """
    def __init__ (self, Sq, q, D0=1.0, vth=1.0):
        super().__init__()
        self.D0 = D0
        self.vth = vth
        self.q = q
        self.Sq = Sq
        self.sq, self.cq = None, None
        self.__init_vertices__()
    def reinit (self):
        self.__init_vertices__()
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Aq (self):
        return self.sq/self.vth
    def Bq (self):
        return self.sq/self.D0
    def dq (self):
        return _dq(self.q)
    def __init_vertices__ (self):
        q = self.q
        self.sq, self.cq = self.Sq.Sq(self.q)
        self.rho = self.Sq.density()
        self.M = q.shape[0]
        self.a = np.zeros((9,self.M))
        self.b = np.zeros((9,self.M,self.M))
        self.__C__ = None
        pre = 1./(32.*np.pi**2) * self.rho
        qSq = q * self.sq
        self.a[0] = qSq * 2 * self.cq
        self.a[1] = - qSq * 2 * q**4 * self.cq
        self.a[2] = qSq * 4 * q**2 * self.cq
        self.a[3] = qSq * q**4 * (self.cq)**2
        self.a[4] = qSq * 2 * q**2 * (self.cq)**2
        self.a[5] = qSq * (self.cq)**2
        self.a[6] = qSq * q**4
        self.a[7] = - qSq * 2 * q**2
        self.a[8] = qSq
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        pSp = p_/q_ * np.outer(self.sq, self.sq) * pre / q_**2
        cq, cp = np.meshgrid(self.cq, self.cq, indexing='ij')
        qsq_psq = q_**2 - p_**2
        qsqppsq = q_**2 + p_**2
        self.b[0] = pSp * (q_**4 - p_**4) * cp
        self.b[1] = pSp * cp
        self.b[2] = pSp * p_**2 * cp
        self.b[3] = pSp
        self.b[4] = pSp * qsq_psq
        self.b[5] = pSp * qsq_psq**2
        self.b[6] = pSp * cp**2
        self.b[7] = pSp * qsqppsq * cp**2
        self.b[8] = pSp * qsqppsq**2 * cp**2
    def kernel_extra_args (self):
        return [self.a, self.b, self.dq()]
    def set_C (self, f):
        if self.__C__ is None:
            self.__C__ = np.zeros((self.M,self.M))
        simple_liquid_model_calc_C (self.__C__, f, self.a, self.b)
    def kernel_extra_args_dm (self):
        return [self.__C__,self.dq(),self.q]
    def kernel_extra_args_dm2 (self):
        return [self.a, self.b, self.dq(), self.q]
    def mrenorm (self, m):
        return m/self.q**2
    def shear_modulus (self, phi, t):
        r"""Return shear modulus or shear stress, given a solution.

        Parameters
        ----------
        phi : array_like
            Correlator or nonergodicity-parameter values.
        t : array_like
            Time points for the integral; not needed in this implementation.

        Returns
        -------
        G : array_like
            The shear modulus corresponding to the nonergodicity parameter
            or the time-dependent shear modulus corresponding to the
            time-dependent solution.
        """
        return scipy.integrate.trapezoid((self.q**4 * self.sq**2 * \
            self.Sq.dcq_dq(self.q)**2 * self.rho*self.dq()) * \
            phi**2, axis=-1) / (60*np.pi**2)


@njit
def tagged_particle_model_calc_V (i, V, phi, a, b):
    f = phi[i]
    M = V.shape[0]
    for qi in range(M):
        for ki in range(M):
            V[qi,ki] = 0.
        for n in range(3):
            pi, ki = qi, 0
            zqk = b[n,qi,pi] * f[pi]
            V[qi,ki] += a[n,ki] * zqk
            for ki in range(1, M):
                if ki <= qi:
                    pi = qi - ki
                    zqk += b[n,qi,pi] * f[pi]
                else:
                    pi = ki - qi - 1
                    zqk -= b[n,qi,pi] * f[pi]
                pi = qi + ki
                if pi < M:
                    zqk += b[n,qi,pi] * f[pi]
                V[qi,ki] += a[n,ki] * zqk
@njit(inline='always')
def tagged_particle_model_kernel (ms, phis, i, t, V, dq):
    #def tagged_particle_model_kernel (ms, phis, i, t, phi, V, last_i, Vfunc, dq):
    #if last_i[0]==0 or not (i==last_i[0]):
    #    Vfunc(V, phi[i])
    #    last_i[0] = i
    ms[:] = dq**2 * np.dot(V,phis)
class tagged_particle_model (model_base, LoadableModel,
                             kernel=tagged_particle_model_kernel,
                             setV=tagged_particle_model_calc_V,
                             fields=['cs'], attrs=['D0','vth'],
                             objects=['base','Sq'],
                             resetable=['base','Sq','D0','vth'],
                             safe_reset=['D0','vth']):
    def __init__ (self, base_model, Sq, D0s=1.0, vth=1.0):
        super().__init__()
        self.D0 = D0s
        self.vth = vth
        self.base = base_model
        self.Sq = Sq
        self.cs = None
        self.reinit()
    def reinit (self):
        self.__init_vertices__()
    def get_bases (self):
        return [self.base]
    def __len__ (self):
        return self.M
    def Aq (self):
        return np.ones(self.M)/self.vth
    def Bq (self):
        return np.ones(self.M)/self.D0
    def Wq (self):
        return self.q*self.q
    def __init_vertices__ (self):
        self.q = self.base.q
        self.cs = self.Sq.cq(self.q)
        self.M = self.q.shape[0]
        pre = (1./(4.*np.pi))**2.
        q = self.q
        self.rho_csq = self.base.rho * self.cs**2 * self.base.sq
        self.a = np.zeros((3,self.M))
        self.b = np.zeros((3,self.M,self.M))
        self.a[0] = q**4 * q
        self.a[1] = - 2 * q**2 * q
        self.a[2] = q
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        pq = p_/q_**3 * pre * self.rho_csq
        qsq_psq = q_**2 + p_**2
        self.b[0] = pq
        self.b[1] = pq * qsq_psq
        self.b[2] = pq * qsq_psq**2
        #
        self.__Vqk__ = np.zeros((self.M,self.M))
    def kernel_extra_args (self):
        return (self.__Vqk__, _dq(self.base.q))
    def set_V_args (self):
        return (self.__Vqk__, self.base.phi, self.a, self.b)
    def mrenorm (self, m):
        return m/self.q**2



@njit(inline='always')
def tagged_particle_q0_kernel (ms0, phis0, i, t, V, phi, phi_s):
    #ms0[:] = np.dot(V,phi[i]*phi_s[i]) # good for one-component only
    ms0[:] = np.sum((V*phi[i]).T*phi_s[i])

class tagged_particle_q0 (model_base, LoadableModel,
                          kernel=tagged_particle_q0_kernel,
                          objects=['base']):
    def __init__ (self, base_model):
        super().__init__()
        self.base = base_model
        self.reinit()
    def reinit(self):
        self.__init_vertices__()
    def get_bases (self):
        return [self.base, self.base.base]
    def __len__ (self):
        return 1
    def Bq (self):
        return np.ones(len(self))/self.base.D0
    def __init_vertices__ (self):
        pre = 1./(6.*np.pi**2) * _dq(self.base.base.q)[0] * self.base.base.rho
        # the double transpose effectively means
        # "broadcast from left"
        # coupling to simple liquid, q.shape = (M,), rho_csq.shape = (M,)
        # but for mixture, rho_csq.shape = (M,S,S,)
        # resulting V has shape (M,) or (M,S,S,)
        self.V = pre * (self.base.q**4 * self.base.rho_csq.T).T
    def set_base_phi (self, array):
        model_base.set_base_phi (self, array)
    def kernel_extra_args (self):
        return [self.V, self.base.base.phi, self.base.phi]

@njit
def tagged_particle_ngp_kernel (ngpm, ngpphi, i, t, phi, phis, V, k):
    dphis = np_gradient(phis[i],k)
    dd = np_gradient(dphis,k) + (2./3)*dphis/k
    ngpm[0] = np.dot(V,phi[i]*phis[i]) # one-component only
    ngpm[1] = (3./5) * np.dot(V,phi[i]*dd)
class tagged_particle_ngp (tagged_particle_q0, LoadableModel,
                           kernel=tagged_particle_ngp_kernel,
                           objects=['msdbase']):
    r"""Standard model for the non-Gaussian parameter.

    Parameters
    ----------
    base_model : object
        A model representing the mean-squared displacement (MSD,
        i.e. zero-wavenumber limit of a tagged-particle kernel model).
        In particular, this model must itself define a base that
        corresponds to the finite-wavenumber tagged-particle kernel,
        which in turn nneds to define a base that is the collective
        finite-wavenumber kernel.

    Notes
    -----
        This model works on a pair of correlation functions, the first is the
        NGP function :math:`a(t)=(1-\alpha_2(t))\delta r^2(t)^2`,
        and the second is the MSD itself. The two memory kernels that are
        returned are the :math:`m^s_0(t)` and :math:`\tilde m^s_0(t)`
        defined in the MCT equation for the NGP.
    """
    def __init__ (self, base_model):
        tagged_particle_q0.__init__ (self, base_model.base)
        self.msdbase = base_model
    def __len__ (self):
        return 2
    def kernel_extra_args (self):
        # since we inherit from tagged_particle_q0, the "base" is phi_s
        # and the base.base is phi
        return [self.base.base.phi, self.base.phi, self.V, self.base.q]
    def phi2 (self):
        return self.msdbase.phi

