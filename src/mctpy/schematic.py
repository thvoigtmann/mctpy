import numpy as np
import scipy.linalg as la

from numba import njit, carray, objmode
from inspect import signature
from .__util__ import nparray, model_base, void

class generic (model_base):
    r"""Generic schematic model.

    Allows to declare some commonly used schematic models, with memory
    kernels of the form :math:`m[f] = \mathcal F[f]` for a given functional.

    Parameters
    ----------
    func : callable
        The memory-kernel functional. Must be a njit-able function.
        If its signature has one arugment, this is expected to be
        a memory-kernel functional specified as the funciton of the
        correlator phi. If it has two arguments, the solver will pass
        the time as well. Use this to specify explicit time-dependent
        memory kernels.
    M : int, default: 1
        The number of correlators expected by the functional.
 
    Notes
    -----
    Use this to implement more "exotic" models; the most commonly used
    schematic models such as the F12 model or the Bosse-Krieger model
    have their own implementations here.

    Currently, the calculation of eigenvalues is not supported for these
    generic models.

    Examples
    --------
    >>> v1, v2 = 0, 3.95
    >>> model = mct.schematic.generic (lambda x : v1*x + v2*x*x)

    >>> model = mct.schematic.generic (lambda x,t: -np.exp(-2*t))
    """
    def __init__ (self, func, M=1):
        model_base.__init__(self)
        self.func = njit(func)
        self.sig = len(signature(func).parameters)
        self.M = M
    def __len__ (self):
        return self.M
    def make_kernel (self):
        F = self.func
        if self.sig==2:
            @njit
            def ker(m, phi, i, t):
                m[:] = F(phi,t)
            return ker
        @njit
        def ker(m, phi, i, t):
            m[:] = F(phi)
        return ker

class f12model (model_base):
    r"""The schematic F12 model of mode-coupling theory.

    Declares a single memory kernel of the form
    :math:`m[f] = v_1 f + v_2 f^2`.

    Parameters
    ----------
    v1, v2 : float, float
        Coupling coefficients of the F12 model.
    delta : float, default: None
        Hopping parameter.
    """
    def __init__ (self, v1, v2, delta=None):
        model_base.__init__(self)
        self.v1 = v1
        self.v2 = v2
        self.delta = delta
    def __len__ (self):
        return 1
    def hopping (self):
        return self.delta
    def make_kernel (self):
        """Kernel-factory method.

        Returns njit-able function to evaluate the F12 model."""
        v1 = self.v1
        v2 = self.v2
        @njit
        def ker(m, phi, i, t):
            m[:] = v1*phi + v2*phi*phi
        return ker
    def make_dm (self):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dm(m, phi, dphi):
            m[:] = (1-phi) * (v1 * dphi + 2*v2 * phi*dphi) * (1-phi)
        return dm
    def make_dmhat (self):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dmhat(m, f, ehat):
            m[:] = (1-f)*(v1*ehat + 2*v2 * f*ehat)*(1-f)
        return dmhat
    def make_dm2 (self):
        v2 = self.v2
        @njit
        def dm2(m, phi, dphi):
            m[:] = (1-phi)**2 * v2 * dphi*dphi * (1-phi)**2
        return dm2

class f12gammadot_model (f12model):
    r"""Schematic F12-gammadot model.

    Schematic version of the MCT model for steady shear following the
    integration-through transients (ITT) approach developed by
    Fuchs and Cates.

    Parameters
    ----------
    v1, v2 : float, float
        The vertices of the F12 model.
    gammadot : float, default=0.0
        The shear rate applied to the model.
    gammac : float, default=0.1
        Parameter setting a strain scale for the loss of correlations.
    """
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1):
        f12model.__init__(self, v1, v2)
        self.gammadot = gammadot
        self.gammac = gammac
    def make_kernel (self):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        @njit
        def ker(m, phi, i, t):
            gt = gammadot*t / gammac
            m[:] = (v1*phi + v2*phi*phi) * 1.0/(1.0 + gt*gt)
        return ker

class f12gammadot_tensorial_model (f12gammadot_model):
    r"""Schematic F12-gammadot model, tensorial version.

    This is the adaptation of the :py:class:`mctspy.f12gammadot_model`
    for tensorial flow fields.

    Parameters
    ----------
    v1, v2 : float, float
        Vertices of the F12 model.
    gammadot : float, default: 0.0
        Flow rate as a scale for the velocity-gradient tensor.
    gammac : float, default: 0.1
        Parameter determining the strain scale for the relaxation dynamics.
    kappa : array_like, optional
        Velocity gradient tensor, normalized by the flow rate `gammadot`.
        Must have shape (3,3). The default is the matrix corresponding to
        simple shear in the xy-plane, i.e., only the [0,1] element different
        from zero.
    use_hhat : bool, default: False
        Flag whether to use the additional strain relaxation factor in
        front of the memory integral.
    nu : float, default: 1.0
        Mixing parameter determining the fraction of the strain reduction
        that is driven by the first invariant of the Finger tensor over
        the second invariant.
    """
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1,
                  kappa=np.array([[0,1,0],[0,0,0],[0,0,0]],dtype=float),
                  use_hhat=False, nu=1.0):
        f12gammadot_model.__init__(self, v1, v2, gammadot, gammac)
        self.kappa = kappa
        self.use_hhat = use_hhat
        self.nu = nu
    def make_kernel (self):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        nu = self.nu
        @njit
        def ker(m, phi, i, t):
            with objmode(ht='float64'):
                ht = self.hhat(t)
            m[:] = (v1*phi + v2*phi*phi) * ht
        return ker
    def hhat (self, t):
        gt = self.gammadot*t
        F = la.expm(self.kappa * gt)
        B = np.dot(F,F.T)
        I1 = np.trace(B)
        I2 = 0.0
        if self.nu != 1.0:
            try:
                I2 = np.trace(la.inv(B))
            except:
                I2 = 0.
        return 1.0 / (1. + (self.nu*I1 + (1-self.nu)*I2 - 3)/self.gammac**2)
    def kernel_prefactor (self, trange):
        if self.use_hhat:
            return np.array([self.hhat(t) for t in trange])
        else:
            return np.ones_like(trange)
    def finger (self, t):
        """Return the Finger tensor for time t."""
        gt = self.gammadot*t
        F = la.expm(self.kappa * gt)
        return np.dot(F,F.T)
    def dB (self, t):
        """Return derivative of the Finger tensor."""
        B = self.finger(t)
        return self.gammadot * (self.kappa @ B + B @ self.kappa.T)
    def dBinv (self, t):
        """Return derivative of the inverse Finger tensor."""
        Binv = la.inv(self.finger(t))
        return -self.gammadot * (self.kappa.T @ Binv + Binv @ self.kappa)
    def shear_modulus (self, phi, t, lc=False):
        """Return shear modulus including Finger-tensor factor.

        Parameters
        ----------
        phi, t : array_like
            Solution array as set by the solver.
        lc : bool, default: False
            If True, calculate a lower-convected variant of the integral,
            default is to use upper-convected version.
        """
        if not lc:
            dB_ = np.array([self.dB(td) for td in t]).reshape(-1,3,3)
        else:
            dB_ = np.array([-self.dBinv(td) for td in t]).reshape(-1,3,3)
        return dB_ * phi[:,0,None,None]**2


class sjoegren_model (model_base):
    r"""Sj\ |ouml|\ gren model.

    Defines a memory kernel that couples to a given (usually, F12) base
    model, and sets :math:`m^s[f^s] = v_s f f^c` where :math:`f` is taken
    from the base model.

    Parameters
    ----------
    vs : float
        Coupling parameter of the Sjoegren model.
    base_model : model_base
        The model to couple this to, such as a F12 model.
    deltas : float, default: None
        Hopping parameter.
    """
    def __init__ (self, vs, base_model, deltas=None):
        model_base.__init__(self)
        self.vs = vs
        self.deltas = deltas
        self.base = base_model
    def __len__ (self):
        return 1
    def hopping (self):
        return self.deltas
    def kernel_extra_args (self):
        return [self.base.phi]
    def make_kernel (self):
        vs = self.vs
        @njit
        def ker(ms, phis, i, t, phi):
            ms[:] = vs * phi[i] * phis
        return ker



class msd_model (model_base):
    def __init__ (self, vs, base_model):
        model_base.__init__(self)
        self.vs = vs
        self.base = base_model
    def __len__ (self):
        return 1
    def kernel_extra_args (self):
        return [self.base.phi]
    def make_kernel (self):
        vs = self.vs
        @njit
        def ker(ms, phis, i, t, phi):
            ms[:] = vs * phi[i] * phi[i]
        return ker


class bosse_krieger_model (model_base):
    """Bosse-Krieger model.

    A two-correlator model famous for the discussion of higher-order
    singularities."""
    def __init__ (self, v1, v2, v3):
        model_base.__init__(self)
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    def __len__ (self):
        return 2
    def make_kernel (self):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def ker (m, phi, i, t):
            m[0] = v1 * phi[0]*phi[0] + v2 * phi[1]*phi[1]
            m[1] = v3 * phi[0]*phi[1]
        return ker
    def make_dm (self):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dm (m, phi, dphi):
            dp = (1-phi)**2 * dphi
            m[0] = 2.*v1*phi[0]*dp[0] + 2.*v2*phi[1]*dp[1]
            m[1] = v3 * (phi[0]*dp[1] + phi[1]*dp[0])
        return dm
    def make_dmhat (self):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dmhat (m, f, ehat):
            m[0] = 2.*v1*f[0]*ehat[0]*(1-f[0])**2 \
                   +v3*f[1]*ehat[1]*(1-f[0])**2
            m[1] = 2.*v2*f[1]*ehat[0]*(1-f[1])**2 \
                   +v3*f[0]*ehat[1]*(1-f[1])**2
        return dmhat
    def make_dm2 (self):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dm2 (m, phi, dphi):
            dp = (1-phi)**2 * dphi
            m[0] = v1*dp[0]*dp[0]+v2*dp[1]*dp[1]
            m[1] = v3 * dp[0]*dp[1]
        return dm2

