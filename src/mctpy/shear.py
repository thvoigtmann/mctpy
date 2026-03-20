import numpy as np
from numba import njit

import scipy.integrate

from .standard import simple_liquid_model

class isotropically_sheared_model (simple_liquid_model):
    """Isotropipcally sheared model according to Fuchs/Cates.

    Parameters
    ----------
    Sq : object
        Static structure factor.
    q : array_like
        Regular wave number grid.
    D0 : float, default: 1.0
        Short-time diffusion coefficient.
    gammadot : float, default: 0.0
        Shear rate.
    gammac : float, default: 0.1
        Scaling factor for the strain.

    Notes
    -----
    See :py:class:`mctspy.simple_liquid_model` for restrictions on the
    wave number grid.

    This model only makes sense within Brownian dynamics.
    """
    def __init__ (self, Sq, q, D0=1.0, gammadot=0.0, gammac=0.1):
        self.gammadot = gammadot
        self.gammac = gammac
        simple_liquid_model.__init__ (self, Sq, q, D0)
        self.fixed_motion_type = 'brownian'
    def make_kernel (self):
        a, b, apre = self.a, self.b, self.apre
        M = self.M
        dq = self.dq()
        c = self.cq
        gammadot, gammac = self.gammadot, self.gammac
        @njit
        def ker (m, phi, i, t):
            advec = np.sqrt(1+(gammadot*t/gammac)**2)
            for qi in range(M):
                mq = 0.
                for n in range(6):
                    pi, ki = qi, 0
                    zqk = b[n,qi,qi] * phi[pi]
                    kt = advec/2
                    kibar = int(kt-0.499999)
                    if kibar < M-1:
                        cki = (c[kibar] + (c[kibar+1]-c[kibar])*(kt-kibar-0.5))/c[ki]
                    else:
                        cki = 0
                    mq += a[n,0] * apre[n] * zqk * cki * phi[ki]
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
                        kt += advec
                        kibar = int(kt-0.499999)
                        if kibar < M-1:
                            cki = (c[kibar] + (c[kibar+1]-c[kibar])*(kt-kibar-0.5))/c[ki]
                        else:
                            cki = 0
                        mq += a[n,ki] * apre[n] * zqk * cki * phi[ki]
                m[qi] = mq * dq[qi]**2
        return ker

    def shear_modulus (self, phi, t):
        r"""Return shear modulus or shear stress, given a solution.

        Parameters
        ----------
        phi : array_like
            Correlator or nonergodicity-parameter values.

        Returns
        -------
        G : array_like
            The shear modulus corresponding to the nonergodicity parameter
            or the time-dependent shear modulus corresponding to the
            time-dependent solution.
        """
        advec = np.sqrt(1 + self.gammadot*t/self.gammac)
        qt = self.q * advec[:,None]
        kt = qt/self.dq()
        kibar = np.array(np.floor(kt - 0.4999),dtype=int)
        kibar1 = np.array(np.floor(kt + 0.50001),dtype=int)
        kibar[kibar>=self.M-1] = self.M-1
        kibar1[kibar>=self.M-1] = self.M-1
        kibar_ = kibar.reshape(-1)
        kibar1_ = kibar1.reshape(-1)
        kt_ = kt.reshape(-1)
        cd = self.Sq.dcq_dq(self.q)
        cdbar = (cd[kibar_] + (cd[kibar1_] - cd[kibar_])*(kt_-kibar_-0.5)).reshape(kt.shape)
        Sqbar = (self.sq[kibar_] + (self.sq[kibar1_] - self.sq[kibar_])*(kt_-kibar_-0.5)).reshape(kt.shape)
        f = np.zeros_like(phi)
        for ti in range(f.shape[0]):
            f[ti,:] = phi[ti,kibar[ti]] + (phi[ti,kibar1[ti]] - phi[ti,kibar[ti]])*(kt[ti]-kibar[ti]-0.5)
            #spl = scipy.interpolate.CubicSpline(self.q, phi[ti,:],extrapolate=False)
            #f[ti,:] = spl(qt[ti])
        #f[np.isnan(f)]=0.
        # put this for now: it looks like that is what Fuchs/Cates did?
        f = phi
        return scipy.integrate.trapezoid(self.q**5/qt * cdbar * cd * \
            Sqbar**2 * f**2 * self.rho * self.dq(), axis=-1) / (60.*np.pi**2)

