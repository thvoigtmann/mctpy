import numpy as np
import scipy.linalg as la

from numba import njit, complex128
from .__util__ import model_base, void, nparray
from .standard_2d import g0unsum, g1unsum, g2unsum

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

@njit
def _calc_phase(phase,L,q,k,p,x,y):
    for l1 in range(-L,L+1):
        for l2 in range(-L,L+1):
            for l3 in range(-L,L+1):
                for l4 in range(-L,L+1):
                    mu, nu = l3-l1, l4-l2
                    smu, snu = np.sign(mu), np.sign(nu)
                    a = np.arange(abs(mu)+1)[:,None]
                    b = np.arange(abs(nu)+1)[None,:]
                    phase[:,L+l1,L+l2,L+l3,L+l4] = np.sum(np.sum(x[:,None,None]**(abs(mu)-a) * y[:,None,None]**a * (smu*1j)**a * (-snu*1j)**b * (k/p)[:,None,None]**abs(nu) * (q/k-x)[:,None,None]**(abs(nu)-b) * y[:,None,None]**b, axis=-1), axis=-1)

class abp_model_2d (model_base):
    def __init__ (self, Sq, q, L=1, D0=1.0, Dr=1.0, v0=0.0, nu=0.0):
        """Initialize active Brownian-particle (ABP) model.

        Parameters
        ----------
        Sq : object or None
            Should be a suitable (2d) static structure factor object
            that provides a density, S(q), and c(q). As a special case,
            can be None if only the analytical low-density solutions are
            needed.
        L : int, default: 1
            Angular base-function cutoff. The model will result in matrices
            of dimension (2L+1) times (2L+1).
        D0 : float, default: 1.0
            Translational diffusion coefficient.
        Dr : float, default: 1.0
            Rotational diffusion coefficient.
        v0 : float, default: 0.0
            Self-propulsion velocity of the ABP.
        nu : float, default: 0.0
            If set to a non-zero value, the model is that of a circle
            swimmer or chiral active Brownian particle."""
        model_base.__init__(self)
        self.dtype = np.dtype('complex')
        self.fixed_motion_type = 'brownian'
        self.q = q
        self.Sq = Sq
        self.M = q.shape[0]
        self.L = L
        self.S = 2*L+1
        self.D0 = D0
        self.Dr = Dr
        self.v0 = v0
        self.nu = nu
        if Sq is not None:
            self.rho = Sq.density()
            self.sq, self.cq = Sq.Sq(q)
            self.__init_vertices__ ()
        else:
            self.rho = 0.0
    def __len__ (self):
        return self.M
    def matrix_dimension (self):
        return self.S
    def scalar (self):
        return False
    def hopping (self):
        S = 2*self.L+1
        res = self.omega_R (self.L)
        res[:,self.L,self.L] = res[:,self.L,self.L] / self.sq
        return res
    def phi0 (self):
        phi0 = np.ones((self.M,1,1),dtype=self.dtype) * np.diag(np.ones(self.S))
        phi0[:,self.L,self.L] = self.sq
        return phi0
    def Bq (self):
        return self.wTinv
    def Bqinv (self):
        return self.omega_T(self.L)
    def Wq (self):
        res = self.WqSq()
        res[:,self.L,self.L] = res[:,self.L,self.L] / self.sq
        return res
    def WqSq (self):
        wTinv_wR = np.zeros((self.M,self.S,self.S),dtype=self.dtype)
        for q in range(self.M):
            wTinv_wR[q] = np.dot(self.wTinv[q],
                          self.Dr * np.diag(np.arange(-self.L, self.L+1)**2))
        return (self.q**2)[:,None,None] * np.diag(np.ones(self.S)) \
               + wTinv_wR
    def dq (self):
        return _dq(self.q)
    def omega_T (self, Lcut=None, lowdens=False):
        r"""Return translation-frequency matrix, divided by q^2.

        Parameters
        ----------
        Lcut : int, optional
            L cutoff to use for the matrix. If None, take the one
            provided during initialization of this object.
        lowdens : boolean, default: False
            If True, omit the S(q) factor in the calculation, returning
            the low-density matrix.
        """
        if Lcut is None:
            L, S = self.L, self.S
        else:
            L, S = Lcut, 2*Lcut+1
        wT = np.zeros((self.M,S,S),dtype=self.dtype)
        for l in range(-L,L+1):
            for ld in range(-L,L+1):
                if l==ld:
                    wT[:,L+l,L+ld] = self.D0
                if abs(l-ld)==1:
                    if l==0 and not lowdens:
                        wT[:,L+l,L+ld] = - 0.5j/self.q*self.v0 * self.sq
                    else:
                        wT[:,L+l,L+ld] = - 0.5j/self.q*self.v0
        return wT
    def omega_s_T_inv (self, Lcut=None):
        r"""Return inverse self-translation-frequency matrix, times q^2D_0."""
        if Lcut is None:
            L, S = self.L, self.S
        else:
            L, S = Lcut, 2*Lcut+1
        Delta = np.sqrt(1. + (self.v0/self.D0/self.q)**2)
        wTinv0 = np.zeros((self.M,S,S),dtype=self.dtype)
        for l in range(-L,L+1):
            for ld in range(-L,L+1):
                wTinv0[:,L+l,L+ld] = np.power(1j*self.v0/self.D0/self.q/ \
                                     (1 + Delta),abs(l-ld)) / Delta
        return wTinv0
    def omega_T_inv (self, Lcut=None, lowdens=False):
        r"""Return inverse translation-frequency matrix, times q^2D_0."""
        # this calculates the cut-off of the proper inf-dim inverse:
        if Lcut is None:
            L = self.L
        else:
            L = Lcut
        wTinv0 = self.omega_s_T_inv (L)
        if lowdens:
            return wTinv0
        wTinv = wTinv0.copy()
        if L > 0:
            u0 = -0.5j*self.q*self.v0/self.D0 * (self.sq - 1)
            for l in range(-L,L+1):
                for ld in range(-L,L+1):
                    wTinv[:,L+l,L+ld] -= u0*wTinv0[:,L+l,L] * \
                        (wTinv0[:,L+1,L+ld]+wTinv0[:,L-1,L+ld])/ \
                        (self.q**2*self.D0 + u0*(wTinv0[:,L+1,L]+wTinv0[:,L-1,L]))
        return wTinv
        # the following would be direct inversion of the cut-off matrix:
        #wTinv = self.omega_T(Lcut)
        #for q in range(self.M):
        #    wTinv[q] = la.inv(wTinv[q])
        #return wTinv
    def omega_R (self, Lcut=None):
        """Return rotation-frequency matrix."""
        if Lcut is None:
            L, S = self.L, self.S
        else:
            L, S = Lcut, 2*Lcut+1
        return np.ones((self.M,S,S),dtype=self.dtype) \
               * (self.Dr *np.diag(np.arange(-Lcut, Lcut+1)**2) \
                 - self.nu * 1j * np.arange(-Lcut, Lcut+1))
    def low_density_solution (self, t, Lcut=None):
        """Return the low-density density correlation matrix.

        Parameters
        ----------
        t : array_like
            Set of time points for which to calculate the solution.
        Lcut : int, optional
            If given, use this L cutoff for the calculation. If None,
            use the one provided during initialization of self.

        Notes
        -----
        The calculation performed is the matrix exponential of the
        frequency matrix in the Mori-Zwanzig representation, using the
        given cutoff. For finite cutoff, this is different than e.g.
        the representation in terms of Mathieu functions.
        """
        if Lcut is None:
            L, S = self.L, self.S
        else:
            L, S = Lcut, 2*Lcut+1
        phi = np.zeros((t.shape[0], self.M, S, S), dtype=self.dtype)
        w = (self.q**2)[:,None,None] * self.omega_T(Lcut,lowdens=True) \
            + self.omega_R(Lcut)
        for q in range(self.M):
            wqt = np.einsum('i,jk->ijk', t, -w[q])
            phi[:,q,:,:] = la.expm(wqt)
        return phi
    def low_density_MSD (self, t):
        """Return the low-density MSD.

        Parameters
        ----------
        t : array_like
            Set of time points for which to calculate the solution."""
        Dt, Dr = self.D0, self.Dr
        v0, nu = self.v0, self.nu
        return 4*Dt*t + 2*v0*v0*Dr/(Dr**2+nu**2) * (t \
              + (Dr**2-nu**2)/(Dr**2+nu**2)*(np.exp(-Dr*t)*np.cos(nu*t)-1)/Dr \
              - 2*nu/(Dr**2+nu**2)*np.exp(-Dr*t)*np.sin(nu*t))
            #+ 2*self.v0*self.v0/self.Dr*(t+(np.exp(-self.Dr*t)-1)/self.Dr)
    def low_density_NGP (self, t):
        """Return the low-density non-Gaussian parameter (NGP).

        Parameters
        ----------
        t : array_like
            Set of time points for which to calculate the solution.

        Notes
        -----
        Use with care. The formula is not implemented in a stale way,
        it is some messy Mathematica output."""
        Dt, Dr = self.D0, self.Dr
        v0, nu = self.v0, self.nu
        if not nu > 0.0:
            return -1 + (Dr**4*((87*v0**4)/(16*Dr**4) \
                + (np.exp(-4*Dr*t)*v0**4)/(144*Dr**4) \
                - (49*np.exp(-Dr*t)*v0**4)/(9*Dr**4) \
                + (t*(-16*Dr*Dt*v0**2 - 15*v0**4))/(4*Dr**3) \
                + (np.exp(-Dr*t)*t*(12*Dr*Dt*v0**2 - 5*v0**4))/(3*Dr**3) \
                + (t**2*(4*Dr**2*Dt**2 + 4*Dr*Dt*v0**2 + v0**4))/ Dr**2)) \
                / (2*Dr**2*Dt*t + (-1 + np.exp(-Dr*t))*v0**2 + Dr*t*v0**2)**2
        return (-1 + (2*((np.exp(-4*Dr*t + 2j*t*nu)*v0**4) \
            /( 64*(2*Dr - 1j*nu)**2*(3*Dr - 1j*nu)**2) \
            + (np.exp(-4*Dr*t - 2j*t*nu)*v0**4) \
            /( 64*(2*Dr + 1j*nu)**2*(3*Dr + 1j*nu)**2) \
            + (np.exp(-Dr*t + 1j*t*nu)*t*(6*Dr**2*Dt*v0**2 - Dr*v0**4 \
            - 8j*Dr*Dt*v0**2*nu - 2*Dt*v0**2*nu**2)) \
            /(8*(Dr - 1j*nu)**3*(3*Dr - 1j*nu)) \
            + (np.exp(-t*(Dr + 1j*nu))*t*(6*Dr**2*Dt*v0**2 - Dr*v0**4 \
            + 8j*Dr*Dt*v0**2*nu - 2*Dt*v0**2*nu**2))\
            /(8*(Dr + 1j*nu)**3*(3*Dr + 1j*nu)) \
            + (np.exp(-t*(Dr + 1j*nu))*(-9j*Dr**3*v0**4 - 7*Dr**2*v0**4*nu \
            - 5j*Dr*v0**4*nu**2 + v0**4*nu**3))\
            /(16*(Dr + 1j*nu)**4*(3*Dr + 1j*nu)**2*nu) \
            + (np.exp(-Dr*t + 1j*t*nu)*(9j*Dr**3*v0**4 - 7*Dr**2*v0**4*nu \
            + 5j*Dr*v0**4*nu**2 + v0**4*nu**3))\
            /(16*(Dr - 1j*nu)**4*(3*Dr - 1j*nu)**2*nu) \
            + (t**2*(4*Dr**4*Dt**2 + 4*Dr**3*Dt*v0**2 + Dr**2*v0**4 \
            + 8*Dr**2*Dt**2*nu**2 + 4*Dr*Dt*v0**2*nu**2 + 4*Dt**2*nu**4)) \
            /(8*(Dr**2 + nu**2)**2) \
            + (t*(-16*Dr**6*Dt*v0**2 - 15*Dr**5*v0**4 - 4*Dr**4*Dt*v0**2*nu**2 \
            + 11*Dr**3*v0**4*nu**2 + 16*Dr**2*Dt*v0**2*nu**4 \
            + 2*Dr*v0**4*nu**4 + 4*Dt*v0**2*nu**6))\
            /(8*(Dr**2 + nu**2)**3*(4*Dr**2 + nu**2)) \
            + (3*(116*Dr**8*v0**4 - 241*Dr**6*v0**4*nu**2 \
            - 61*Dr**4*v0**4*nu**4 + 9*Dr**2*v0**4*nu**6 + v0**4*nu**8))\
            /(32*(Dr**2 + nu**2)**4*(4*Dr**2 + nu**2)**2)))\
            /(-((np.exp(-Dr*t + 1j*t*nu)*v0**2)/(4*(Dr - 1j*nu)**2)) \
            - (np.exp(-t*(Dr + 1j*nu))*v0**2)/(4*(Dr + 1j*nu)**2) \
            + (t*(-2*Dr**2*Dt - Dr*v0**2 - 2*Dt*nu**2))/(2*(Dr**2 + nu**2)) \
            + (Dr**2*v0**2 - v0**2*nu**2) \
            /(2*(Dr - 1j*nu)**2*(Dr + 1j*nu)**2))**2).real
    def __init_vertices__ (self):
        q = self.q
        L=self.L
        self.wTinv = self.omega_T_inv(self.L)
        self.__A__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        self.__B__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        self.__C__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        self.__D__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
        #self.__phase__ = np.zeros((self.M,self.S,self.S,self.S,self.S),dtype=self.dtype)
        A, B, C, D = self.__A__, self.__B__, self.__C__, self.__D__
        Pe_t = self.v0/self.D0
        if not Pe_t == 0:
            self.__Av__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Bv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Cv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Dv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Cvc__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Dvc__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Avv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Avvc__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Bvv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Bvvc__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Cvv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            self.__Dvv__ = np.zeros((self.M,self.M,self.M,self.S,self.S),dtype=self.dtype)
            Av, Bv, Cv, Dv = self.__Av__, self.__Bv__, self.__Cv__, self.__Dv__
            Cvc, Dvc = self.__Cvc__, self.__Dvc__
            Avv, Bvv, Cvv, Dvv = self.__Avv__, self.__Bvv__, self.__Cvv__, self.__Dvv__
            Avvc, Bvvc = self.__Avvc__, self.__Bvvc__
        else:
            self.__Av__ = np.zeros((1,1,1,1,1))
            self.__Bv__ = np.zeros((1,1,1,1,1))
            self.__Cv__ = np.zeros((1,1,1,1,1))
            self.__Dv__ = np.zeros((1,1,1,1,1))
            self.__Cvc__ = np.zeros((1,1,1,1,1))
            self.__Dvc__ = np.zeros((1,1,1,1,1))
            self.__Avv__ = np.zeros((1,1,1,1,1))
            self.__Bvv__ = np.zeros((1,1,1,1,1))
            self.__Avvc__ = np.zeros((1,1,1,1,1))
            self.__Bvvc__ = np.zeros((1,1,1,1,1))
            self.__Cvv__ = np.zeros((1,1,1,1,1))
            self.__Dvv__ = np.zeros((1,1,1,1,1))
        q = self.q
        M = self.M
        lr = np.arange(-L,L+1)
        pre = self.rho/(8*np.pi**2)
        v0pre = pre * 0.5j*Pe_t*self.sq
        Sq, c = self.sq, self.cq
        for qi in range(M):
            for ki in range(M):
                if ki <= qi:
                    pmin = q[qi] - q[ki]
                else:
                    pmin = q[ki] - q[qi]
                pmax = q[qi] + q[ki]
                x = (q[qi]**2 + q[ki]**2 - q**2) / (2*q[qi]*q[ki])
                mask = (x>=-1) & (x<=1) & (q>=pmin)
                x = x[mask]
                p = np.arange(M)[mask]
                minval = -1.0
                if pmax > q[-1]:
                    minval = x[-1]
                y = np.sqrt(1-x*x)
                thk = np.power(x[:,None]+1j*y[:,None],lr)
                thp = np.power(q[ki]/q[p,None],lr) * np.power((q[qi]/q[ki]-x[:,None])-1j*y[:,None],lr)
                assert (not np.isnan(thk).any())
                assert (not np.isnan(thp).any())
                # A: terms phip(0 ,0 ), phik(l ,l')
                # B: terms phip(0 ,l'), phik(l, 0 )
                # C: terms phip(l, 0 ), phik(0, l')
                # D: terms phip(l, l'), phik(0, 0 )
                for l in lr:
                    for ld in lr:
                        tmp = c[p]**2 * thk[:,L-l]*thk[:,L+ld]
                        A[qi,ki,p,L+l,L+ld] = pre * (\
                            2*q[qi]**2 * g0unsum(tmp,x,1,minval) \
                            - 4*q[qi]*q[ki] * g1unsum(tmp,x,1,minval) \
                            + 2*q[ki]**2 * g2unsum(tmp,x,1,minval) )
                        tmp = c[ki]*c[p] * thk[:,L-l]*thp[:,L+ld]
                        B[qi,ki,p,L+l,L+ld] = pre * (\
                                 2*q[qi]*q[ki] * g1unsum(tmp,x,1,minval) \
                                 - 2*q[ki]**2 * g2unsum(tmp,x,1,minval) )
                        tmp = c[ki]*c[p] * thk[:,L+ld]*thp[:,L-l]
                        C[qi,ki,p,L+l,L+ld] = pre * (\
                                 2*q[qi]*q[ki] * g1unsum(tmp,x,1,minval) \
                                 - 2*q[ki]**2 * g2unsum(tmp,x,1,minval) )
                        tmp = c[ki]**2 * thp[:,L-l]*thp[:,L+ld]
                        D[qi,ki,p,L+l,L+ld] = pre * 2*q[ki]**2 * \
                            g2unsum(tmp,x,1,minval)
                        if not Pe_t==0:
                            # Av: terms phip(-1,0 )+phip(1 , 0), phik(l ,l')
                            # Bv: terms phip(-1,l')+phip(1 ,l'), phik(l, 0 )
                            # Cv: terms phip(l-1,0), phik(0,l')
                            # Cv*(-l,-l'): phip(l+1,0), phik(0,l')
                            # Dv: terms phip(l-1,l'), phik(0,0)
                            # Dv*(-l,-l'): phip(l+1,l'), phik(0,0)
                            # Avv: terms phip(0,0), phik(l-1,l')
                            # Avv*(-l,-l'): phip(0,0), phik(l+1,l')
                            # Bvv: terms phip(0,l') phik(l-1,0)
                            # Bvv*(-l,-l'): phip(0,l') phik(l+1,0)
                            # Cvv: terms phip(l,0) phik(-1,l')+phik(1,l')
                            # Dvv: terms phip(l,l') phik(-1,0)+phik(1,0)
                            if L>=1:
                                tmp = p*Sq[p]*c[p]**2 * thk[:,L-l]*thk[:,L+ld]
                                Av[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    2 * g0unsum(tmp,x,1,minval) \
                                    - 2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval) )
                                tmp = p*Sq[p]*c[p]*c[ki] * thk[:,L-l]*thp[:,L+ld]
                                Bv[qi,ki,p,L+l,L+ld] = v0pre[qi] * \
                                    2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval)
                            if L>=abs(l)+1:
                                tmp = q[ki]*c[ki]*c[p] * thk[:,L+ld]*thp[:,L-l] * thk[:,L-1]*thp[:,L+1]
                                Cv[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    - 2 * g0unsum(tmp,x,1,minval) \
                                    + 2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval) )
                                tmp = q[ki]*c[ki]*c[p] * thk[:,L+ld]*thp[:,L-l] * thk[:,L+1]*thp[:,L-1]
                                Cvc[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    - 2 * g0unsum(tmp,x,1,minval) \
                                    + 2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval) )
                                tmp = q[ki]*c[ki]**2 * thp[:,L-l]*thp[:,L+ld] * thk[:,L-1]*thp[:,L+1]
                                Dv[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    (-2) * q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                                tmp = q[ki]*c[ki]**2 * thp[:,L-l]*thp[:,L+ld] * thk[:,L+1]*thp[:,L-1]
                                Dvc[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    (-2) * q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                                tmp = p*c[p]**2 * thk[:,L-l]*thk[:,L+ld] * thk[:,L+1]*thp[:,L-1]
                                Avv[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    -2 * g0unsum(tmp,x,1,minval) \
                                    + 2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                                tmp = p*c[p]**2 * thk[:,L-l]*thk[:,L+ld] * thk[:,L-1]*thp[:,L+1]
                                Avvc[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    -2 * g0unsum(tmp,x,1,minval) \
                                    + 2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                                tmp = p*c[ki]*c[p] * thk[:,L-l]*thp[:,L+ld] * thk[:,L+1]*thp[:,L-1]
                                Bvv[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    -2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                                tmp = p*c[ki]*c[p] * thk[:,L-l]*thp[:,L+ld] * thk[:,L-1]*thp[:,L+1]
                                Bvvc[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    -2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                            if L>=1:
                                tmp = q[ki]*Sq[ki]*c[ki]*c[p] * thp[:,L+ld]*thk[:,L-l]
                                Cvv[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    + 2 * g0unsum(tmp,x,1,minval) \
                                    - 2*q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                                tmp = q[ki]*Sq[ki]*c[ki]**2 * thp[:,L-l]*thp[:,L+ld]
                                Dvv[qi,ki,p,L+l,L+ld] = v0pre[qi] * ( \
                                    2 * q[ki]/q[qi] * g1unsum(tmp,x,1,minval))
                assert (not np.isnan(A[qi,ki]).any())
                assert (not np.isnan(B[qi,ki]).any())
                assert (not np.isnan(C[qi,ki]).any())
                assert (not np.isnan(D[qi,ki]).any())
                if not Pe_t == 0:
                    assert (not np.isnan(Av[qi,ki]).any())
                    assert (not np.isnan(Bv[qi,ki]).any())
                    assert (not np.isnan(Cv[qi,ki]).any())
                    assert (not np.isnan(Dv[qi,ki]).any())
                    assert (not np.isnan(Avv[qi,ki]).any())
                    assert (not np.isnan(Bvv[qi,ki]).any())
                    assert (not np.isnan(Cvv[qi,ki]).any())
                    assert (not np.isnan(Dvv[qi,ki]).any())

    def make_kernel (self):
        if not self.rho > 0.0:
            @njit
            def ker (m, phi, i, t):
                return
            return ker
        M, S, L = self.M, self.S, self.L
        q = self.q
        pre = self.rho/(8*np.pi**2)
        Aqk = void(self.__A__)
        Bqk = void(self.__B__)
        Cqk = void(self.__C__)
        Dqk = void(self.__D__)
        Avqk = void(self.__Av__)
        Bvqk = void(self.__Bv__)
        Cvqk = void(self.__Cv__)
        Dvqk = void(self.__Dv__)
        Cvcqk = void(self.__Cvc__)
        Dvcqk = void(self.__Dvc__)
        Avvqk = void(self.__Avv__)
        Bvvqk = void(self.__Bvv__)
        Avvcqk = void(self.__Avvc__)
        Bvvcqk = void(self.__Bvvc__)
        Cvvqk = void(self.__Cvv__)
        Dvvqk = void(self.__Dvv__)
        #Pqk = void(self.__phase__)
        omega_T_inv = void(self.wTinv)
        Sq, c = self.sq, self.cq
        Pe_t = self.v0/self.D0
        _nu = self.nu
        @njit
        def ker (m, phi, i, t):
            A = nparray(Aqk)
            B = nparray(Bqk)
            C = nparray(Cqk)
            D = nparray(Dqk)
            Av = nparray(Avqk)
            Bv = nparray(Bvqk)
            Cv = nparray(Cvqk)
            Dv = nparray(Dvqk)
            Cvc = nparray(Cvcqk)
            Dvc = nparray(Dvcqk)
            Avv = nparray(Avvqk)
            Bvv = nparray(Bvvqk)
            Avvc = nparray(Avvcqk)
            Bvvc = nparray(Bvvcqk)
            Cvv = nparray(Cvvqk)
            Dvv = nparray(Dvvqk)
            #phase = nparray(Pqk)
            wTinv = nparray(omega_T_inv)
            lr = np.arange(-L,L+1)
            Ainner = np.sum(A*phi[:,L,L,None,None],axis=-3)
            Binner = np.sum(B*phi[:,L,None,:],axis=-3)
            Cinner = np.sum(C*phi[:,:,L,None],axis=-3)
            Dinner = np.sum(D*phi[:,:,:],axis=-3)
            if not Pe_t == 0.0:
                if L>=1:
                    Ainner += np.sum(Av*(phi[:,L-1,L,None,None]+phi[:,L+1,L,None,None]),axis=-3)
                    Binner += np.sum(Bv*(phi[:,L-1,None,:]+phi[:,L+1,None,:]),axis=-3)
                    # Cinner[q,k,l,ld]*phi[k,0,ld]
                    # = C[q,k,p,l,ld]*phi[p,l,0]*phi[k,0,ld]
                    # + Cv[q,k,p,l,ld]*phi[p,l-1,0]*phi[k,0,ld]
                    # + Cv*[q,k,p,-l,-ld]*phi[p,l+1,0]*phi[k,0,ld]
                    #Cinner[:,:,1:,:] += np.sum(Cv[:,:,:,1:,:]*phi[:,:-1,L,None],axis=-3) + np.sum(np.conjugate(Cv)[:,:,:,:0:-1,::-1]*phi[:,1:,L,None],axis=-3)
                    Cinner[:,:,1:-1,:] += np.sum(Cv[:,:,:,1:-1,:]*phi[:,:-2,L,None],axis=-3)
                    Cinner[:,:,1:-1,:] += np.sum(Cvc[:,:,:,1:-1,:]*phi[:,2:,L,None],axis=-3)
                    Dinner[:,:,1:-1,:] += np.sum(Dv[:,:,:,1:-1,:]*phi[:,:-2,:],axis=-3) + np.sum(Dvc[:,:,:,1:-1,:]*phi[:,2:,:],axis=-3)
                    A2inner = np.sum(Avv*phi[:,L,L,None,None],axis=-3)
                    A2innerc = np.sum(Avvc*phi[:,L,L,None,None],axis=-3)
                    B2inner = np.sum(Bvv*phi[:,L,None,:],axis=-3)
                    B2innerc = np.sum(Bvvc*phi[:,L,None,:],axis=-3)
                    C2inner = np.sum(Cvv*phi[:,:,L,None],axis=-3)
                    D2inner = np.sum(Dvv*phi[:,:,:],axis=-3)
            q_ = q[:,None,None]
            dq = np.diff(q)[:,None,None]
            m[:,:,:] = - ( \
                np.sum((q_[:-1]*phi[:-1,:,:]*Ainner[:,:-1,:,:] \
                      + q_[1:]*phi[1:,:,:]*Ainner[:,1:,:,:])/2 \
                      * dq, axis=1) \
                + np.sum((q_[:-1]*phi[:-1,:,L,None]*Binner[:,:-1,:,:] \
                      + q_[1:]*phi[1:,:,L,None]*Binner[:,1:,:,:])/2 \
                      * dq, axis=1) \
                + np.sum((q_[:-1]*phi[:-1,L,None,:]*Cinner[:,:-1,:,:] \
                      + q_[1:]*phi[1:,L,None,:]*Cinner[:,1:,:,:])/2 \
                      * dq, axis=1) \
                + np.sum((q_[:-1]*phi[:-1,L,L,None,None]*Dinner[:,:-1,:,:] \
                      + q_[1:]*phi[1:,L,L,None,None]*Dinner[:,1:,:,:])/2 \
                      * dq, axis=1))
            if (not Pe_t == 0.0) and L>=1:
                m[:,:,:] -= ( \
                    np.sum((q_[:-1]*(phi[:-1,L-1,None,:]+phi[:-1,L+1,None,:])*C2inner[:,:-1,:,:] \
                            + q_[1:]*(phi[1:,L-1,None,:]+phi[1:,L+1,None,:])*C2inner[:,1:,:,:])/2 \
                            * dq, axis=1) \
                    + np.sum((q_[:-1]*(phi[:-1,L-1,L,None,None]+phi[:-1,L+1,L,None,None])*D2inner[:,:-1,:,:] \
                            + q_[1:]*(phi[1:,L-1,L,None,None]+phi[1:,L+1,L,None,None])*D2inner[:,1:,:,:])/2 \
                            * dq, axis=1) \
                )
                m[:,1:-1,:] -= ( \
                    + np.sum((q_[:-1]*phi[:-1,:-2,:]*A2inner[:,:-1,1:-1,:] \
                            + q_[1:]*phi[1:,:-2,:]*A2inner[:,1:,1:-1,:])/2 \
                            * dq, axis=1) \
                )
                m[:,1:-1,:] -= ( \
                    + np.sum((q_[:-1]*phi[:-1,2:,:]*A2innerc[:,:-1,1:-1,:] \
                            + q_[1:]*phi[1:,2:,:]*A2innerc[:,1:,1:-1,:])/2 \
                            * dq, axis=1) \
                )
                m[:,1:-1,:] -= ( \
                    + np.sum((q_[:-1]*phi[:-1,:-2,L,None]*B2inner[:,:-1,1:-1,:] \
                            + q_[1:]*phi[1:,:-2,L,None]*B2inner[:,1:,1:-1,:])/2 \
                            * dq, axis=1) \
                )
                m[:,1:-1,:] -= ( \
                    + np.sum((q_[:-1]*phi[:-1,2:,L,None]*B2innerc[:,:-1,1:-1,:] \
                            + q_[1:]*phi[1:,2:,L,None]*B2innerc[:,1:,1:-1,:])/2 \
                            * dq, axis=1) \
                )
            for qi in range(M):
                m[qi,:,:] = np.dot(wTinv[qi],np.dot(m[qi],wTinv[qi]))
                if _nu == 0.0:
                    for l in lr:
                        for ld in lr:
                            if not (l-ld)%2: # even
                                m[qi,L+l,L+ld] = m[qi,L+l,L+ld].real + 0.j
                            else:
                                m[qi,L+l,L+ld] = m[qi,L+l,L+ld].imag*1j + 0.0
        return ker
