import numpy as np
import scipy

from numba import njit
from .__util__ import model_base, void, nparray, _dq
from .__h5__ import LoadableModel

## mixture_model kernels
@njit
def mixture_model_kernel (m, phi, i, t, a1, a2, a3, b1, b2, b3, sr, q, dq):
    pre = 1./(32.*np.pi**2) * dq[0]**2
    M = q.shape[0]
    S = b3.shape[-1]
    for qi in range(M):
        for a in range(S):
            for b in range(S):
                mq = 0.
                for n in range(3):
                    pi, ki = qi, 0
                    zqk1 = b1[n,qi,pi] * phi[pi,a,b]
                    zqk2 = 0.0
                    for g in range(S):
                        zqk2 += b2[n,qi,pi,b,g] *  phi[pi,a,g] * sr[g]
                    for g in range(S):
                        mq += a2[n,ki,a,g] * zqk2 * phi[ki,b,g] * sr[g]
                        for d in range(S):
                            mq += a3[n,ki,a,g,b,d] * zqk1 \
                                  * phi[ki,g,d] * sr[g] * sr[d]
                    for ki in range(1, M):
                        if ki <= qi:
                            pi = qi - ki
                            sgn = 1
                        else:
                            pi = ki - qi - 1
                            sgn = -1
                        zqk1 += sgn*b1[n,qi,pi] * phi[pi,a,b]
                        for g in range(S):
                            zqk2 += sgn*b2[n,qi,pi,b,g]*phi[pi,a,g]*sr[g]
                        pi = qi + ki
                        if pi < M:
                            zqk1 += b1[n,qi,pi] * phi[pi,a,b]
                            for g in range(S):
                                zqk2 += b2[n,qi,pi,b,g]*phi[pi,a,g]*sr[g]
                        for g in range(S):
                            mq += a2[n,ki,a,g]*zqk2*phi[ki,b,g]*sr[g]
                            for d in range(S):
                                mq += a3[n,ki,a,g,b,d] * zqk1 \
                                      * phi[ki,g,d] * sr[g] * sr[d]
                mq *= 2.0
                m[qi,a,b] = mq * pre / q[qi]
@njit
def mixture_model_dm(m, f, h, a1, a2, a3, b1, b2, b3, sr, q, dq):
    """Return method to evaluate stability matrix for mixtures.

    Notes
    -----
    The implementation uses the appropriate variant of Bengtzelius' trick.
    The stability matrix is evaluated on-the-fly, and not pre-cached in
    memory.
    """
    pre = 1./(32.*np.pi**2) * dq[0]**2
    M = q.shape[0]
    S = b3.shape[-1]
    for qi in range(M):
        for a in range(S):
            for b in range(S):
                res = 0.0
                for n in range(3):
                    pi, ki = qi, 0
                    zqk1 = b1[n,qi,pi] * f[pi,a,b]
                    zqk2, zqk3, zqk4 = 0.0, 0.0, 0.0
                    for g in range(S):
                        zqk2 += b2[n,qi,pi,b,g] * f[pi,a,g] * sr[g]
                        zqk3 += b2[n,qi,pi,a,g] * f[pi,b,g] * sr[g]
                        for d in range(S):
                            zqk4 += b3[n,qi,pi,a,g,b,d] \
                                * f[pi,g,d] * sr[g]*sr[d]
                    res += a1[n,ki] * zqk4 * h[ki,a,b]
                    for g in range(S):
                        res += a2[n,ki,a,g] * zqk2 * h[ki,b,g] * sr[g]
                        res += a2[n,ki,b,g] * zqk3 * h[ki,a,g] * sr[g]
                        for d in range(S):
                            res += a3[n,ki,a,g,b,d] * zqk1 \
                                * h[ki,g,d] * sr[g]*sr[d]
                    for ki in range(1, M):
                        if ki <= qi:
                            pi = qi - ki
                            sgn = 1
                        else:
                            pi = ki - qi - 1
                            sgn = -1
                        zqk1 += sgn*b1[n,qi,pi] * f[pi,a,b]
                        for g in range(S):
                            zqk2 += sgn*b2[n,qi,pi,b,g]*f[pi,a,g]*sr[g]
                            zqk3 += sgn*b2[n,qi,pi,a,g]*f[pi,b,g]*sr[g]
                            for d in range(S):
                                zqk4 += sgn*b3[n,qi,pi,a,g,b,d] \
                                    * f[pi,g,d] * sr[g]*sr[d]
                        pi = qi + ki
                        if pi < M:
                            zqk1 += b1[n,qi,pi] * f[pi,a,b]
                            for g in range(S):
                                zqk2 += b2[n,qi,pi,b,g]*f[pi,a,g]*sr[g]
                                zqk3 += b2[n,qi,pi,a,g]*f[pi,b,g]*sr[g]
                                for d in range(S):
                                    zqk4 += b3[n,qi,pi,a,g,b,d] \
                                        * f[pi,g,d] * sr[g]*sr[d]
                        res += a1[n,ki] * zqk4 * h[ki,a,b]
                        for g in range(S):
                            res += a2[n,ki,a,g]*zqk2*h[ki,b,g]*sr[g]
                            res += a2[n,ki,b,g]*zqk3*h[ki,a,g]*sr[g]
                            for d in range(S):
                                res += a3[n,ki,a,g,b,d] * zqk1 \
                                    * h[ki,g,d] * sr[g]*sr[d]
                res *= 2.0
                m[qi,a,b] = res * pre / q[qi]**3
@njit
def mixture_model_dmhat(m, f, h, a1, a2, a3, b1, b2, b3, sr, q, dq):
    """Return method to evaluate left-multiplication to stability matrix
    for mixtures.

    Notes
    -----
    The implementation uses the appropriate variant of Bengtzelius' trick.
    The stability matrix is evaluated on-the-fly, and not pre-cached in
    memory.
    """
    pre = 1./(32.*np.pi**2) * dq[0]**2
    M = q.shape[0]
    S = b3.shape[-1]
    zqp1 = np.zeros((S,S))
    zqp2 = np.zeros(S)
    zqp3 = np.zeros(S)
    for ki in range(M):
        for a in range(S):
            for b in range(S):
                res = 0.0
                for n in range(3):
                    qi, pi = 0, ki
                    zqp4 = 0.0
                    for g in range(S):
                        zqp2[g],zqp3[g] = 0.0, 0.0
                        for d in range(S):
                            zqp1[g,d] = a1[n,pi] * f[pi,g,d]
                            zqp2[g] += a2[n,pi,a,d] * f[pi,d,g] * sr[d]
                            zqp3[g] += a2[n,pi,b,d] * f[pi,g,d] * sr[d]
                            zqp4+=a3[n,pi,a,g,b,d]*f[pi,g,d]*sr[g]*sr[d]
                    tmp = b1[n,qi,ki] * h[qi,a,b] * zqp4
                    for g in range(S):
                        tmp += b2[n,qi,ki,g,b] * h[qi,a,g] \
                            * zqp2[g] * sr[b]
                        tmp += b2[n,qi,ki,g,a] * h[qi,g,b] \
                            * zqp3[g] * sr[a]
                        for d in range(S):
                            tmp += b3[n,qi,ki,g,a,d,b] \
                                * zqp1[g,d] * h[qi,g,d] * sr[a]*sr[b]
                    res += tmp / q[qi]**3
                    for qi in range(1,M):
                        if qi <= ki:
                            pi = ki - qi
                            sgn = 1
                        else:
                            pi = qi - ki - 1
                            sgn = -1
                        for g in range(S):
                            for d in range(S):
                                zqp1[g,d] += sgn*a1[n,pi] * f[pi,g,d]
                                zqp2[g] += sgn*a2[n,pi,a,d]*f[pi,d,g]*sr[d]
                                zqp3[g] += sgn*a2[n,pi,b,d]*f[pi,g,d]*sr[d]
                                zqp4 += sgn*a3[n,pi,a,g,b,d]*f[pi,g,d]*sr[g]*sr[d]
                        pi = qi + ki
                        if pi < M:
                            for g in range(S):
                                for d in range(S):
                                    zqp1[g,d] += a1[n,pi] * f[pi,g,d]
                                    zqp2[g] += a2[n,pi,a,d]*f[pi,d,g]*sr[d]
                                    zqp3[g] += a2[n,pi,b,d]*f[pi,g,d]*sr[d]
                                    zqp4 += a3[n,pi,a,g,b,d]*f[pi,g,d]*sr[g]*sr[d]
                        tmp = b1[n,qi,ki] * h[qi,a,b] * zqp4
                        for g in range(S):
                            tmp += b2[n,qi,ki,g,b] * h[qi,a,g] \
                                * zqp2[g] * sr[b]
                            tmp += b2[n,qi,ki,g,a] * h[qi,g,b] \
                                * zqp3[g] * sr[a]
                            for d in range(S):
                                tmp += b3[n,qi,ki,g,a,d,b] \
                                    * zqp1[g,d] * h[qi,g,d] *sr[a]*sr[b]
                        res += tmp / q[qi]**3
                m[ki,a,b] = 2.0 * res * pre
def mixture_model_dm2 (m, f, h, a1, a2, a3, b1, b2, b3, sr, q, dq):
    """Return method to evaluate second variation of memory kernel.

    Since the memory kernel is bilinear, this is the same as the kernel,
    but we include the division by q**2 here."""
    mixture_model_kernel (m, h, 0, 0.0, a1, a2, a3, b1, b2, b3, sr, q, dq)
    m[:] = m/q[:,None,None]**2

class mixture_model (model_base, LoadableModel,
                     kernel=mixture_model_kernel,
                     dm=mixture_model_dm, dmhat=mixture_model_dmhat,
                     dm2=mixture_model_dm2,
                     fields=['q','sq','cq','densities','D0'],
                     objects=['Sq'], resetable=['q','Sq','D0'],
                     safe_reset=['D0']):
    """MCT model for simple-liquid mixtures.

    Parameters
    ----------
    Sq : structure_factor
        Should implement a structure factor; determines the number of
        species and concentrations used in the model.
    q : array_like
        Wave number grid, should follow the convention.
    D0 : float or array_like, default: 1.0
        The short-time diffusion coefficients; either an array whose
        length matches the number of species defined in the structure
        factor, or a scalar value that will be applied to all species.
    """
    def __init__ (self, Sq=None, q=None, D0=1.0):
        super().__init__()
        self.q = q
        self.D0 = D0
        self.Sq = Sq
        self.sq, self.cq = None, None
        self.densities = None
        self.__init_vertices__()
    def reinit (self):
        self.__init_vertices__ ()
    def __len__ (self):
        return self.M
    def matrix_dimension (self):
        return self.S
    def scalar (self):
        return False
    def phi0 (self):
        return self.sq
    def diag (self, array_or_value):
        if isinstance(array_or_value, (list, np.ndarray)):
            return np.diag(array_or_value)
        else:
            return np.diag(np.ones(self.S)*array_or_value)
    def Bq (self):
        return np.ones((self.M,self.S,self.S))*self.diag(1/self.D0)
    def Bqinv (self):
        return np.ones((self.M,self.S,self.S))*self.diag(self.D0)
    def Wq (self):
        return (self.q**2)[:,None,None] * np.linalg.inv(self.sq)
    def WqSq (self):
        return (self.q**2)[:,None,None] * np.diag(np.ones(self.S))
    def dq (self):
        return _dq(self.q)
    def __init_vertices__ (self):
        self.densities = self.Sq.densities
        self.rho = np.sum(self.densities) # used by tagged_q0
        self.sq, self.cq = self.Sq.Sq(self.q)
        q = self.q
        self.M = self.q.shape[0]
        self.S = self.densities.shape[0]
        self.a1 = np.zeros((3,self.M))
        self.a2 = np.zeros((3,self.M,self.S,self.S))
        self.a3 = np.zeros((3,self.M,self.S,self.S,self.S,self.S))
        self.b1 = np.zeros((3,self.M,self.M))
        self.b2 = np.zeros((3,self.M,self.M,self.S,self.S))
        self.b3 = np.zeros((3,self.M,self.M,self.S,self.S,self.S,self.S))
        self.a1[0] = q**5
        self.a1[1] = - 2 * q**3
        self.a1[2] =  q
        q = self.q[:,None,None]
        qcq = q * self.cq
        self.a2[0] = qcq
        self.a2[1] = - q**4 * qcq
        self.a2[2] = 2 * q**2 * qcq
        q = self.q[:,None,None,None,None]
        qcqcq = qcq[...,:,:,None,None]*self.cq[...,None,None,:,:]
        self.a3[0] = qcqcq * q**4
        self.a3[1] = qcqcq * 2 * q**2
        self.a3[2] = qcqcq
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        pq = p_/q_**2
        self.b1[0] = pq
        self.b1[1] = pq * (q_**2 - p_**2)
        self.b1[2] = pq * (q_**2 - p_**2)**2
        q_, p_ = q_[...,None,None], p_[...,None,None]
        pcp = pq[:,:,None,None] * self.cq[None,:,:,:]
        self.b2[0] = (q_**4 - p_**4) * pcp
        self.b2[1] = pcp
        self.b2[2] = p_**2 * pcp
        q_, p_ = q_[...,None,None], p_[...,None,None]
        pcpcp = pcp[...,:,:,None,None]*self.cq[...,None,None,:,:]
        self.b3[0] = pcpcp
        self.b3[1] = (q_**2 + p_**2) * pcpcp
        self.b3[2] = (q_**2 + p_**2)**2 * pcpcp
        self.sqrtrho = np.sqrt(self.densities)
    def kernel_extra_args (self):
        return [self.a1, self.a2, self.a3, self.b1, self.b2, self.b3,
                self.sqrtrho, self.q, self.dq()]
    def kernel_extra_args_dm (self):
        return [self.a1, self.a2, self.a3, self.b1, self.b2, self.b3,
                self.sqrtrho, self.q, self.dq()]
    def mrenorm (self, m):
        return m/self.q[:,None,None]**2
    def shear_modulus (self, phi, t):
        r"""Return shear modulus / shear stress, given a solution.

        Parameters
        ----------
        phi : array_like
            Correlator or nonergodicity-parameter values, needs to be
            in the form matching the current model.
        t : array_like
            Time points for the integral (not needed).

        Returns
        -------
        G : array_like
            The shear modulus corresponding to the nonergodicity parameter
            or the time-dependent shear modulus corresponding to the given
            time-dependent solution.
        """

        # phi can be shape (T,M,S,S) or (M,S,S)
        rho_total = np.sum(self.densities)
        # dcq: sqrt(rho_a rho_b) c'_{ab}(q), shape (M,S,S)
        rho_ab = np.sqrt(np.outer(self.densities,self.densities))
        dcq = rho_ab * self.Sq.dcq_dq(self.q)
        # C-style, "by hand":
        #inner = np.zeros(phi[...,0,0].shape)
        #for a in range(S):
        #    for b in range(S):
        #        for g in range(S):
        #            for d in range(S):
        #                inner += dcq[:,a,b]*dcq[:,g,d]*phi[...,a,g]*phi[...,b,d]
        # or, the wonderful numpy.einsum at work:
        inner = np.einsum('...ab,...gd,...ag,...bd',dcq,dcq,phi,phi)
        return scipy.integrate.trapezoid(self.dq() *
            self.q**4 * inner) / (60*np.pi**2)/rho_total
    def longitudinal_modulus (self, phi, t):
        r"""Return the longitudinal modulus, given a solution.

        Parameters
        ----------
        phi : array_like
            Correlator or nonergodicity-parameter values, needs to be in
            the form matching the surrent model. Needs to have shape
            `(...,M,S,S)`.
        t : array_like
            Time points for the integral (not needed).

        Returns
        -------
        M : array_like
            The longitudinal modulus corresponding to the input solution.
        """
        # phi can be shape (T,M,S,S) or (M,S,S)
        rho_total = np.sum(self.densities)
        # dcq: sqrt(rho_a rho_b) c'_{ab}(q), shape (M,S,S)
        rho_ab = np.sqrt(np.outer(self.densities,self.densities))
        dcq = rho_ab * self.Sq.dcq_dq(self.q)
        cq = rho_ab * self.Sq.cq(self.q)
        summation = '...ab,...gd,...ag,...bd'
        inner = np.einsum(summation,cq,cq,phi,phi) \
          + (2./3)*self.q*np.einsum(summation,dcq,cq,phi,phi) \
          + 0.2*self.q**2*np.einsum(summation,dcq,dcq,phi,phi)
        return scipy.integrate.trapezoid(self.dq() *
            self.q**2 * inner) / (4*np.pi**2)/rho_total
    def cc_modulus (self, phi, t):
        rho = np.sqrt(np.outer(self.densities,self.densities))
        cq = self.Sq.cq(self.q)
        sum1 = 'gd,...ag,...bd,...ab,...gd->...ab'
        sum2 = 'gd,...ag,...bd,...gb,...ad->...ab'
        inner = np.einsum(sum1,rho,cq,cq,phi,phi) - np.einsum(sum2,rho,cq,cq,phi,phi)
        qw = self.dq() * self.q**4
        return scipy.integrate.trapezoid(
                qw[...,None,None] * inner, axis=-3) / (6.*np.pi**2)


@njit
def mixture_tagged_calc_V (i, V, phi, a, b):
    f = phi[i]
    M = V.shape[0]
    S = b.shape[-1]
    for qi in range(M):
        for ki in range(M):
            V[qi,ki] = 0.
        for n in range(3):
            pi, ki = qi, 0
            zqk = 0.
            for alpha in range(S):
                for beta in range(S):
                    zqk += b[n,qi,pi,alpha,beta] * f[pi,alpha,beta]
            V[qi,ki] += a[n,ki] * zqk
            for ki in range(1,M):
                if ki <= qi:
                    pi = qi - ki
                    for alpha in range(S):
                        for beta in range(S):
                            zqk += b[n,qi,pi,alpha,beta] * f[pi,alpha,beta]
                else:
                    pi = ki - qi - 1
                    for alpha in range(S):
                        for beta in range(S):
                            zqk -= b[n,qi,pi,alpha,beta] * f[pi,alpha,beta]
                pi = qi + ki
                if pi < M:
                    for alpha in range(S):
                        for beta in range(S):
                            zqk += b[n,qi,pi,alpha,beta] * f[pi,alpha,beta]
                V[qi,ki] += a[n,ki] * zqk
@njit
def mixture_tagged_kernel (ms, phis, i, t, V, dq):
    ms[:] = dq**2 * np.dot(V,phis)

class mixture_tagged_particle (model_base, LoadableModel,
                               kernel=mixture_tagged_kernel,
                               setV=mixture_tagged_calc_V,
                               attrs=['D0'], fields=['cs'],
                               objects=['base','Sq']):
    def __init__ (self, base_model, Sq, D0s=1.0):
        super().__init__()
        self.D0 = D0s
        self.Sq = Sq
        self.base = base_model
        self.__init_vertices__()
    def reinit (self):
        self.__init_vertices__()
    def get_bases (self):
        return [self.base]
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Bq (self):
        return np.ones(self.M)/self.D0
    def __init_vertices__ (self):
        self.q = self.base.q
        self.cs = self.Sq.cq(self.q)
        self.M = self.q.shape[0]
        self.S = self.base.S
        pre = (1./(4.*np.pi))**2.
        rho = self.base.Sq.densities
        q = self.q
        self.a = np.zeros((3,self.M))
        self.b = np.zeros((3,self.M,self.M,self.S,self.S))
        self.a[0] = q**4 * q
        self.a[1] = - 2 * q**2 * q
        self.a[2] = q
        q_, p_ = np.meshgrid(q,q,indexing='ij')
        q_, p_ = q_[:,:,None,None], p_[:,:,None,None]
        pq = p_/q_**3 * pre
        qsq_psq = q_**2 + p_**2
        rho_c = np.sqrt(rho) * self.cs
        self.rho_csq = rho_c[:,:,None] * rho_c[:,None,:]
        cacb = self.rho_csq[:,None,:,:]
        self.b[0] = pq * cacb
        self.b[1] = pq * qsq_psq * cacb
        self.b[2] = pq * qsq_psq**2 * cacb
        self.__Vqk__ = np.zeros((self.M,self.M))
    def kernel_extra_args (self):
        return (self.__Vqk__, _dq(self.base.q))
    def set_V_args (self):
        return (self.__Vqk__, self.base.phi, self.a, self.b)
    def mrenorm (self, m):
        return m/self.q**2
