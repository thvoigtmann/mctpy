import numpy as np

from numba import njit
from .__util__ import model_base, void, nparray

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

class mixture_model (model_base):
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
    def __init__ (self, Sq, q, D0=1.0):
        model_base.__init__(self)
        self.rho = Sq.density()
        self.q = q
        self.Sq = Sq
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        self.S = Sq.densities.shape[0]
        self.__init_vertices__ ()
        if isinstance(D0, (list, np.ndarray)):
          self.D0 = np.array(D0)
        else:
          self.D0 = np.ones(self.S)*D0
    def __len__ (self):
        return self.M
    def matrix_dimension (self):
        return self.S
    def scalar (self):
        return False
    def phi0 (self):
        return self.sq
    def Bq (self):
        return np.ones((self.M,self.S,self.S))*np.diag(1/self.D0)
    def Bqinv (self):
        return np.ones((self.M,self.S,self.S))*np.diag(self.D0)
    def Wq (self):
        return (self.q**2)[:,None,None] * np.linalg.inv(self.sq)
    def WqSq (self):
        return (self.q**2)[:,None,None] * np.diag(np.ones(self.S))
    def dq (self):
        return _dq(self.q)
    def __init_vertices__ (self):
        self.pre = 1./(32.*np.pi**2) * self.dq()[0]**2
        q = self.q
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
        self.sqrtrho = np.sqrt(self.Sq.densities)
    def make_kernel (self):
        """Return kernel-evaluation function, using Bengtzelius' trick."""
        a1, a2, a3 = self.a1, self.a2, self.a3
        b1, b2, b3 = self.b1, self.b2, self.b3
        M, S = self.M, self.S
        sr = self.sqrtrho
        q = self.q
        dq = self.dq()
        pre = self.pre
        @njit
        def ker (m, phi, i, t):
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
        return ker
    def make_dm (self):
        """Return method to evaluate stability matrix.

        Notes
        -----
        The implementation uses the appropriate variant of Bengtzelius' trick.
        The stability matrix is evaluated on-the-fly, and not pre-cached in
        memory.
        """
        a1, a2, a3 = self.a1, self.a2, self.a3
        b1, b2, b3 = self.b1, self.b2, self.b3
        M, S = self.M, self.S
        sr = self.sqrtrho
        q = self.q
        dq = self.dq()
        pre = self.pre
        @njit
        def dm(m, f, h):
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
        return dm

    def make_dmhat (self):
        """Return method to evaluate left-multiplication to stability matrix.

        Notes
        -----
        The implementation uses the appropriate variant of Bengtzelius' trick.
        The stability matrix is evaluated on-the-fly, and not pre-cached in
        memory.
        """
        a1, a2, a3 = self.a1, self.a2, self.a3
        b1, b2, b3 = self.b1, self.b2, self.b3
        M, S = self.M, self.S
        sr = self.sqrtrho
        q = self.q
        dq = self.dq()
        pre = self.pre
        @njit
        def dmhat(m, f, h):
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
        return dmhat

    def make_dm2 (self):
        """Return method to evaluate second variation of memory kernel.

        Since the memory kernel is bilinear, this is the same as the kernel,
        but we include the division by q**2 here."""
        def dm2 (m, f, h):
            ker = self.get_kernel()
            ker (m, h, 0, 0.)
            m[:] = m/self.q[:,None,None]**2
        return dm2
