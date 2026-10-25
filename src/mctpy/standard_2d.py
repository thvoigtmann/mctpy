import numpy as np

from numba import njit
from .util import filon_integrate
from .__util__ import model_base, void, nparray, _dq, modelPrefactors
from .__h5__ import LoadableModel

# try to perform the integrals well, going correctly to the boundaries
# with extrapolation if possible; the interval is [-1,1] usually,
# but the kernel method may adjust the -1 in order to stop wrong
# extrapolations occurring near the cutoff wave vector
# as it stands, the method now seems to be similar to what Caraglio et al
# show, perhaps even slightly better in terms of convergence with M

@njit
def g0(f,x,x0,x1):
    G0 = np.arcsin(x)
    G1 = -np.sqrt(1-x*x)
    G00,G01 = np.arcsin(x0),np.arcsin(x1)
    G10,G11 = -np.sqrt(1-x0*x0),-np.sqrt(1-x1*x1)
    b = (f[...,1:] - f[...,:-1]) / np.diff(x)
    a = f[...,:-1] - x[:-1]*b
    if len(x)>1:
        bd1 = a[...,0]*(G0[0]-G00) + b[...,0]*(G1[0]-G10)
        bd2 = a[...,-1]*(G01-G0[-1]) + b[...,-1]*(G11-G1[-1])
    else:
        return f[...,0]*(G01-G00)
    return np.sum(a*np.diff(G0) + b*np.diff(G1),axis=-1) + bd1 + bd2
@njit
def g1(f,x,x0,x1):
    G1 = -np.sqrt(1-x*x)
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G10,G11 = -np.sqrt(1-x0*x0),-np.sqrt(1-x1*x1)
    G20,G21 = 0.5*np.arcsin(x0) - 0.5*x0*np.sqrt(1-x0*x0), \
              0.5*np.arcsin(x1) - 0.5*x1*np.sqrt(1-x1*x1)
    b = (f[...,1:] - f[...,:-1]) / np.diff(x)
    a = f[...,:-1] - x[:-1]*b
    if len(x)>1:
        bd1 = a[...,0]*(G1[0]-G10) + b[...,0]*(G2[0]-G20)
        bd2 = a[...,-1]*(G11-G1[-1]) + b[...,-1]*(G21-G2[-1])
    else:
        return f[...,0]*(G11-G10)
    return np.sum(a*np.diff(G1) + b*np.diff(G2),axis=-1) + bd1 + bd2
@njit
def g2(f,x,x0,x1):
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G3 = (-1./3)*np.sqrt(1-x*x) * (2+x*x)
    G20,G21 = 0.5*np.arcsin(x0) - 0.5*x0*np.sqrt(1-x0*x0), \
              0.5*np.arcsin(x1) - 0.5*x1*np.sqrt(1-x1*x1)
    G30,G31 = (-1./3)*np.sqrt(1-x0*x0) * (2+x0*x0), \
              (-1./3)*np.sqrt(1-x1*x1) * (2+x1*x1)
    b = (f[...,1:] - f[...,:-1]) / np.diff(x)
    a = f[...,:-1] - x[:-1]*b
    if len(x)>1:
        bd1 = a[...,0]*(G2[0]-G20) + b[...,0]*(G3[0]-G30)
        bd2 = a[...,-1]*(G21-G2[-1]) + b[...,-1]*(G31-G3[-1])
    else:
        return f[...,0]*(G21-G20)
    return np.sum(a*np.diff(G2) + b*np.diff(G3),axis=-1) + bd1 + bd2

@njit
def g0unsum(f,x,x0,x1):
    G0 = np.arcsin(x)
    G1 = -np.sqrt(1-x*x)
    G0[0], G0[-1] = np.arcsin(x0), np.arcsin(x1)
    G1[0], G1[-1] = -np.sqrt(1-x0*x0), -np.sqrt(1-x1*x1)
    if len(x)==1:
        return np.array([f[...,0]*(G0[-1]-G0[0])])
    res = np.zeros_like(f)
    res[...,:-1] = (f[...,:-1] + x[:-1]*f[...,:-1]/np.diff(x)) * np.diff(G0) \
                 - (f[...,:-1]/np.diff(x) * np.diff(G1))
    res[...,1:] += (-x[:-1]*f[...,1:]/np.diff(x)) * np.diff(G0) \
                 + f[...,1:]/np.diff(x) * np.diff(G1)
    return res
@njit
def g1unsum(f,x,x0,x1):
    G1 = -np.sqrt(1-x*x)
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G1[0], G1[-1] = -np.sqrt(1-x0*x0), -np.sqrt(1-x1*x1)
    G2[0], G2[-1] = 0.5*np.arcsin(x0) - 0.5*x0*np.sqrt(1-x0*x0), \
                    0.5*np.arcsin(x1) - 0.5*x1*np.sqrt(1-x1*x1)
    if len(x)==1:
        return np.array([f[...,0]*(G1[-1]-G1[0])])
    res = np.zeros_like(f)
    res[...,:-1] = (f[...,:-1] + x[:-1]*f[...,:-1]/np.diff(x)) * np.diff(G1) \
                 - (f[...,:-1]/np.diff(x) * np.diff(G2))
    res[...,1:] += (-x[:-1]*f[...,1:]/np.diff(x)) * np.diff(G1) \
                 + f[...,1:]/np.diff(x) * np.diff(G2)
    return res
@njit
def g2unsum(f,x,x0,x1):
    G2 = 0.5*np.arcsin(x) - 0.5*x*np.sqrt(1-x*x)
    G3 = (-1./3)*np.sqrt(1-x*x) * (2+x*x)
    G2[0], G2[-1] = 0.5*np.arcsin(x0) - 0.5*x0*np.sqrt(1-x0*x0), \
                    0.5*np.arcsin(x1) - 0.5*x1*np.sqrt(1-x1*x1)
    G3[0],G3[-1] = (-1./3)*np.sqrt(1-x0*x0) * (2+x0*x0), \
                   (-1./3)*np.sqrt(1-x1*x1) * (2+x1*x1)
    if len(x)==1:
        return np.array([f[...,0]*(G2[-1]-G2[0])])
    res = np.zeros_like(f)
    res[...,:-1] = (f[...,:-1] + x[:-1]*f[...,:-1]/np.diff(x)) * np.diff(G2) \
                 - (f[...,:-1]/np.diff(x) * np.diff(G3))
    res[...,1:] += (-x[:-1]*f[...,1:]/np.diff(x)) * np.diff(G2) \
                 + f[...,1:]/np.diff(x) * np.diff(G3)
    return res



@njit
def simple_liquid_model_2d_kernel (m, phi, i, t, A3, S, q):
    #for qi in range(M):
    #    for ki in range(M):
    #        if ki <= qi:
    #            pmin = q[qi] - q[ki]
    #        else:
    #            pmin = q[ki] - q[qi]
    #        pmax = q[qi] + q[ki]
    #        x = (q[qi]**2 + q[ki]**2 - q**2) / (2*q[qi]*q[ki])
    #        mask = (x>=-1) & (x<=1) & (q>=pmin)
    #        x = x[mask]
    #        p = np.arange(M)[mask]
    #        minval = -1.0
    #        if pmax > q[-1]:
    #            minval = x[-1]
    #        A[qi,ki] = 2*q[qi]**2 * g0(phi[p]*S[p]*c[p]**2,x,1,minval) \
    #              + 4*q[qi]*q[ki] * g1(phi[p]*S[p]*c[p]*(c[ki]-c[p]),x,1,minval) \
    #              + 2*q[ki]**2 * g2(phi[p]*S[p]*(c[ki]-c[p])**2,x,1,minval)
            #assert(not np.isnan(A[qi,ki]))
    #m[:] = -pre * np.sum((q[:-1]*phi[:-1]*S[:-1]*A[:,:-1] + q[1:]*phi[1:]*S[1:]*A[:,1:])/2 * np.diff(q), axis=-1)
    Ainner = np.sum(A3*phi,axis=-1)
    m[:] = -np.sum((q[:-1]*phi[:-1]*S[:-1]*Ainner[:,:-1] + q[1:]*phi[1:]*S[1:]*Ainner[:,1:])/2 * np.diff(q), axis=-1)

class simple_liquid_model_2d (model_base, LoadableModel,
                              kernel=simple_liquid_model_2d_kernel,
                              attrs=['rho','D0'], fields=['q','sq','cq'],
                              objects=['Sq'], resetable=['q','Sq','D0','vth'],
                              safe_reset=['D0','vth']):
    """Simple liquid model, two-dimensional version.

    This implements the standard MCT memory kernel in two dimensions,
    with an integration scheme that takes care of the square-root
    singularity arising from the coordinate transform in the inner
    wave-number integral. For this, the integrals are evaluated using
    a Filon-Tuck-type weighted-trapezoidal integration method.
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
    def spatial_dimension (self):
        return 2
    def prefactors (self, motion_type=None):
        pre = { 'phi0': 1.0, 'phi0d': 0.0, 'Wq': self.q*self.q }
        if motion_type == None:
            return modelPrefactors(pre)
        elif motion_type == 'brownian':
            return modelPrefactors(pre | { 'Bq': self.sq/self.D0 })
        raise NotImplementedError
    def dq (self):
        return _dq(self.q)
    def __init_vertices__ (self):
        q = self.q
        self.rho = self.Sq.density()
        self.sq, self.cq = self.Sq.Sq(q)
        self.M = q.shape[0]
        self.__A3__ = np.zeros((self.M,self.M,self.M))
        A3 = self.__A3__
        M = self.M
        c = self.cq
        S = self.sq
        pre = self.rho*self.sq/(8*np.pi**2)
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
                A3[qi,ki,p] = (2*q[qi]**2 * g0unsum(S[p]*c[p]**2,x,1,minval) \
                      + 4*q[qi]*q[ki] * g1unsum(S[p]*c[p]*(c[ki]-c[p]),x,1,minval) \
                      + 2*q[ki]**2 * g2unsum(S[p]*(c[ki]-c[p])**2,x,1,minval))\
                      * pre[qi]
    def kernel_extra_args (self):
        return [self.__A3__,self.sq,self.q]

@njit
def tagged_particle_model_2d_calc_V (i, A, A3, phi):
    A[:,:] = np.sum(A3*phi[i],axis=-1)
@njit
def tagged_particle_model_2d_kernel (ms, phis, i, t, phi, A, q):
    ms[:] = -np.sum((q[:-1]*phis[:-1]*A[:,:-1] + q[1:]*phis[1:]*A[:,1:])/2 * np.diff(q), axis=-1)
class tagged_particle_model_2d (model_base, LoadableModel,
                                kernel=tagged_particle_model_2d_kernel,
                                setV=tagged_particle_model_2d_calc_V,
                                fields=['cs'], attrs=['D0'],
                                objects=['base', 'Sq'],
                                resetable=['base','Sq', 'D0'],
                                safe_reset=['D0']):
    def __init__ (self, base_model, Sq, D0s=1.0):
        super().__init__()
        self.base = base_model
        self.q = base_model.q
        self.Sq = Sq
        self.cs = None
        self.D0 = D0s
        self.__init_vertices__ ()
    def reinit (self):
        self.__init_vertices__ ()
    def get_bases (self):
        return [self.base]
    def __len__ (self):
        return self.M
    def prefactors (self, motion_type=None):
        pre = { 'phi0': 1.0, 'phi0d': 0.0, 'Wq': self.q*self.q }
        if motion_type == None:
            return modelPrefactors(pre)
        elif motion_type == 'brownian':
            return modelPrefactors(pre | { 'Bq': 1./self.D0 })
        raise NotImplementedError
    def __init_vertices__ (self):
        self.cs = self.Sq.cq(self.q)
        self.M = self.q.shape[0]
        self.__A__ = np.zeros((self.M,self.M))
        self.__A3__ = np.zeros((self.M,self.M,self.M))
        q = self.q
        c = self.cs
        S = self.base.sq
        M = self.M
        A3 = self.__A3__
        pre = self.base.rho/(4*np.pi**2)
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
                A3[qi,ki,p] = (2*q[qi]**2 * g0unsum(S[p]*c[p]**2,x,1,minval) \
                      - 4*q[qi]*q[ki] * g1unsum(S[p]*c[p]**2,x,1,minval) \
                      + 2*q[ki]**2 * g2unsum(S[p]*c[p]**2,x,1,minval)) \
                      * pre
    def kernel_extra_args (self):
        return [self.base.phi, self.__A__, self.q]
    def set_V_args (self):
        return (self.__A__, self.__A3__, self.base.phi)
    def mrenorm (self, m):
        return m/self.q**2
