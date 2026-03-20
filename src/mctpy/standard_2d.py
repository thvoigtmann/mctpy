import numpy as np

from numba import njit
from .util import filon_integrate
from .__util__ import model_base, void, nparray

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



class simple_liquid_model_2d (model_base):
    """Simple liquid model, two-dimensional version.

    This implements the standard MCT memory kernel in two dimensions,
    with an integration scheme that takes care of the square-root
    singularity arising from the coordinate transform in the inner
    wave-number integral.
    """
    def __init__ (self, Sq, q, D0=1.0):
        model_base.__init__(self)
        self.rho = Sq.density()
        self.q = q
        self.sq, self.cq = Sq.Sq(q)
        self.M = q.shape[0]
        self.__init_vertices__ ()
        self.D0 = D0
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Bq (self):
        return self.sq/self.D0
    def __init_vertices__ (self):
        #self.__A__ = np.zeros((self.M,self.M))
        self.__A3__ = np.zeros((self.M,self.M,self.M))
        A3 = self.__A3__
        M = self.M
        c = self.cq
        S = self.sq
        q = self.q
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
                A3[qi,ki,p] = 2*q[qi]**2 * g0unsum(S[p]*c[p]**2,x,1,minval) \
                      + 4*q[qi]*q[ki] * g1unsum(S[p]*c[p]*(c[ki]-c[p]),x,1,minval) \
                      + 2*q[ki]**2 * g2unsum(S[p]*(c[ki]-c[p])**2,x,1,minval)
    def make_kernel (self):
        """Kernel factory, two-dimensional MCT for simple liquids.

        Returns a numba-jit kernel method that implements the integration
        of the memory kernel, taking care of the Jacobian in two wave-vector
        dimensions that contains a square-root singularity, treated with
        a Filon-Tuck-type weighted-trapezoidal integration method.
        """
        q = self.q
        #M = self.M
        pre = self.rho*self.sq/(8*np.pi**2)
        #Aqk = void(self.__A__)
        Aqk3 = void(self.__A3__)
        c = self.cq
        S = self.sq
        @njit
        def ker (m, phi, i, t):
            #A = nparray(Aqk)
            A3 = nparray(Aqk3)
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
            m[:] = -pre * np.sum((q[:-1]*phi[:-1]*S[:-1]*Ainner[:,:-1] + q[1:]*phi[1:]*S[1:]*Ainner[:,1:])/2 * np.diff(q), axis=-1)
        return ker

class tagged_particle_model_2d (model_base):
    def __init__ (self, base_model, cs, D0s=1.0):
        model_base.__init__(self)
        self.base = base_model
        self.q = base_model.q
        self.cs = cs.cq(self.q)
        self.M = self.q.shape[0]
        self.__init_vertices__()
        self.D0 = D0s
    def __len__ (self):
        return self.M
    def Wq (self):
        return self.q*self.q
    def Bq (self):
        return np.ones(self.M)/self.D0
    def __init_vertices__ (self):
        self.__A__ = np.zeros((self.M,self.M))
        self.__A3__ = np.zeros((self.M,self.M,self.M))
        #Aqk = void(self.__A__)
        Aqk3 = void(self.__A3__)
        q = self.q
        c = self.cs
        S = self.base.sq
        M = self.M
        #@njit
        #def calc_V (V, f):
        #    A = nparray(Aqk)
        #    for qi in range(M):
        #        for ki in range(M):
        #            if ki <= qi:
        #                pmin = q[qi] - q[ki]
        #            else:
        #                pmin = q[ki] - q[qi]
        #            pmax = q[qi] + q[ki]
        #            x = (q[qi]**2 + q[ki]**2 - q**2) / (2*q[qi]*q[ki])
        #            mask = (x>=-1) & (x<=1) & (q>=pmin)
        #            x = x[mask]
        #            p = np.arange(M)[mask]
        #            minval = -1.0
        #            if pmax > q[-1]:
        #                minval = x[-1]
        #            A[qi,ki] = 2*q[qi]**2 * g0(f[p]*S[p]*c[p]**2,x,1,minval) \
        #                  - 4*q[qi]*q[ki] * g1(f[p]*S[p]*c[p]**2,x,1,minval) \
        #                  + 2*q[ki]**2 * g2(f[p]*S[p]*c[p]**2,x,1,minval)
        #self.__calcV__ = calc_V
        A3 = self.__A3__
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
                A3[qi,ki,p] = 2*q[qi]**2 * g0unsum(S[p]*c[p]**2,x,1,minval) \
                      - 4*q[qi]*q[ki] * g1unsum(S[p]*c[p]**2,x,1,minval) \
                      + 2*q[ki]**2 * g2unsum(S[p]*c[p]**2,x,1,minval)
    def kernel_extra_args (self):
        return [self.base.phi]
    def make_kernel (self):
        q = self.q
        pre = self.base.rho/(4*np.pi**2)
        Aqk = void(self.__A__)
        Aqk3 = void(self.__A3__)
        self.__i__ = np.zeros(1,dtype=int)
        __i__ = void(self.__i__)
        #Vfunc = self.__calcV__
        @njit
        def ker (ms, phis, i, t, phi):
            A = nparray(Aqk)
            A3 = nparray(Aqk3)
            last_i = nparray(__i__)
            if last_i[0]==0 or not (i==last_i[0]):
                #Vfunc(A, phi[i])
                A[:,:] = np.sum(A3*phi[i],axis=-1)
                last_i[0] = i
            ms[:] = -pre * np.sum((q[:-1]*phis[:-1]*A[:,:-1] + q[1:]*phis[1:]*A[:,1:])/2 * np.diff(q), axis=-1)
        return ker
