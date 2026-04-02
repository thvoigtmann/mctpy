import numpy as np
from numba import njit
from numba.extending import overload

import h5py

from .solver import correlator
from .util import regula_falsi, exponents
from .__util__ import model_base, np_isclose_all

# stencil: the non-central part of the cubic-lattice laplacian
# used by the SBR solver

def stencil_impl(x):
    raise NotImplementedError

@overload(stencil_impl)
def stencil_impl_overload(x):
    if x.ndim == 1:
        def f1d(x):
            return np.roll(x,1) + np.roll(x,-1)
        return f1d
    if x.ndim == 2:
        def f2d(x):
            res = np.zeros_like(x)
            res[1:-1] += x[2:] + x[:-2]
            res[0] += x[1] + x[-1]
            res[-1] += x[0] + x[-2]
            res[:,1:-1] += x[:,2:] + x[:,:-2]
            res[:,0] += x[:,1] + x[:,-1]
            res[:,-1] += x[:,0] + x[:,-2]
            return res
        return f2d
    if x.ndim == 3:
        def f3d(x):
            res = np.zeros_like(x)
            res[1:-1] += x[2:] + x[:-2]
            res[0] += x[1] + x[-1]
            res[-1] += x[0] + x[-2]
            res[:,1:-1] += x[:,2:] + x[:,:-2]
            res[:,0] += x[:,1] + x[:,-1]
            res[:,-1] += x[:,0] + x[:,-2]
            res[:,:,1:-1] += x[:,:,2:] + x[:,:,:-2]
            res[:,:,0] += x[:,:,1] + x[:,:,-1]
            res[:,:,-1] += x[:,:,0] + x[:,:,-2]
            return res
        return f3d

@njit
def stencil(x):
    return stencil_impl(x)

# helper function for iterative solution of SBR
# not currently used

@njit
def __gfunc__ (g, a0, b0):
    return 2*b0*g - g*g + a0

def __find_g_impl__(x0, x1, a0, b0, accuracy):
    raise NotImplementedError
@overload(__find_g_impl__)
def __find_g_impl_overload__(x0, x1, a0, b0, accuracy):
    __rf__ = njit(regula_falsi)
    def rf (x0, x1, a0, b0, accuracy):
        return __rf__(__gfunc__, x0, x1, accuracy=accuracy, isclose=np_isclose_all, fargs=(a0,b0))
    return rf
@njit
def __find_g__ (x0, x1, a0, b0, accuracy):
    return __find_g_impl__(x0, x1, a0, b0, accuracy)

# beta-scaling solver implementation
@njit
def _solve_block_g (istart, iend, h, Bq, Wq, g, dG, lambda_, sigma, delta, maxiter, accuracy, calc_moments, *kernel_args):

    for i in range(istart,iend):

        ibar = i//2
        C = - g[i-1] * dG[1]
        for k in range(2,ibar+1):
            C += (g[i-k+1] - g[i-k]) * dG[k]
        C *= 2
        if (i-ibar > ibar):
            C += (g[i-ibar] - g[i-ibar-1]) * dG[k]
        C += g[i-ibar] * g[ibar]
        C += delta * h*i - sigma

        gg = dG[1]/lambda_
        g[i] = gg - np.sqrt(gg*gg + C/lambda_)

        if calc_moments:
            dG[i] = 0.5 * (g[i-1] + g[i])

# SBR solver implementation
@njit
def _solve_block_g_iter (istart, iend, h, Bq, Wq, g, dG, lambda_, sigma, delta, maxiter, accuracy, calc_moments, M, dim, alpha, dx, *kernel_args):

    for i in range(istart,iend):

        ibar = i//2
        C = - g[i-1] * dG[1]
        for k in range(2,ibar+1):
            C += (g[i-k+1] - g[i-k]) * dG[k]
        C *= 2
        if (i-ibar > ibar):
            C += (g[i-ibar] - g[i-ibar-1]) * dG[k]
        C += g[i-ibar] * g[ibar]
        C += delta * h*i - sigma

        A = dG[1] + dim * alpha/dx**2

        iterations = 0
        converged = False
        newg = g[i-1].copy()
        while (not converged and iterations < maxiter):
            g[i] = newg
            Ci = C - alpha/dx**2 * stencil(g[i])
            #for n in range(M):
            #    newg.reshape(-1)[n] = __find_g__ ((2*g[i-1]-g[i-2]).reshape(-1)[n], g[i].reshape(-1)[n], Ci.reshape(-1)[n]/lambda_, A.reshape(-1)[n]/lambda_, accuracy=accuracy)
            newg = A/lambda_ - np.sqrt((A/lambda_)**2 + Ci/lambda_)
            iterations += 1
            if np.isclose (newg.reshape(-1), g[i].reshape(-1),
                           rtol=accuracy, atol=accuracy).all():
                converged = True
                g[i] = newg
        if calc_moments:
            dG[i] = 0.5 * (g[i-1] + g[i])


class beta_scaling_function (correlator):
    """Solver for the beta-scaling function of MCT.

    Solves either the standard asymptotic function of MCT, or the
    version introduced in stochastic beta-relaxation theory (SBR).

    Parameters
    ----------
    lam : float, default: 0.735
        Exponent parameter, must be in the range (0.5,1).
    sigma : float or array_like, default: 0.0
        Distance parameter to the glass transition.
    delta : float, default: 0.0
        Hopping parameter, must be non-negative.
    t0 : float, default: 1.0
        Time scale for the solutions, must be positive.
    M : int, default: 1
        Number of beta correlators to solve (for SBR).
    dim : int, default: 1
        Spatial dimension for SBR.
    alpha : float, default: 0.0
        Spatial coupling strength of SBR.
    dx : float, default: 1.0
        Lattice spacing for SBR.
    Dsigma : float, default: 1.0
        Variance of the sigma field for SBR, if sigma is a scalar.
    rng : object, default: None
        Random-number generator instance (for SBR if needed).

    Notes
    -----
    The other parameters take the same meaning as in
    :py:class:`mctpy.correlator`. Note that this solver here
    does not allow to specify a model: the beta-scaling equation
    is model-independent.

    If `M` is set to a value larger than zero, this solver solves the
    equations of SBR.
    In this case, `dim` can be set to the dimensionality of the spatial
    coupling. The correlators will then be matrices of shape (M,)*dim
    (dimensions 1, 2, and 3 are implemented), and the parameter `alpha`
    controls the strength of the dim-dimensional lattice Laplacian.
    The values `sigma` can then be an array of shape (M,)*dim to pass
    directly the values of the fluctuating distance parameter. If
    `sigma` is a scalar, it will be taken as the mean of a Gaussian field
    with standard deviation given by `Dsigma` that will be initialized (using
    the provided random-number generator, if given).
    Note that in this case, the value of the lattice spacing `dx` will
    be taken into account to properly scale `Dsigma` (as well as `alpha`
    that is always correctly scaled).
    """
    def __init__ (self, blocksize=256, h=1e-9, blocks=60, Tend=0.0,
                  maxinit=50, maxiter=10000, accuracy=1e-9, store=False,
                  lam=0.735, sigma=0., delta=0., t0=1.,
                  M=1, dim=1, alpha=0, dx=1.0, Dsigma=1.0, rng=None):
        self.M = M
        self.dim = dim
        correlator.__init__ (self, blocksize=blocksize, h=h, blocks=blocks,
            Tend=Tend, maxinit=maxinit, maxiter=maxiter, accuracy=accuracy,
            store=store, model=model_base(), base=None)
        self.lambda_ = lam
        self.sigma = sigma
        self.delta = delta
        self.t0 = t0
        if M > 1:
            self.alpha = alpha
            self.dx = dx
            if not hasattr(sigma, "__len__"):
                # here we assume that sigma is not an array
                # so then we take it to be the mean of a Gaussian
                if rng is None:
                    myrng = np.random.default_rng()
                else:
                    myrng = rng
                s = myrng.normal(loc=sigma, scale=Dsigma/np.power(dx,dim/2),
                                 size=(M,)*dim)
                s += sigma - np.mean(s)
                self.sigma = s
    def __alloc__ (self):
        self.phi_ = np.zeros((self.blocksize,*(self.M,)*self.dim))
        self.dPhi_ = np.zeros((self.halfblocksize+1,*(self.M,)*self.dim))
        self.m_ = np.zeros((self.blocksize,1))
        self.dM_ = np.zeros((self.halfblocksize+1,1))
        if self.store:
            self.t = np.zeros(self.halfblocksize*(self.blocks+1))
            self.phi = np.zeros((self.halfblocksize*(self.blocks+1),*(self.M,)*self.dim))
            self.m = np.zeros((self.halfblocksize*(self.blocks+1),1))

    def initial_values (self, imax=50):
        """Initial values for the beta-correlation functions."""
        iend = imax
        if (iend >= self.halfblocksize): iend = self.halfblocksize-1
        self.model.set_base(self.phi_)
        a, _ = exponents(self.lambda_)
        A1 = 1./(2 * (a * np.pi / np.sin(a*np.pi) - self.lambda_))
        self.phi_[0] = 0.0
        for i in range(1,iend):
            t = i*self.h0
            self.phi_[i] = np.power(t/self.t0,-a) \
                         + A1*self.sigma*np.power(t/self.t0,a)
        self.dPhi_[1] = np.power(self.h0/self.t0,-a)/(1-a) \
                      + A1*self.sigma*np.power(self.h0/self.t0,a)/(1+a)
        for i in range(2,iend+1):
            t = i*self.h0
            self.dPhi_[i] = np.power(self.h0/self.t0,-a)/(1-a) \
                            * (np.power(1./i,a-1) - np.power(i-1.,1.-a)) \
                          + A1*self.sigma*np.power(self.h0/self.t0,a)/(1+a) \
                            * (np.power(1./i,-a-1) - np.power(i-1.,1.+a))
        self.iend = iend

    def solve_block (self, istart, iend):
        if self.M == 1:
            _solve_block_g (istart, iend, self.h, self.model.Bq()/self.h, self.model.Wq(), self.phi_, self.dPhi_, self.lambda_, self.sigma, self.delta, self.maxiter, self.accuracy, (istart<self.blocksize//2), *self.model.kernel_extra_args())
        else:
            _solve_block_g_iter (istart, iend, self.h, self.model.Bq()/self.h, self.model.Wq(), self.phi_, self.dPhi_, self.lambda_, self.sigma, self.delta, self.maxiter, self.accuracy, (istart<self.blocksize//2), self.M, self.dim, self.alpha, self.dx, *self.model.kernel_extra_args())

    def type (self):
        return 'g'

    def h5save (self, base):
        super(beta_scaling_function,self).h5save(base)
        grp = base.create_group("model").create_group("beta_scaling")
        if self.M==1:
            grp.attrs['sigma'] = self.sigma
        else:
            grp.create_dataset("sigma",data=self.sigma)
        grp.attrs['lambda'] = self.lambda_
        grp.attrs['delta'] = self.delta
        grp.attrs['t0'] = self.t0
        grp.attrs['M'] = self.M
        grp.attrs['dim'] = self.dim
        if self.M>1:
            grp.attrs['alpha'] = self.alpha
            grp.attrs['dx'] = self.dx
        else:
            grp.attrs['alpha'] = 0.
            grp.attrs['dx'] = 1.

    @staticmethod
    def load (file):
        with h5py.File(file, 'r') as f:
            attrs = f['correlator/time_domain'].attrs
            gattrs = f['model/beta_scaling'].attrs
            newself = beta_scaling_function(
                blocksize = attrs['blocksize'], h=attrs['h0'],
                blocks=attrs['blocks'], maxiter=attrs['maxiter'],
                accuracy=attrs['accuracy'],
                lam=gattrs['lambda'], sigma=0.,
                delta=gattrs['delta'], t0=gattrs['t0'],
                M=gattrs['M'], dim=gattrs['dim'],
                alpha=gattrs['alpha'], dx=gattrs['dx'])
            newself.t = np.array(f['correlator/time_domain/t'])
            newself.phi = np.array(f['correlator/time_domain/phi'], dtype=newself.model.dtype)
            newself.m = np.array(f['correlator/time_domain/m'], dtype=newself.model.dtype)
            newself.store = True
            newself.solved = attrs['solved_blocks']
            if newself.M>1:
                newself.sigma = np.array(f['model/beta_scaling/sigma'])
            else:
                newself.sigma = gattrs['sigma']
        return newself
