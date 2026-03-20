import numpy as np

from numba import njit

from .__util__ import void

@njit
def _fsolve (f, m, W, f0, phi0, kernel, M, accuracy, maxiter, *kernel_args):
    iterations = 0
    converged = False
    newf = f0
    while (not converged and iterations < maxiter):
        iterations+=1
        f[0] = newf
        kernel (m[0], f[0], 0, 0., *kernel_args)
        newf = m[0] / (W+m[0]) * phi0
        if np.isclose (newf, f[0], rtol=accuracy, atol=0.0).all():
            converged = True
            #f[0] = newf
        #if not iterations%100: print("iter",iterations,np.mean(np.abs(newf-f[0])))
    #df = np.mean(np.abs(newf-f[0]))
    f[0] = newf
    return converged

@njit
def _fsolve_mat (f, m, W, f0, phi0, WS, kernel, M, accuracy, maxiter, *kernel_args):
    iterations = 0
    converged = False
    lowval = False
    newf = f0.copy()
    #newf = phi0.copy()
    while (not converged and iterations < maxiter):
        iterations+=1
        f[0] = newf
        kernel (m[0], f[0], 0, 0., *kernel_args)
        if 0 and not lowval:
            for q in range(M):
                newf[q] = phi0[q] - np.linalg.inv(W[q] + m[0][q]) @ WS[q]
        else:
            for q in range(M):
                newf[q] = np.linalg.inv(W[q] + m[0][q]) @ m[0][q] @ phi0[q]
        if np.isclose (newf.reshape(-1), f[0].reshape(-1), rtol=accuracy, atol=0.0).all():
            converged = True
            #f[0] = newf
        if np.isclose (newf.reshape(-1), 0.0, rtol=0.0, atol=10*accuracy).all():
            # suspect liquid solution
            lowval = True
    #df = np.mean(np.abs(newf-f[0]))
    f[0] = newf
    return converged

class nonergodicity_parameter (object):
    """Solver for the nonergodicity parameter.

    This implements a robust iteration scheme to solve for the NEP.
    It works with the standard "simple-liquid" model of MCT, as well as
    with the matrix-valued models.

    Parameters
    ----------
    model : object
        MCT model defining the memory kernel for the NEP.
    accuracy : float, default: 1e-12
        Accuracy at which to terminate the iterative solution scheme.
    maxiter : int
        Maximum number of iterations, to ensure termination.
    """
    def __init__ (self, model, accuracy=1e-12, maxiter=1000000):
        self.model = model
        self.accuracy = accuracy
        self.maxiter = maxiter
        self.dim = model.matrix_dimension()
        self.M = len(model)
        self.f_ = np.zeros((1,self.M*self.dim**2),dtype=model.dtype)
        self.m_ = np.zeros((1,self.M*self.dim**2),dtype=model.dtype)
        self.jit_kernel = model.get_kernel()

    def solve (self, callback=None, callback_every=0):
        """Solve for the nonergodicity parameter.

        Result: The object's field `f` will be set to the NEP values,
        and `m` to the corresponding memory kernel.
        """
        self.model.set_base(self.f_)
        if callback is not None and callback_every>0:
            blocks = self.maxiter//callback_every
            block_iter = callback_every
            if self.maxiter > block_iter*blocks:
                print ("dropping",self.maxiter-block_iter*blocks,"iterations")
        else:
            blocks = 1
            block_iter = self.maxiter
        if self.model.scalar():
            f0 = self.model.phi0().copy()
            for b in range(blocks):
                converged = _fsolve(self.f_, self.m_, self.model.Wq(), f0, self.model.phi0(), self.jit_kernel, self.M, self.accuracy, block_iter, *self.model.kernel_extra_args())
                if callback is not None:
                    callback(block_iter*(b+1),self)
                if converged: break
                if b < blocks-1:
                    f0 = self.f_[0]
            self.f = self.f_[0]
            self.m = self.m_[0]
        else:
            f0 = self.model.phi0()
            for b in range(blocks):
                converged = _fsolve_mat(self.f_.reshape(1,-1,self.dim,self.dim), self.m_.reshape(1,-1,self.dim,self.dim), self.model.Wq().reshape(-1,self.dim,self.dim), f0.reshape(-1,self.dim,self.dim), self.model.phi0().reshape(-1,self.dim,self.dim), self.model.WqSq().reshape(-1,self.dim,self.dim), self.jit_kernel, self.M, self.accuracy, block_iter, *self.model.kernel_extra_args())
                if callback is not None:
                    callback(block_iter*(b+1),self)
                if converged: break
                if b < blocks-1:
                    f0 = self.f_[0]
            self.f = self.f_[0].reshape(-1,self.dim,self.dim)
            self.m = self.m_[0].reshape(-1,self.dim,self.dim)


@njit
def _esolve(e, dm, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newe = np.ones(M)
    while (not converged and iterations < maxiter):
        e[:] = newe
        #newe = (1-f)*model.dm(f,self.e)*(1-f)
        dm(newe,f,e)
        norm = np.sqrt(np.dot(newe,newe))
        if norm>1e-10: newe = newe/norm
        if np.isclose (newe, e, rtol=accuracy, atol=0.0).all():
            converged = True
            e[:] = newe
    return norm

@njit
def _esolve_mat(e, dm, f, phi0, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newe = phi0.copy()
    S_F = phi0 - f
    while (not converged and iterations < maxiter):
        iterations+=1
        e[:] = newe
        dm(newe,f,e)
        for q in range(M):
            newe[q] = S_F[q] @ newe[q] @ S_F[q]
        norm = np.sqrt(np.dot(newe.reshape(-1),newe.reshape(-1)))
        if norm>1e-10: newe = newe/norm
        if np.isclose (newe.reshape(-1), e.reshape(-1),
                       rtol=accuracy, atol=0.0).all():
            converged = True
            e[:] = newe
    return norm

@njit
def _ehatsolve(ehat, dmhat, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newehat = np.ones(M)
    while (not converged and iterations < maxiter):
        ehat[:] = newehat
        dmhat(newehat,f,ehat)
        norm = np.sqrt(np.dot(newehat,newehat))
        if norm>1e-10: newehat = newehat/norm
        if np.isclose (newehat,ehat,rtol=accuracy,atol=0.0).all():
            converged = True
            ehat[:] = newehat
    return norm

@njit
def _ehatsolve_mat(ehat, dmhat, f, phi0, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newehat = phi0.copy()
    S_F = phi0 - f
    tmp = np.zeros_like(ehat)
    while (not converged and iterations < maxiter):
        iterations+=1
        ehat[:] = newehat
        for q in range(M):
            tmp[q] = S_F[q] @ ehat[q] @ S_F[q]
        dmhat(newehat,f,tmp)
        norm = np.sqrt(np.dot(newehat.reshape(-1),newehat.reshape(-1)))
        if norm>1e-10: newehat = newehat/norm
        if np.isclose(newehat.reshape(-1),ehat.reshape(-1),
                      rtol=accuracy, atol=0.0).all():
            converged = True
            ehat[:] = newehat
    return norm


class eigenvalue (object):
    """Critical eigenvalue solver for a simple liquid.

    This implements iteration schemes to evaluate the right and left
    eigenvectors of the MCT stability matrix, as well as the exponent
    parameter lambda.

    It works with the standard "simple-liquid" model of MCT, as well as
    with the matrix-valued models. The latter use a somewhat different
    normalization convention.
    """
    def __init__ (self, nep, accuracy=1e-12, maxiter=1000000):
        self.nep = nep
        self.accuracy = accuracy
        self.maxiter = maxiter
        nep.model.set_C (self.nep.f_[0])
        f, m = self.nep.f_[0], self.nep.m_[0]
        self.dm = self.nep.model.get_dm()
        self.dmhat = self.nep.model.get_dmhat()
        self.dm2 = self.nep.model.get_dm2()

    def solve (self):
        """Solve for the critical eigenvectors.

        This calculates the right- and left critical eigenvector, the
        corresponding eigenvalues, and the exponent parameter.
        """
        model = self.nep.model
        dim = model.matrix_dimension()
        if model.scalar():
            f = self.nep.f_[0]
            self.e = np.zeros(len(model)*dim**2,dtype=model.dtype)
            self.ehat = np.zeros(len(model)*dim**2,dtype=model.dtype)
            self.eval = _esolve(self.e, self.dm, f, len(model), self.maxiter, self.accuracy)
            self.eval2 = _ehatsolve(self.ehat, self.dmhat, f, len(model), self.maxiter, self.accuracy)
        else:
            f = self.nep.f_[0].reshape(-1,dim,dim)
            self.e = np.zeros((len(model),dim,dim),dtype=model.dtype)
            self.ehat = np.zeros((len(model),dim,dim),dtype=model.dtype)
            phi0 = model.phi0()
            self.eval = _esolve_mat(self.e, self.dm, f, phi0.reshape(-1,dim,dim), len(model), self.maxiter, self.accuracy)
            self.eval2 = _ehatsolve_mat(self.ehat, self.dmhat, f, phi0.reshape(-1,dim,dim), len(model), self.maxiter, self.accuracy)

        if self.eval > 0:
            dq = model.dq()
            if model.scalar():
                nl = np.dot(dq * self.ehat, self.e)
                nr = np.dot(dq * self.ehat, self.e*self.e * (1-f))
            else:
                S_F = phi0 - f
                S_F_inv = np.linalg.inv(S_F)
                nl = np.einsum('i,iab,iab', dq, self.ehat, self.e)
                nr = np.einsum('i,iab,iak,ikl,ilb', dq, self.ehat,
                               self.e, S_F_inv, self.e)
            self.e = self.e * nl/nr
            self.ehat = self.ehat * nr/(nl*nl)
            #self.lam = np.dot(self.ehat, (1-f)*self.dm2(f,self.e)*(1-f))
            L = np.zeros_like(self.e)
            self.dm2(L,f,self.e)
            if model.scalar():
                self.lam = np.dot(dq * self.ehat, L)
            else:
                self.lam = np.einsum('i,iab,iak,ikl,ilb',dq,self.ehat,
                                     S_F, L, S_F)
        else:
            self.lam = 0.0


