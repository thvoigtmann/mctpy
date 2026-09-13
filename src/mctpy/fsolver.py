import numpy as np

from numba import njit
#import h5py

from .__util__ import void
from .__h5__ import load_model

@njit
def _fsolve (f, m, W, f0, phi0, kernel, accuracy, maxiter, *kernel_args):
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
    #df = np.mean(np.abs(newf-f[0]))
    f[0] = newf
    return converged

@njit
def _fsolve_mat (f, m, W, f0, phi0, WS, kernel, accuracy, maxiter, *kernel_args):
    iterations = 0
    converged = False
    lowval = False
    newf = f0.copy()
    M = f[0].shape[0]
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
    """
    def __init__ (self, model):
        self.model = model
        dim = model.matrix_dimension()
        vdim = model.vector_dimension()
        M = len(model)
        if self.model.scalar():
            self.shape = (M * vdim,)
            self.phishape = (M, vdim) if vdim > 1 else self.shape
        else:
            self.shape = (M * vdim, dim, dim)
            self.phishape = (M, vdim, dim, dim) if vdim > 1 else self.shape
        self.f_ = np.zeros((1,*self.shape),dtype=model.dtype)
        self.m_ = np.zeros((1,*self.shape),dtype=model.dtype)
        self.jit_kernel = model.get_kernel()
        self.solved = False

    def solve (self, accuracy=1e-12, maxiter=1000000,
               callback=None, callback_every=0):
        """Solve for the nonergodicity parameter.

        Parameters
        ----------
        accuracy : float, default: 1e-12
            Accuracy at which to terminate the iterative solution scheme.
        maxiter : int
            Maximum number of iterations, to ensure termination.
        callback : callable or None
            If set, this is called every `callback_every` iterations,
            with two parameters, first the iteration count, and second
            a reference to this object.
        callback_every : int, default 0
            How often to call the callback, if defined. A zero value
            implies on every iteration. If larger than zero, the total
            number of iterations will be adjusted to be an integer multiple
            of this number.

        Notes 
        -----
        The object's field `f` will be set to the NEP values,
        and `m` to the corresponding memory kernel.
        If the calculation converged within the set maximum number
        of iterations, the object's field `converged' will be set to
        `True`.
        """
        self.model.set_base_phi(self.f_)
        self.model.get_set_V()(0, *self.model.set_V_args())
        if callback is not None and callback_every>0:
            blocks = maxiter//callback_every
            block_iter = callback_every
            if maxiter > block_iter*blocks:
                print ("dropping",self.maxiter-block_iter*blocks,"iterations")
        else:
            blocks = 1
            block_iter = maxiter
        f0 = self.model.phi0().copy() # TODO do we need copy() ?
        converged = False
        for b in range(blocks):
            if self.model.scalar():
                converged = _fsolve(self.f_, self.m_, self.model.Wq(), f0, self.model.phi0(), self.jit_kernel, accuracy, block_iter, *self.model.kernel_extra_args())
            else:
                converged = _fsolve_mat(self.f_, self.m_, self.model.Wq(), f0, self.model.phi0(), self.model.WqSq(), self.jit_kernel, accuracy, block_iter, *self.model.kernel_extra_args())
            if callback is not None:
                callback(block_iter*(b+1),self)
            if converged: break
            if b < blocks-1:
                f0 = self.f_[0]
        self.converged = converged
        self.f = self.f_[0].reshape(self.phishape)
        self.m = self.m_[0].reshape(self.phishape)
        self.solved = True
        self.accuracy = accuracy
    def h5save (self, h5file):
        grp = h5file.create_group('nonergodicity_parameter')
        if self.solved:
            grp.create_dataset("f", data=self.f)
            grp.create_dataset("m", data=self.m)
            grp.attrs['converged'] = self.converged
            grp.attrs['accuracy'] = self.accuracy
        self.model.h5save (grp)
    #@staticmethod
    #def load (file):
    #    """Load nonergodicity parameter from h5 file.
    #
    #    This returns a newly created nonergodicity_parameter object.
    #    """
    #    with h5py.File(file, 'r') as f:
    #        nep = nonergodicity_parameter.h5load(f)
    #    return nep
    @staticmethod
    def h5load (h5file):
        h5grp = h5file['nonergodicity_parameter']
        model = load_model(h5grp)
        newself = nonergodicity_parameter(model)
        if 'f' in h5grp:
            newself.f = np.array(h5grp['f'], dtype=model.dtype)
            newself.m = np.array(h5grp['m'], dtype=model.dtype)
            newself.solved = True
            newself.converged = h5grp.attrs['converged']
            newself.accuracy = h5grp.attrs['accuracy']
            newself.f_ = newself.f.reshape((1,*newself.shape))
            newself.m_ = newself.m.reshape((1,*newself.shape))
        else:
            newself.solved = False
        return newself



@njit
def _esolve(e, dm, f, maxiter, accuracy, *kernel_args):
    iterations = 0
    converged = False
    newe = np.ones_like(e)
    while (not converged and iterations < maxiter):
        e[:] = newe
        #newe = (1-f)*model.dm(f,self.e)*(1-f)
        dm(newe,f,e,*kernel_args)
        norm = np.sqrt(np.dot(newe,newe))
        if norm>1e-10: newe = newe/norm
        if np.isclose (newe, e, rtol=accuracy, atol=0.0).all():
            converged = True
            e[:] = newe
    return norm, converged

@njit
def _esolve_mat(e, dm, f, phi0, maxiter, accuracy, *kernel_args):
    iterations = 0
    converged = False
    newe = phi0.copy()
    S_F = phi0 - f
    M = newe.shape[0]
    while (not converged and iterations < maxiter):
        iterations+=1
        e[:] = newe
        dm(newe,f,e,*kernel_args)
        for q in range(M):
            newe[q] = S_F[q] @ newe[q] @ S_F[q]
        norm = np.sqrt(np.dot(newe.reshape(-1),newe.reshape(-1)))
        if norm>1e-10: newe = newe/norm
        converged = np.isclose (newe.reshape(-1), e.reshape(-1),
                                rtol=accuracy, atol=0.0).all()
        if not converged and iterations >= 50:
           converged = (np.mean(np.abs(newe.reshape(-1)-e.reshape(-1))) / newe.reshape(-1).shape[0] < accuracy)
        if converged:
            e[:] = newe
    return norm, converged

@njit
def _ehatsolve(ehat, dmhat, f, maxiter, accuracy, *kernel_args):
    iterations = 0
    converged = False
    newehat = np.ones_like(ehat)
    while (not converged and iterations < maxiter):
        ehat[:] = newehat
        dmhat(newehat,f,ehat,*kernel_args)
        norm = np.sqrt(np.dot(newehat,newehat))
        if norm>1e-10: newehat = newehat/norm
        if np.isclose (newehat,ehat,rtol=accuracy,atol=0.0).all():
            converged = True
            ehat[:] = newehat
    return norm, converged

@njit
def _ehatsolve_mat(ehat, dmhat, f, phi0, maxiter, accuracy, *kernel_args):
    iterations = 0
    converged = False
    newehat = phi0.copy()
    S_F = phi0 - f
    tmp = np.zeros_like(ehat)
    M = tmp.shape[0]
    while (not converged and iterations < maxiter):
        iterations+=1
        ehat[:] = newehat
        for q in range(M):
            tmp[q] = S_F[q] @ ehat[q] @ S_F[q]
        dmhat(newehat,f,tmp,*kernel_args)
        norm = np.sqrt(np.dot(newehat.reshape(-1),newehat.reshape(-1)))
        if norm>1e-10: newehat = newehat/norm
        converged = np.isclose (newehat.reshape(-1), ehat.reshape(-1),
                                rtol=accuracy, atol=0.0).all()
        if not converged and iterations >= 50:
           converged = (np.mean(np.abs(newehat.reshape(-1)-ehat.reshape(-1))) / newehat.reshape(-1).shape[0] < accuracy)
        if converged:
            ehat[:] = newehat
    return norm, converged


class eigenvalue (object):
    """Critical eigenvalue solver for a simple liquid.

    This implements iteration schemes to evaluate the right and left
    eigenvectors of the MCT stability matrix, as well as the exponent
    parameter lambda.

    It works with the standard "simple-liquid" model of MCT, as well as
    with the matrix-valued models. The latter use a somewhat different
    normalization convention.
    """
    def __init__ (self, nep):
        self.nep = nep
        self.dm = self.nep.model.get_dm()
        self.dmhat = self.nep.model.get_dmhat()
        self.dm2 = self.nep.model.get_dm2()
        self.shape = self.nep.shape
        self.phishape = self.nep.phishape

    def solve (self, accuracy=1e-12, maxiter=1000):
        """Solve for the critical eigenvectors.

        This calculates the right- and left critical eigenvector, the
        corresponding eigenvalues, and the exponent parameter.
        The calculation of the "largest eigenvector" is done by an
        iteration scheme, once for the left and once for the right vector.
        If both converge within the set limit of maximum iterations,
        the classes' `converged` field will be set to `True`.
        """
        model = self.nep.model
        vdim = model.vector_dimension()
        self.nep.model.set_C (self.nep.f_[0])
        f = self.nep.f_[0]
        self.e = np.zeros(self.shape,dtype=model.dtype)
        self.ehat = np.zeros(self.shape,dtype=model.dtype)
        if model.scalar():
            model.set_base_phi(self.e)
            self.eval, conv1 = _esolve(self.e, self.dm, f, maxiter, accuracy, *self.nep.model.kernel_extra_args_dm())
            model.set_base_phi(self.ehat)
            self.eval2, conv2 = _ehatsolve(self.ehat, self.dmhat, f, maxiter, accuracy, *self.nep.model.kernel_extra_args_dmhat())
        else:
            phi0 = model.phi0()
            model.set_base_phi(self.e)
            self.eval, conv1 = _esolve_mat(self.e, self.dm, f, phi0, maxiter, accuracy, *self.nep.model.kernel_extra_args_dm())
            model.set_base_phi(self.ehat)
            self.eval2, conv2 = _ehatsolve_mat(self.ehat, self.dmhat, f, phi0, maxiter, accuracy, *self.nep.model.kernel_extra_args_dmhat())

        self.converged = conv1 & conv2

        if self.eval > 0:
            dq = model.dq()
            if vdim > 1:
                dq = np.repeat(dq,vdim)
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
            L = np.zeros_like(self.e)
            self.dm2(L,f,self.e,*self.nep.model.kernel_extra_args_dm2())
            if model.scalar():
                self.lam = np.dot(dq * self.ehat, L)
            else:
                self.lam = np.einsum('i,iab,iak,ikl,ilb',dq,self.ehat,
                                     S_F, L, S_F)
        else:
            self.lam = 0.0
        self.e = self.e.reshape(self.phishape)
        self.ehat = self.ehat.reshape(self.phishape)
        self.solved = True
        self.accuracy = accuracy

    def sigma (self, model_nc):
        """Calculate distance parameter with respect to this state point.

        Parameters
        ----------
        model_nc : mctpy.model_base
            `mctpy.model_base` object representing a "non-critical" point
            whose distance to the object here (taken to represent the
            "critical" point) shall be calculated. Should match the type
            of this object's `model`; if not, results are undefined.

        Returns
        -------
        sigma : float
            MCT distance parameter.
        """
        # TODO test for vector-models
        dq = self.nep.model.dq()
        vdim = self.nep.model.vector_dimension()
        if vdim > 1:
            dq = np.repeat(dq,vdim)
        e = self.e.reshape(self.shape)
        ehat = self.ehat.reshape(self.shape)
        fc = self.nep.f.reshape(self.shape)
        mc = self.nep.model.mrenorm(self.nep.m.reshape(self.shape))
        m = model_nc.m(self.nep.f.reshape(self.shape))
        # TODO divide not by self.nep.model.Wq() but by mrenorm
        if self.nep.model.scalar():
            norm = np.dot(dq * ehat, e*e * (1-fc))
            s = np.dot(dq * self.ehat, (m - mc))
            return s/norm
        else:
            Sc = self.nep.model.phi0().reshape(self.shape)
            Scinv = np.linalg.inv(Sc)
            S = model_nc.phi0().reshape(self.shape)
            Sc_Fc = Sc - fc
            S_Fc = S - fc
            Sc_Fc_inv = np.linalg.inv(Sc_Fc)
            norm = np.einsum('i,iab,iak,ikl,ilb', dq, ehat, e, Sc_Fc_inv, e)
            dSM = np.einsum('...ak,...kl,...lb', S, m, S_Fc) \
                - np.einsum('...ak,...kl,...lb', Sc, mc, Sc_Fc)
            s = np.einsum('i,iab,iak,ikl,ilb', dq, self.ehat, Sc_Fc, Scinv, dSM)
            return s/norm
    def h5save (self, h5file):
        grp = h5file.create_group('eigenvalue')
        if self.solved:
            grp.create_dataset("e", data=self.e)
            grp.create_dataset("ehat", data=self.ehat)
            grp.attrs['eval'] = self.eval
            grp.attrs['eval2'] = self.eval2
            grp.attrs['lambda'] = self.lam
            grp.attrs['converged'] = self.converged
            grp.attrs['accuracy'] = self.accuracy
        self.nep.h5save (grp)
    @staticmethod
    def h5load (h5file):
        h5grp = h5file['eigenvalue']
        nep = nonergodicity_parameter.h5load (h5grp)
        newself = eigenvalue(nep)
        if 'e' in h5grp:
            model = nep.model
            newself.e = np.array(h5grp['e'], dtype=model.dtype)
            newself.ehat = np.array(h5grp['ehat'], dtype=model.dtype)
            newself.solved = True
            newself.converged = h5grp.attrs['converged']
            newself.accuracy = h5grp.attrs['accuracy']
            newself.lam = h5grp.attrs['lambda']
            newself.eval = h5grp.attrs['eval']
            newself.eval2 = h5grp.attrs['eval2']
        else:
            newself.solved = False
        return newself
