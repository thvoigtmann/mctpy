import numpy as np

from numba import njit

import h5py

from .__util__ import model_base, loaded_model, np_isclose_all
from .util import exponents, CorrelatorBase

import scipy.sparse.linalg
from numba import objmode

@njit
def _decimize (phi, m, dPhi, dM, blocksize):
    halfblocksize = blocksize//2
    imid = halfblocksize//2
    for i in range(1,imid):
        di = i+i
        dPhi[i] = 0.5 * (dPhi[di-1] + dPhi[di])
        dM[i] = 0.5 * (dM[di-1] + dM[di])
    for i in range(imid,halfblocksize):
        di = i+i
        dPhi[i] = 0.25 * (phi[di-2] + 2*phi[di-1] + phi[di])
        dM[i] = 0.25 * (m[di-2] + 2*m[di-1] + m[di])
    dPhi[halfblocksize] = phi[blocksize-1]
    dM[halfblocksize] = m[blocksize-1]
    for i in range(halfblocksize):
        di = i+i
        phi[i] = phi[di]
        m[i] = m[di]

@njit
def _solve_block (istart, iend, h, Aq, Bq, Wq, Xq, phi, m, dPhi, dM, kernel, maxiter, accuracy, calc_moments, hopping, pre_m, *kernel_args):

    for i in range(istart,iend):
        mpre = pre_m[i-istart]
        if not hopping:
            A = mpre*dM[1] + Wq + 1.5*Bq / h + 2.0*Aq / (h*h)
            B = mpre*(-dPhi[1] + phi[0]) / A
        else:
            A = mpre*(1.+0.5*h*Xq)*dM[1] + Wq + 1.5*Bq / h + 2.0*Aq / (h*h)
            B = mpre*(-(1+0.5*h*Xq)*dPhi[1] + phi[0]) / A

        ibar = i//2
        C = -(m[i-1]*dPhi[1] + phi[i-1]*dM[1])
        for k in range(2,ibar+1):
            C += (m[i-k+1] - m[i-k]) * dPhi[k]
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        if (i-ibar > ibar):
            C += (phi[i-ibar] - phi[i-ibar-1]) * dM[k]
        C += m[i-ibar] * phi[ibar]
        if hopping:
            Cd = (m[i-1]*dPhi[1] + phi[i-1]*dM[1])
            for k in range(2,ibar+1):
                Cd += (m[i-k+1] + m[i-k]) * dPhi[k]
                Cd += (phi[i-k+1] + phi[i-k]) * dM[k]
            if (i-ibar > ibar):
                Cd += (phi[i-ibar] + phi[i-ibar-1]) * dM[k]
            C += 0.5*h*Xq*Cd
        C = mpre*C + (-2.0*phi[i-1] + 0.5*phi[i-2]) * Bq/h \
                   + (-5.0*phi[i-1] + 4.0*phi[i-2] - 1.0*phi[i-3]) * Aq/(h*h)
        C = C/A

        iterations = 0
        converged = False
        newphi = phi[i-1]
        while (not converged and iterations < maxiter):
            phi[i] = newphi
            kernel (m[i], phi[i], i, h*i, *kernel_args)
            newphi = B*m[i] - C
            iterations += 1
            if np.isclose (newphi, phi[i],
                           rtol=accuracy, atol=accuracy).all():
                converged = True
                phi[i] = newphi
                if calc_moments:
                    dPhi[i] = 0.5 * (phi[i-1] + phi[i])
                    dM[i] = 0.5 * (m[i-1] + m[i])

##@njit
#def __ffunc__ (g, a0, b0):
#    return 2*b0*g - g*g + a0
#
#def __find_f_impl__(x0, x1, a0, b0, accuracy):
#    raise NotImplementedError
#@overload(__find_f_impl__)
#def __find_f_impl_overload__(x0, x1, a0, b0, accuracy):
#    __rf__ = njit(regula_falsi)
#    def rf (x0, x1, a0, b0, accuracy):
#        return __rf__(__ffunc__, x0, x1, accuracy=accuracy, isclose=np_isclose_all, fargs=(a0,b0))
#    return rf
#@njit
#def __find_f__ (x0, x1, a0, b0, accuracy):
#    return __find_f_impl__(x0, x1, a0, b0, accuracy)
#
#
@njit
def _solve_block_mat (istart, iend, h, Aq, Bq, Wq, Xq, phi, m, dPhi, dM, M, kernel, maxiter, accuracy, calc_moments, useA, useB, useX, *kernel_args):

    Ainv = np.zeros_like(Wq,dtype=phi.dtype)
    B = np.zeros_like(Wq,dtype=phi.dtype)
    B0 = np.zeros_like(Wq,dtype=phi.dtype)
    C = np.zeros_like(Wq,dtype=phi.dtype)
    if useX:
        Cd = np.zeros_like(Wq,dtype=phi.dtype)

    L = (Ainv.shape[1]-1)//2
    lr = np.arange(-L,L+1)

    Ainv = Wq + dM[1] + Bq*1.5/h + Aq*2.0/h**2
    if useX:
        AinvX = Wq + dM[1] + Bq*1.5/h + Aq*2.0/h**2 #+ dM[1] @ Xq[q] * 0.5*h
    Anotinv = Ainv.copy()
    #X = np.ones((M,2*L+1,2*L+1),dtype=complex)*np.diag(np.ones(2*L+1))
    Y = np.zeros_like(Ainv)
    Y[:,L,L] = 1.0 + 0.0j
    #X = X-Y
    X = np.zeros_like(Ainv)
    for l in lr:
        if l<0:
            X[:,L+l,L+l] = 1.0 + 0.0j
        elif l>0:
            X[:,L+l,L+l] = 1.0 + 0.0j
    PY = np.zeros((M,2*L+1,1),dtype=phi.dtype)
    PY[:,L,0] = 1.0
    PX = np.zeros((M,2*L+1,2*L),dtype=phi.dtype)
    for l in lr:
        if l<0:
            PX[:,L+l,L+l] = 1.0
        elif l>0:
            PX[:,L+l,L+l-1] = 1.0
    A_D = np.zeros((M,1,1),dtype=phi.dtype)
    A_B = np.zeros((M,2*L,1),dtype=phi.dtype)
    A_C = np.zeros((M,1,2*L),dtype=phi.dtype)
    A_A = np.zeros((M,2*L,2*L),dtype=phi.dtype)
    L_A = np.zeros_like(A_A)
    L_B = np.zeros_like(A_B)
    L_C = np.zeros_like(A_C)
    L_D = np.zeros_like(A_D)
    schur = np.zeros_like(A_A)
    for q in range(M):
        AinvX[q] += dM[1,q] @ Xq[q] * 0.5*h
        A_A[q] = PX[q].T @ AinvX[q] @ PX[q]
        A_B[q] = PX[q].T @ AinvX[q] @ PY[q]
        A_C[q] = PY[q].T @ AinvX[q] @ PX[q]
        A_D[q] = PY[q].T @ AinvX[q] @ PY[q]
        schur[q] = A_A[q] - A_B[q] @ np.linalg.inv(A_D[q]) @ A_C[q]
        schur[q] = np.linalg.inv(schur[q])
        L_A[q] = schur[q]
        L_B[q] = - schur[q] @ A_B[q] @ np.linalg.inv(A_D[q])
        L_C[q] = - np.linalg.inv(A_D[q]) @ A_C[q] @ schur[q]
        L_D[q] = np.linalg.inv(A_D[q]) + np.linalg.inv(A_D[q]) @ A_C[q] @ schur[q] @ A_B[q] @ np.linalg.inv(A_D[q])
        Ainv[q] = np.linalg.inv(Ainv[q])
        AinvX[q] = np.linalg.inv(AinvX[q])
        B[q] = -dPhi[1,q] + phi[0,q]
        B0[q] = -dPhi[1,q] + phi[0,q]
        if useX:
            B[q] -= Xq[q] @ dPhi[1,q] * 0.5*h

    for i in range(istart,iend):
        ibar = i//2
        for q in range(M):
            C[q] = -( m[i-1,q] @ dPhi[1,q] + dM[1,q] @ phi[i-1,q] )
        for k in range(2,ibar+1):
            for q in range(M):
                C[q] += (m[i-k+1,q] - m[i-k,q]) @ dPhi[k,q] \
                      + dM[k,q] @ (phi[i-k+1,q] - phi[i-k,q])
        if (i-ibar > ibar):
            for q in range(M):
                C[q] += dM[k,q] @ (phi[i-ibar,q] - phi[i-ibar-1,q])
        if useX:
            for q in range(M):
                Cd[q] = m[i-1,q] @ Xq[q] @ dPhi[1,q] + dM[1,q] @ Xq[q] @ phi[i-1,q]
            for k in range(2,ibar+1):
                for q in range(M):
                    Cd[q] += (m[i-k+1,q] + m[i-k,q]) @ Xq[q] @ dPhi[k,q] \
                          + dM[k,q] @ Xq[q] @ (phi[i-k+1,q] + phi[i-k,q])
            if (i-ibar > ibar):
                for q in range(M):
                    Cd[q] += dM[k,q] @ Xq[q] @ (phi[i-ibar,q] + phi[i-ibar-1,q])
            # tst
            #for q in range(M):
            #    C[q] += 0.5*h*Cd[q]

        for q in range(M):
            C[q] += m[i-ibar,q] @ phi[ibar,q]
            if useB:
                C[q] += Bq[q] @ (-2*phi[i-1,q] + 0.5*phi[i-2,q]) / h
            if useA:
                C[q] += Aq[q] @ (-5*phi[i-1,q] + 4*phi[i-2,q] - phi[i-3,q])/h**2
            #C[q] = Ainv[q] @ C[q]

        iterations = 0
        converged = False
        newphi = phi[i-1].copy()
        newphiY = newphi.copy()
        while (not converged and iterations < maxiter):
            phi[i] = newphi
            kernel (m[i], phi[i], i, h*i, *kernel_args)
            for q in range(M):
                # 3. strange idea
                #newphi[q] = AinvX[q] @ (m[i,q] @ Xq[q] @ dPhi[1,q] *0.5*h - Cd[q]*0.5*h)
                #newphiY[q] = Ainv[q] @ (m[i,q] @ B0[q] - C[q])
                #newphi[q] = X[q] @ newphi[q] + Y[q] @ newphiY[q]
                #
                #newphiY[q] = Ainv[q] @ (m[i,q] @ B[q] - C[q]) - X[q] @ newphi[q] - AinvX[q] @ (Cd[q]*0.5*h + dM[1,q] @ Xq[q]*0.5*h @ phi[i,q])
                #newphiY[q] = Ainv[q] @ (m[i,q] @ B0[q] - C[q])
                #newphiY[q] = Ainv[q] @ (m[i,q] @ B0[q] - C[q]) - AinvX[q] @ (dM[1,q] @ Xq[q] @ Ainv[q] @ (m[i,q] @ B0[q] - C[q]) + m[i,q] @ Xq[q] @ dPhi[1,q] + Cd[q])*0.5*h
                #newphi[q] = AinvX[q] @ (m[i,q] @ X[q] @ B[q] - X[q] @ C[q] - Cd[q]*0.5*h)
                #newphi[q] = AinvX[q] @ (m[i,q] @ B[q] - C[q] - Cd[q]*0.5*h) - AinvX[q] @ (Anotinv[q] @ Y[q] @ phi[i,q])
                #newphiY[q] = Ainv[q] @ (m[i,q] @ Y[q] @ B0[q] - Y[q] @ C[q])
                #newphi[q] = X[q] @ newphi[q] + Y[q] @ newphiY[q]
                # 2. inversion using Schur complement
                #tmp = m[i,q] @ B[q] - C[q] - 0.5*h*Cd[q]
                #tmp_A = PX[q].T @ tmp @ PX[q]
                #tmp_B = PX[q].T @ tmp @ PY[q]
                #tmp = m[i,q] @ B0[q] - C[q]
                #tmp_C = PY[q].T @ tmp @ PX[q]
                #tmp_D = PY[q].T @ tmp @ PY[q]
                #newphi_A = L_A[q] @ tmp_A + L_B[q] @ tmp_C
                #newphi_B = L_A[q] @ tmp_B + L_B[q] @ tmp_D
                #newphi_C = L_C[q] @ tmp_A + L_D[q] @ tmp_C
                #newphi_D = L_C[q] @ tmp_B + L_D[q] @ tmp_D
                #newphi[q] = PX[q] @ newphi_A @ PX[q].T \
                #          + PX[q] @ newphi_B @ PY[q].T \
                #          + PY[q] @ newphi_C @ PX[q].T \
                #          + PY[q] @ newphi_D @ PY[q].T
                # 0. direct iteration
                newphi[q] = AinvX[q] @ (m[i,q] @ B[q] - C[q] - Cd[q]*0.5*h)
                # 1. semi-implicit iteration
                #newphi[q] = Ainv[q] @ (m[i,q] @ B[q] - C[q] - Cd[q]*0.5*h)
                #newphi[q] = Ainv[q] @ (m[i,q] @ B[q] - C[q])
                #if useX:
                #    newphi[q] -= Ainv[q] @ (Cd[q] + dM[1,q] @ Xq[q] @ phi[i,q]) * 0.5*h
                #
                # 3. bisection?

                for l in lr:
                    for ld in lr:
                        if not (l-ld)%2: # even
                            newphi[q,L+l,L+ld] = newphi[q,L+l,L+ld].real + 0.j
                        else:
                            newphi[q,L+l,L+ld] = newphi[q,L+l,L+ld].imag*1j + 0.0

            iterations += 1
            if np_isclose_all(newphi.reshape(-1), phi[i].reshape(-1),
                          rtol=accuracy, atol=accuracy):
                converged = True
                phi[i,...] = newphi
                if calc_moments:
                    dPhi[i] = 0.5 * (phi[i-1] + phi[i])
                    dM[i] = 0.5 * (m[i-1] + m[i])

class correlator (CorrelatorBase):
    """Class for calculating MCT correlators in the time domain.

    This sets up the standard solver developed for MCT-type equations,
    using a block-wise regular time-domain grid, and a "decimation"
    procedure doubling the step size from block to block.

    Parameters
    ----------
    blocksize : int
        Number of equidistant time steps per block. Will be adjusted
        to be divisible by four.
    h : float
        Stepsize in the first block.
    blocks : int
        Number of blocks to solve for.
    Tend : float, optional
        If larger than zero, fix the number of blocks such that
        the solution at least span the times up to `Tend`.
        If given, the user-specified value of `blocks` will be ignored,
        but both `blocksize` and `h` will be kept. The actual
        final time reached by the solver can be larger than `Tend`.
    maxinit : int, optional
        Number of points in the first block to initialize from a
        short-time expansion of the equations. Must be smaller than
        half the blocksize.
    maxiter : int, optional
        Maximum number of iterations taken per time step.
    accuracy : float, optional
        Accuracy goal for the iterations taken at each time step.
    store : bool, default: False
        If set to True, the full solutions will be stored in memory.
        This is needed for later processing in the main program, and
        for saving. If False, only the last block is kept.
        Mostly there for historical reasons and to save some memory
        if needed.
    model : model_base
        The actual MCT model to solve. This object needs to define
        at least the function returning the value of the memory kernel
        given the current correlator values and will be repeatedly
        called during the iteration for each time step.
    base : correlator, optional
        In case of models that refer to underlying base models, the
        memory-kernel implementation usually assumes the underlying model
        to have been solved one the same time grid. This parameter
        then should point to the underlying correlator for that model,
        and the values of `h`, `blocksize`, and `blocks` will be copied
        from there instead of taking the ones supplied here.

    Notes
    -----
    The accuracy of the solution is determined primarily by the
    parameters `blocksize` and `h`.

    The methods defined here for the solver will usually not be called
    directly, but from the `solve_all` method of the correlator stack.
    The `correlator` object will then be filled with the solution arrays.

    If `store` is `True`, the fields `t`, `phi`, and `m` will be set
    to contain the solutions. They have shape (T,M) for scalar models,
    and shape (T,M,S,S) for matrix-valued models.
    """
    def __init__ (self, blocksize=256, h=1e-9, blocks=60, Tend=0.0,
                  maxinit=50, maxiter=10000, accuracy=1e-9, store=False,
                  model = model_base, base = None,
                  motion_type = "brownian"):
        if base is None:
            self.h0 = h
            self.halfblocksize = (blocksize//4)*2
            self.blocksize = self.halfblocksize * 2
            if Tend > 0.:
                self.blocks = 1 + int(np.ceil(np.log(Tend/self.h0/(self.blocksize-1))/np.log(2.)))
                print("adjusting to",self.blocks,"blocks")
            else:
                self.blocks = blocks
        else:
            self.h0 = base.h0
            self.blocks = base.blocks
            self.blocksize = base.blocksize
            self.halfblocksize = blocksize//2
            #self.base = base
        self.h = self.h0
        self.maxinit = maxinit
        self.maxiter = maxiter
        self.accuracy = accuracy

        self.model = model
        self.jit_kernel = model.get_kernel()

        self.store = store
        self.solved = -2

        if 'fixed_motion_type' in dir(model):
            self.motion_type = model.fixed_motion_type
            if not self.motion_type == motion_type:
                print("adjusting to",self.motion_type,"dynamics")
        else:
            self.motion_type = motion_type
        if self.motion_type == 'brownian':
            self.brownian =  True
            self.damped = False
        elif self.motion_type == 'newtonian':
            self.brownian = False
            self.damped = False
        elif self.motion_type == 'damped_newtonian':
            self.brownian = False
            self.damped = True
        else:
            raise NotImplementedError

        self.__alloc__ ()

    def __alloc__ (self):
        model = self.model
        self.mdimen = len(model)
        self.dim = model.matrix_dimension()
        self.phi_ = np.zeros((self.blocksize,self.mdimen*self.dim**2),dtype=model.dtype)
        self.m_ = np.zeros((self.blocksize,self.mdimen*self.dim**2),dtype=model.dtype)
        self.dPhi_ = np.zeros((self.halfblocksize+1,self.mdimen*self.dim**2),dtype=model.dtype)
        self.dM_ = np.zeros((self.halfblocksize+1,self.mdimen*self.dim**2),dtype=model.dtype)
        if self.store:
            self.t = np.zeros(self.halfblocksize*(self.blocks+1))
            if model.scalar():
                self.phi = np.zeros((self.halfblocksize*(self.blocks+1),self.mdimen*self.dim**2),dtype=model.dtype)
                self.m = np.zeros((self.halfblocksize*(self.blocks+1),self.mdimen*self.dim**2),dtype=model.dtype)
            else:
                self.phi = np.zeros((self.halfblocksize*(self.blocks+1),self.mdimen,self.dim,self.dim),dtype=model.dtype)
                self.m = np.zeros((self.halfblocksize*(self.blocks+1),self.mdimen,self.dim,self.dim),dtype=model.dtype)



    def initial_values (self, imax):
        """Initialize with short-time expansion.

        Parameters
        ----------
        imax : int
            Number of time steps to initialize. Will be adjusted to be
            less than half the blocksize.
        """
        iend = imax
        if (iend >= self.halfblocksize): iend = self.halfblocksize-1
        phi0 = self.model.phi0()
        self.model.set_base(self.phi_)
        if self.model.scalar():
            if not self.brownian:
                Aq = self.model.Aq()
                if self.damped:
                    phi0d = self.model.phi0d()
                    tauinv = phi0d - self.model.Bq() * phi0d / (2*Aq)
                else:
                    tauinv = 0.
                omega = self.model.Wq() * phi0 / Aq
                for i in range(iend):
                    t = i*self.h0
                    self.phi_[i] = phi0 + tauinv * t - omega * t*t/2
                    self.jit_kernel (self.m_[i], self.phi_[i], i, t,
                                     *self.model.kernel_extra_args())
            else:
                tauinv = self.model.Wq() * phi0 / self.model.Bq()
                for i in range(iend):
                    t = i*self.h0
                    self.phi_[i] = phi0 - tauinv * t
                    self.jit_kernel (self.m_[i], self.phi_[i], i, t,
                                     *self.model.kernel_extra_args())
        else:
            if not self.brownian:
                Aqinv = self.model.Aqinv()
                if self.damped:
                    phi0d = self.model.phi0d()
                    Bq = self.model.Bq()
                    tauinv = np.zeros_like(phi0,dtype=self.model.dtype)
                    for q in range(self.mdimen):
                        tauinv[q] = phi0d - Aqinv[q] @ Bq[q] @ phi0d/2
                else:
                    tauinv = 0.
                WqSq = self.model.WqSq()
                omega = np.zeros_like(phi0,dtype=self.model.dtype)
                for q in range(self.mdimen):
                    omega[q] = Aqinv[q] @ WqSq[q]
                for i in range(iend):
                    t = i*self.h0
                    self.phi_[i] = (phi0 + tauinv*t - omega*t*t/2).reshape(-1)
                    self.jit_kernel (self.m_[i].reshape(-1,self.dim,self.dim),
                                     self.phi_[i].reshape(-1,self.dim,self.dim),
                                     i, t, *self.model.kernel_extra_args())
            else:
                tauinv = np.zeros_like(phi0,dtype=self.model.dtype)
                Bqinv = self.model.Bqinv()
                WqSq = self.model.WqSq()
                for q in range(self.mdimen):
                    tauinv[q] = Bqinv[q] @ WqSq[q]
                for i in range(iend):
                    t = i*self.h0
                    self.phi_[i] = (phi0 - tauinv * t).reshape(-1)
                    self.jit_kernel (self.m_[i].reshape(-1,self.dim,self.dim),
                                     self.phi_[i].reshape(-1,self.dim,self.dim),
                                     i, t, *self.model.kernel_extra_args())
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def decimize (self):
        _decimize (self.phi_, self.m_, self.dPhi_, self.dM_, self.blocksize)
        self.h = self.h * 2.0

    # core interface, can be reimplemented by derived solvers
    # this is not safe to be reused with save/restore functionality
    def solve_block (self, istart, iend):
        self.model.set_base(self.phi_)
        if 'kernel_prefactor' in dir(self.model):
            pre_m = self.model.kernel_prefactor(np.arange(istart,iend)*self.h)
        else:
            pre_m = np.ones_like(range(istart,iend),dtype=self.model.dtype)
        if self.brownian:
            Aq = None
        else:
            if self.model.scalar():
                Aq = self.model.Aq()
            else:
                Aq = self.model.Aq().reshape(-1,self.dim,self.dim)
        if self.brownian or self.damped:
            if self.model.scalar():
                Bq = self.model.Bq()
            else:
                Bq = self.model.Bq().reshape(-1,self.dim,self.dim)
        else:
            Bq = None
        Xq = self.model.hopping()

        if self.model.scalar():
            if Aq is None:
                Aq = 0.
            if Bq is None:
                Bq = 0.
            if Xq is None:
                Xq = 0.
            _solve_block (istart, iend, self.h, Aq, Bq, self.model.Wq(), Xq, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy,(istart<self.blocksize//2), not (Xq is None), pre_m, *self.model.kernel_extra_args())
        else:
            if 'kernel_prefactor' in dir(self.model):
                raise NotImplementedError
            if Aq is None:
                Aq = np.zeros((1,self.dim,self.dim), dtype=self.model.dtype)
            if Bq is None:
                Bq = np.zeros((1,self.dim,self.dim), dtype=self.model.dtype)
            if Xq is None:
                Xq = np.zeros((1,self.dim,self.dim), dtype=self.model.dtype)
            _solve_block_mat (istart, iend, self.h, Aq, Bq, self.model.Wq().reshape(-1,self.dim,self.dim), Xq, self.phi_.reshape(-1,self.mdimen,self.dim,self.dim), self.m_.reshape(-1,self.mdimen,self.dim,self.dim), self.dPhi_.reshape(-1,self.mdimen,self.dim,self.dim), self.dM_.reshape(-1,self.mdimen,self.dim,self.dim), self.mdimen, self.jit_kernel, self.maxiter, self.accuracy,(istart<self.blocksize//2),not self.brownian,(self.damped or self.brownian),not (Xq is None),*self.model.kernel_extra_args())

    # new interface with reconstruction of already solved cases
    # does not call the jitted _solve_block directly because
    # for example the MSD solver wants to put its own implementation there
    def solve_first (self):
        """Solver interface, solve first half of time-domain block.

        This should be the first method called to solve the correlators.
        It initializes a given number of points from a short-time expansion,
        and fills the first half of the first block by calling the solver.

        Notes
        -----
        If `store` is True, the solver keeps track of whether the solutions
        are already in memory, and only calls the `solve_block` if needed.
        """
        self.h = self.h0
        if not self.store or not self.solved >= -1:
            self.initial_values (self.maxinit)
            self.solve_block (self.iend, self.halfblocksize)
            if self.store:
                N2 = self.halfblocksize
                self.t[:N2] = self.h * np.arange(N2)
                if self.model.scalar():
                    self.phi[:N2,:] = self.phi_[:N2,:]
                    self.m[:N2,:] = self.m_[:N2,:]
                else:
                    self.phi[:N2,:] = self.phi_[:N2,:].reshape(N2,self.mdimen,self.dim,self.dim)
                    self.m[:N2,:] = self.m_[:N2,:].reshape(N2,self.mdimen,self.dim,self.dim)
            self.solved = -1
        else:
            N2 = self.halfblocksize
            if self.model.scalar():
                self.phi_[:N2,:] = self.phi[:N2,:]
                self.m_[:N2,:] = self.m[:N2,:]
            else:
                self.phi_[:N2,:] = self.phi[:N2,:].reshape(N2,self.mdimen*self.dim**2)
                self.m_[:N2,:] = self.m[:N2,:].reshape(N2,self.mdimen*self.dim**2)
            self.dPhi_[1:N2,:] = ((self.phi_[:N2-1,:] + self.phi_[1:N2,:])/2.)
            self.dM_[1:N2,:] = ((self.m_[:N2-1,:] + self.m_[1:N2,:])/2.)
    def solve_next (self, d):
        """Solver interface, solve in a given block.

        This assumes that the first half of the block has been pre-filled
        will valid solutions, and then completes the solutions in the block.
        The assumption will be valid if `solve_first` has been called before.

        Notes
        -----
        If `store` is True, the solver keeps track of whether the solutions
        are already in memory, and only calls the `solve_block` if needed.
        """
        if not self.store or not self.solved >= d:
            self.solve_block (self.halfblocksize, self.blocksize)
            if self.store:
                N2 = self.halfblocksize
                N = self.blocksize
                self.t[d*N2+N2:d*N2+N] = self.h * np.arange(N2,N)
                if self.model.scalar():
                    self.phi[d*N2+N2:d*N2+N,:] = self.phi_[N2:,:]
                    self.m[d*N2+N2:d*N2+N,:] = self.m_[N2:,:]
                else:
                    self.phi[d*N2+N2:d*N2+N,:] = self.phi_[N2:,:].reshape(N2,self.mdimen,self.dim,self.dim)
                    self.m[d*N2+N2:d*N2+N,:] = self.m_[N2:,:].reshape(N2,self.mdimen,self.dim,self.dim)
            self.solved = d
        else:
            N2 = self.halfblocksize
            N = self.blocksize
            if self.model.scalar():
                self.phi_[N2:,:] = self.phi[d*N2+N2:d*N2+N,:]
                self.m_[N2:,:] = self.m[d*N2+N2:d*N2+N,:]
            else:
                self.phi_[N2:,:] = self.phi[d*N2+N2:d*N2+N,:].reshape(N2,self.mdimen*self.dim**2)
                self.m_[N2:,:] = self.m[d*N2+N2:d*N2+N,:].reshape(N2,self.mdimen*self.dim**2)

    def shear_modulus (self, **kwargs):
        """Return the dynamical shear modulus evaluated from the solution.

        If the solutions have been calculated and stored, and the model
        defines a `shear_modulus` method, call this.

        Parameters
        ----------
        **kwargs : dict
            Optional model-dependent arguments.

        Returns
        -------
        Gt : array_like
            The shear modulus such that integration over time results in
            the stress (tensor).
        """
        if self.store and self.solved>0 and 'shear_modulus' in dir(self.model):
            return self.model.shear_modulus(self.phi, self.t, **kwargs)

    def save (self, file):
        """Save the correlator data to the given file.

        The data will be stored in the mcspy h5 file format.

        Parameters
        ----------
        file : str
            Filename.
        """
        if not self.store or not self.solved > -2:
            return # should throw an exception
        with h5py.File(file, 'w') as f:
            self.h5save (f)
            self.model.h5save (f)
    def h5save (self, base):
        grp = base.create_group("correlator")
        grp.attrs['type'] = self.type()
        grp2 = grp.create_group("time_domain")
        grp2.create_dataset("t",data=self.t)
        grp2.create_dataset("phi",data=self.phi)
        grp2.create_dataset("m",data=self.m)
        grp2.attrs['solver'] = 'moment'
        grp2.attrs['h0'] = self.h0
        grp2.attrs['blocks'] = self.blocks
        grp2.attrs['solved_blocks'] = self.solved
        grp2.attrs['blocksize'] = self.blocksize
        grp2.attrs['maxiter'] = self.maxiter
        grp2.attrs['accuracy'] = self.accuracy
    def type (self):
        return 'phi'
    @staticmethod
    def load (file):
        """Load correlator from h5 file.

        This is a static method that returns a newly created
        correlator object holding data read from disk. It can be
        used as a base correlator object for the calculation of
        models that need the loaded data as a reference.
        """
        with h5py.File(file, 'r') as f:
            attrs = f['correlator/time_domain'].attrs
            newself = correlator(
                blocksize=attrs['blocksize'], h=attrs['h0'],
                blocks=attrs['blocks'], maxiter=attrs['maxiter'],
                accuracy=attrs['accuracy'], model = loaded_model(f))
            newself.t = np.array(f['correlator/time_domain/t'])
            newself.phi = np.array(f['correlator/time_domain/phi'], dtype=newself.model.dtype)
            newself.m = np.array(f['correlator/time_domain/m'], dtype=newself.model.dtype)
            newself.store = True
            newself.solved = attrs['solved_blocks']
        return newself

@njit
def _msd_solve_block (istart, iend, h, nutmp, phi, m, dPhi, dM, kernel, maxiter, accuracy, calc_moments, *kernel_args):
    A = dM[1] + 1.5*nutmp

    for i in range(istart,iend):
        kernel (m[i], None, i, h*i, *kernel_args)

        ibar = i//2
        C = (m[i]-m[i-1])*dPhi[1] - phi[i-1]*dM[1]
        for k in range(2,ibar+1):
            C += (m[i-k+1] - m[i-k]) * dPhi[k]
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        if (i-ibar > ibar):
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        C += m[i-ibar] * phi[ibar]
        C += (-2.0*phi[i-1] + 0.5*phi[i-2]) * nutmp
        C += -6.0
        C = C/A

        phi[i] = - C
        if calc_moments:
            dPhi[i] = 0.5 * (phi[i-1] + phi[i])
            dM[i] = 0.5 * (m[i-1] + m[i])


class mean_squared_displacement (correlator):
    """Solver for the mean-squared displacement.
    """
    def initial_values (self, imax=50):
        iend = imax
        if (iend >= self.halfblocksize): iend = self.halfblocksize-1
        self.model.set_base(self.phi_)
        for i in range(iend):
            t = i*self.h0
            self.phi_[i] = np.ones(self.mdimen,dtype=self.model.dtype)*6.*t/self.model.Bq()
            self.jit_kernel (self.m_[i], None, i, t, *self.model.kernel_extra_args())
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def solve_block (self, istart, iend):
        _msd_solve_block (istart, iend, self.h, self.model.Bq()/self.h, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy, (istart<self.blocksize//2), *self.model.kernel_extra_args())

    def type (self):
        return 'MSD'


@njit
def _ngp_solve_block (istart, iend, h, nutmp, phi, m, dPhi, dM, kernel, maxiter, accuracy, calc_moments, *kernel_args):
    A = dM[1] + 1.5*nutmp

    for i in range(istart,iend):
        kernel (m[i], None, i, h*i, *kernel_args)

        ibar = i//2
        # this is 2-dim, calculates m*a and m2*dr2
        # where a is (1+a2)dr2^2
        # we assume phi[i,1] to be pre-filled
        C = (m[i]-m[i-1])*dPhi[1] - phi[i-1]*dM[1]
        C[1] += phi[i,1]*dM[1,1]
        for k in range(2,ibar+1):
            C += (m[i-k+1] - m[i-k]) * dPhi[k]
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        if (i-ibar > ibar):
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        C += m[i-ibar] * phi[ibar]
        C[0] += (-2.0*phi[i-1,0] + 0.5*phi[i-2,0]) * nutmp[0]
        C[0] += -12.0 * phi[i,1] - 6.0 * C[1]
        C = C/A

        phi[i,0] = - C[0]
        if calc_moments:
            dPhi[i] = 0.5 * (phi[i-1] + phi[i])
            dM[i] = 0.5 * (m[i-1] + m[i])

class non_gaussian_parameter (correlator):
    """Solver for the non-Gaussian parameter.
    """

    def initial_values (self, imax=50):
        iend = imax
        if (iend >= self.halfblocksize): iend = self.halfblocksize-1
        self.model.set_base(self.phi_)
        for i in range(iend):
            t = i*self.h0
            self.phi_[i] = np.zeros(self.mdimen, dtype=self.model.dtype)
            self.phi_[i,1] = self.model.phi2()[i,0]
            self.jit_kernel (self.m_[i], None, i, t, *self.model.kernel_extra_args())
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def solve_block (self, istart, iend):
        r"""
        Note
        ----
        This solves for :math:`a(t)=(1+\alpha_2(t))\delta r^2(t)^2`
        in component 0. Component 1 of phi will be filled with the MSD.
        """
        self.phi_[istart:iend,1] = self.model.phi2()[istart:iend,0]
        _ngp_solve_block (istart, iend, self.h, self.model.Bq()/self.h, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy, (istart<self.blocksize//2), *self.model.kernel_extra_args())

    def type (self):
        return 'NGP'

