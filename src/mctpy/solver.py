import numpy as np

from numba import njit, literally

from .__util__ import model_base, np_isclose_all
from .__h5__ import load_model, Loadable
from .util import exponents, CorrelatorBase

@njit
def _decimize (f, dF, blocksize):
    halfblocksize = blocksize//2
    imid = halfblocksize//2
    for i in range(1,imid):
        di = i+i
        dF[i] = 0.5 * (dF[di-1] + dF[di])
    for i in range(imid,halfblocksize):
        di = i+i
        dF[i] = 0.25 * (f[di-2] + 2*f[di-1] + f[di])
    # TODO rethink, maybe accuracy can be improved here
    # if i==halfblocksize, di==blocksize, this is too much
    # but we could use a backwards integral rule utilizing
    # di-1 and di-2 instead of just di-1
    dF[halfblocksize] = f[blocksize-1]
    for i in range(halfblocksize):
        di = i+i
        f[i] = f[di]

@njit
def _solve_block (istart, iend, h, Aq, Bq, Wq, Xq, phi, m, dPhi, dM, kernel, maxiter, accuracy, calc_moments, hopping, pre_m, set_V, kernel_V_args, *kernel_args):

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
        set_V (i, *kernel_V_args)
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

@njit
def _solve_block_mat (istart, iend, h, Aq, Bq, Wq, Xq, phi, m, dPhi, dM, M, kernel, maxiter, accuracy, calc_moments, useA, useB, useX, set_V, kernel_V_args, *kernel_args):

    use_hopping = literally(useX)
    Ainv = np.zeros_like(Wq,dtype=phi.dtype)
    B = np.zeros_like(Wq,dtype=phi.dtype)
    C = np.zeros_like(Wq,dtype=phi.dtype)
    if use_hopping:
        Cd = np.zeros_like(Wq,dtype=phi.dtype)

    Ainv = Wq + dM[1] + Bq*1.5/h + Aq*2.0/h**2
    for q in range(M):
        if use_hopping:
            Ainv[q] += dM[1,q] @ Xq[q] * 0.5*h
        Ainv[q] = np.linalg.inv(Ainv[q])
        B[q] = -dPhi[1,q] + phi[0,q]
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
        if use_hopping:
            for q in range(M):
                Cd[q] = m[i-1,q] @ Xq[q] @ dPhi[1,q] + dM[1,q] @ Xq[q] @ phi[i-1,q]
            for k in range(2,ibar+1):
                for q in range(M):
                    Cd[q] += (m[i-k+1,q] + m[i-k,q]) @ Xq[q] @ dPhi[k,q] \
                          + dM[k,q] @ Xq[q] @ (phi[i-k+1,q] + phi[i-k,q])
            if (i-ibar > ibar):
                for q in range(M):
                    Cd[q] += dM[k,q] @ Xq[q] @ (phi[i-ibar,q] + phi[i-ibar-1,q])
            for q in range(M):
                C[q] += 0.5*h*Cd[q]

        for q in range(M):
            C[q] += m[i-ibar,q] @ phi[ibar,q]
            if useB:
                C[q] += Bq[q] @ (-2*phi[i-1,q] + 0.5*phi[i-2,q]) / h
            if useA:
                C[q] += Aq[q] @ (-5*phi[i-1,q] + 4*phi[i-2,q] - phi[i-3,q])/h**2
            C[q] = Ainv[q] @ C[q]

        iterations = 0
        converged = False
        newphi = phi[i-1].copy()
        #set_V (i, *kernel_V_args)
        while (not converged and iterations < maxiter):
            phi[i] = newphi
            kernel (m[i], phi[i], i, h*i, *kernel_args)
            for q in range(M):
                newphi[q] = Ainv[q] @ m[i,q] @ B[q] - C[q]
            iterations += 1
            if np_isclose_all(newphi.reshape(-1), phi[i].reshape(-1),
                          rtol=accuracy, atol=accuracy):
                converged = True
                phi[i,...] = newphi
                if calc_moments:
                    dPhi[i] = 0.5 * (phi[i-1] + phi[i])
                    dM[i] = 0.5 * (m[i-1] + m[i])

class correlator (CorrelatorBase, Loadable):
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
        Note that this defaults to `None` which will crash because
        apart from the initialization, the methods of this class assume
        `model` to be reasonably set. The default can be useful in derived
        classes though, for example for the beta-scaling equations.
    quiet : bool, default False
        If set, suppress printed warning messages (e.g. about adjustment
        of number of blocks, or type of dynamics supported by model).

    Notes
    -----
    The accuracy of the solution is determined primarily by the
    parameters `blocksize` and `h`.

    The methods defined here for the solver will usually not be called
    directly, but from the `solve_all` method of the correlator stack.
    The `correlator` object will then be filled with the solution arrays.

    If `store` is `True`, the fields `t`, `phi`, and `m` will be set
    to contain the solutions. They have shape (T,M) for scalar models,
    shape (T,M,V) for vectorial models, and shape (T,M,S,S) for matrix-valued
    models.

    :ivar shape: Shape of the correlation functions if stored.
    :ivar phishape: Internal shape of the memory for the solver.
    :ivar model: Instance of the model being solved.
    """
    def __init__ (self, blocksize=256, h=1e-9, blocks=60, Tend=0.0,
                  maxinit=50, maxiter=10000, accuracy=1e-9, store=False,
                  model = None, motion_type = "brownian", quiet=False):

        if model is None or not model.get_bases():
            self.h0 = h
            self.halfblocksize = (blocksize//4)*2
            self.blocksize = self.halfblocksize * 2
            if Tend > 0.:
                self.blocks = 1 + int(np.ceil(np.log(Tend/self.h0/(self.blocksize-1))/np.log(2.)))
                if not quiet:
                    print("adjusting to",self.blocks,"blocks")
            else:
                self.blocks = blocks
        else:
            for base in model.get_bases():
                if not base.correlator:
                    # TODO could auto-create a suitable solver here
                    # would need to give each solver a type ("phi", "msd", ...)
                    # and each model a corresponding type (have that already?)
                    raise
                # TODO check if multiple bases define conflicting values
                self.h0 = base.correlator.h0
                self.blocks = base.correlator.blocks
                self.blocksize = base.correlator.blocksize
                self.halfblocksize = blocksize//2
        self.h = self.h0
        self.maxinit = maxinit
        self.maxiter = maxiter
        self.accuracy = accuracy

        self.model = model
        if model is not None:
            self.model.correlator = self
            self.jit_kernel = model.get_kernel()
            self.jit_set_V = model.get_set_V()

        self.store = store
        self.solved = -2

        self.__alloc__ ()

        if model is not None:
            if 'fixed_motion_type' in dir(model):
                self.motion_type = model.fixed_motion_type
                if not self.motion_type == motion_type and not quiet:
                    print("adjusting to",self.motion_type,"dynamics")
            else:
                self.motion_type = motion_type
            self.prefactors = model.prefactors(self.motion_type)

    def __alloc__ (self):
        if self.model is None:
            return
        model = self.model
        self.mdimen = len(model)
        self.vdim = model.vector_dimension()
        self.dim = model.matrix_dimension()
        if model.scalar():
            self.shape = (self.mdimen * self.vdim,)
            if self.vdim > 1:
                self.phishape = (self.mdimen, self.vdim)
            else:
                self.phishape = self.shape
        else:
            self.shape = (self.mdimen * self.vdim, self.dim, self.dim)
            if self.vdim > 1:
                self.phishape = (self.mdimen, self.vdim, self.dim, self.dim)
            else:
                self.phishape = self.shape
        self.phi_ = np.zeros((self.blocksize,*self.shape),dtype=model.dtype)
        self.m_ = np.zeros((self.blocksize,*self.shape),dtype=model.dtype)
        self.dPhi_ = np.zeros((self.halfblocksize+1,*self.shape),dtype=model.dtype)
        self.dM_ = np.zeros((self.halfblocksize+1,*self.shape),dtype=model.dtype)
        self.model.phi = self.phi_
        if self.store:
            self.t = np.zeros(self.halfblocksize*(self.blocks+1))
            self.phi = np.zeros((self.halfblocksize*(self.blocks+1),*self.phishape),dtype=model.dtype)
            self.m = np.zeros((self.halfblocksize*(self.blocks+1),*self.phishape),dtype=model.dtype)


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
        #self.model.set_base_phi(self.phi_) # TODO FIXME check MSD
        scalar = self.model.scalar()
        phi0 = self.prefactors['phi0']
        WqSq = self.prefactors['WqSq']
        if 'Aq' in self.prefactors or 'Aqinv' in self.prefactors:
            # we have a second-derivative term
            Aqinv = self.prefactors['Aqinv']
            if 'Bq' in self.prefactors or 'Bqinv' in self.prefactors:
                # we have a damping term as well
                phi0d = self.prefactors['phi0d']
                Bq = self.prefactors['Bq']
                if scalar:
                    tauinv = phi0d - Bq*phi0d * Aqinv / 2
                else:
                    tauinv = np.zeros(self.shape,dtype=self.model.dtype)
                    for q in range(self.shape[0]):
                        tauinv[q] = phi0d - Aqinv[q] @ Bq[q] @ phi0d/2
            else:
                tauinv = 0.
            if scalar:
                omega = Aqinv * WqSq
            else:
                omega = np.zeros(self.shape,dtype=self.model.dtype)
                for q in range(self.mdimen):
                    omega[q] = Aqinv[q] @ WqSq[q]
            for i in range(iend):
                t = i*self.h0
                self.phi_[i] = phi0 + tauinv * t - omega * t*t/2
                self.jit_set_V (i, *self.model.set_V_args())
                self.jit_kernel (self.m_[i], self.phi_[i], i, t,
                                 *self.model.kernel_extra_args())
        else: # no second derivative
            Bqinv = self.prefactors['Bqinv']
            if Bqinv is None:
                raise NotImplementedError
            if scalar:
                tauinv = Bqinv * WqSq
            else:
                tauinv = np.zeros(self.shape,dtype=self.model.dtype)
                for q in range(self.shape[0]):
                    tauinv[q] = Bqinv[q] @ WqSq[q]
            for i in range(iend):
                t = i*self.h0
                self.phi_[i] = phi0 - tauinv * t
                self.jit_set_V (i, *self.model.set_V_args())
                self.jit_kernel (self.m_[i], self.phi_[i], i, t,
                                 *self.model.kernel_extra_args())
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def decimize (self):
        _decimize (self.phi_, self.dPhi_, self.blocksize)
        if 'm' in self.__dict__:
            _decimize (self.m_, self.dM_, self.blocksize)
        self.h = self.h * 2.0

    # core interface, can be reimplemented by derived solvers
    # this is not safe to be reused with save/restore functionality
    def solve_block (self, istart, iend):
        #self.model.set_base_phi(self.phi_) # TODO FIXME REDUNDANT?
        if 'kernel_prefactor' in dir(self.model):
            pre_m = self.model.kernel_prefactor(np.arange(istart,iend)*self.h)
        else:
            pre_m = np.ones_like(range(istart,iend),dtype=self.model.dtype)

        if self.model.scalar():
            Aq = self.prefactors.get('Aq',0.)
            Bq = self.prefactors.get('Bq',0.)
            Xq = self.prefactors.get('Xq',0.)
            Wq = self.prefactors.get('Wq',0.)
            if not 'Xq' in self.prefactors:
                use_X = False
            else:
                use_X = True
            _solve_block (istart, iend, self.h, Aq, Bq, Wq, Xq, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy,(istart<self.blocksize//2), use_X, pre_m, self.jit_set_V, self.model.set_V_args(), *self.model.kernel_extra_args())
        else:
            if 'kernel_prefactor' in dir(self.model):
                raise NotImplementedError
            z = np.zeros((1,self.dim,self.dim),dtype=self.model.dtype)
            Aq = self.prefactors.get('Aq',z)
            Bq = self.prefactors.get('Bq',z)
            Xq = self.prefactors.get('Xq',z)
            if not 'Xq' in self.prefactors:
                use_X = False
            else:
                use_X = True
            Wq = self.prefactors.get('Wq',z)
            _solve_block_mat (istart, iend, self.h, Aq, Bq, Wq, Xq, self.phi_, self.m_, self.dPhi_, self.dM_, self.mdimen, self.jit_kernel, self.maxiter, self.accuracy,(istart<self.blocksize//2),not self.brownian,(self.damped or self.brownian),use_X,self.jit_set_V, self.model.set_V_args(), *self.model.kernel_extra_args())

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
        #if self.model is not None:
        #    self.model.set_base_phi(self.phi_)
        if not self.store or not self.solved >= -1:
            self.initial_values (self.maxinit)
            self.solve_block (self.iend, self.halfblocksize)
            if self.store:
                N2 = self.halfblocksize
                self.t[:N2] = self.h * np.arange(N2)
                self.phi[:N2,:] = self.phi_[:N2,:].reshape(N2,*self.phishape)
                if 'm' in self.__dict__:
                    self.m[:N2,:] = self.m_[:N2,:].reshape(N2,*self.phishape)
            self.solved = -1
        else:
            N2 = self.halfblocksize
            self.phi_[:N2,:] = self.phi[:N2,:].reshape(N2,*self.shape)
            self.dPhi_[1:N2,:] = ((self.phi_[:N2-1,:] + self.phi_[1:N2,:])/2.)
            if 'm' in self.__dict__:
                self.m_[:N2,:] = self.m[:N2,:].reshape(N2,*self.shape)
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
        # TODO check this should be sufficient once at the beginning?!
        # looks almost redundant since we now also set self.model.correlator
        # to ourselves, but do we want the field phi_ to be exposed
        # in the API, or what?
        # FIXME what is currently completely missing is a compatility check
        # for solvers
        ##if self.model is not None:
        ##    self.model.set_base_phi(self.phi_)
        if not self.store or not self.solved >= d:
            self.solve_block (self.halfblocksize, self.blocksize)
            if self.store:
                N2 = self.halfblocksize
                N = self.blocksize
                self.t[d*N2+N2:d*N2+N] = self.h * np.arange(N2,N)
                self.phi[d*N2+N2:d*N2+N,:] = self.phi_[N2:,:].reshape(N2,*self.phishape)
                if 'm' in self.__dict__:
                    self.m[d*N2+N2:d*N2+N,:] = self.m_[N2:,:].reshape(N2,*self.phishape)
            self.solved = d
        else:
            N2 = self.halfblocksize
            N = self.blocksize
            self.phi_[N2:,:] = self.phi[d*N2+N2:d*N2+N,:].reshape(N2,*self.shape)
            if 'm' in self.__dict__:
                self.m_[N2:,:] = self.m[d*N2+N2:d*N2+N,:].reshape(N2,*self.shape)

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
        G : array_like
            The shear modulus such that integration over time results in
            the stress (tensor).
        """
        if self.store and self.solved>0 and 'shear_modulus' in dir(self.model):
            return self.model.shear_modulus(self.phi, self.t, **kwargs)
        else:
            return None
    def longitudinal_modulus (self, **kwargs):
        """Returns the longitudinal modulus evaluate from the solution.

        This works if the solutions have been calculated and stored,
        and the model defines a `longitudinal_modulus` method.

        Parameters
        ----------
        **kwargs : dict
            Optional model-dependent arguments.

        Returns
        -------
        M : array_like
            The dynamical longitudinal modulus.
        """
        if self.store and self.solved>0 and 'longitudinal_modulus' in dir(self.model):
            return self.model.longitudinal_modulus(self.phi, self.t, **kwargs)
        else:
            return None

    def h5save (self, h5file):
        grp = h5file.create_group('correlator')
        grp.attrs['type'] = self.type()
        grp2 = grp.create_group("time_domain")
        if self.store and self.solved > -2:
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
        self.model.h5save (grp)
    def type (self):
        return 'phi'
    @classmethod
    def h5load (cls, h5file):
        """Load correlator from h5 file.

        This is a class method that returns a newly created
        correlator object holding data read from disk. It can be
        used as a base correlator object for the calculation of
        models that need the loaded data as a reference.
        """
        h5grp = h5file['correlator']
        attrs = h5grp['time_domain'].attrs
        model = load_model(h5grp)
        if model is not None:
            dtype = model.dtype
        else:
            dtype = np.float64
        newself = cls(
            blocksize=attrs['blocksize'], h=attrs['h0'],
            blocks=attrs['blocks'], maxiter=attrs['maxiter'],
            accuracy=attrs['accuracy'], model = model)
        # TODO avoid errors if empty correlator saved
        newself.t = np.array(h5grp['time_domain/t'])
        newself.phi = np.array(h5grp['time_domain/phi'], dtype=dtype)
        newself.m = np.array(h5grp['time_domain/m'], dtype=dtype)
        newself.store = True
        newself.solved = attrs['solved_blocks']
        return newself

@njit
def _msd_solve_block (istart, iend, h, nutmp, phi, m, dPhi, dM, kernel, d, maxiter, accuracy, calc_moments, *kernel_args):
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
        C += -2*d
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
        self.model.set_base_phi(self.phi_) # TODO FIXME
        # TODO FIXME different motion types!
        for i in range(iend):
            t = i*self.h0
            self.phi_[i] = np.ones(self.mdimen*self.vdim,dtype=self.model.dtype)*2*self.model.spatial_dimension()*t/self.model.Bq()
            self.jit_kernel (self.m_[i], None, i, t, *self.model.kernel_extra_args())
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def solve_block (self, istart, iend):
        _msd_solve_block (istart, iend, self.h, self.model.Bq()/self.h, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.model.spatial_dimension(), self.maxiter, self.accuracy, (istart<self.blocksize//2), *self.model.kernel_extra_args())

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
        self.model.set_base_phi(self.phi_) # TODO FIXME
        # TODO FIXME different motion types
        for i in range(iend):
            t = i*self.h0
            self.phi_[i] = np.zeros(self.mdimen*self.vdim, dtype=self.model.dtype)
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

