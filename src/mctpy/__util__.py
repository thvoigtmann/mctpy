import numpy as np
import numba as nb

# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
# we need this to keep references in jit-compiled models that
# reference underlying data of a base model
@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """ returns a void pointer from a given memory address """
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

# take 3-tuple of addres, shape, dtype
@nb.njit
def nparray(ast,i=None):
    return nb.carray(address_as_void_pointer(ast[0]),ast[1],ast[2])

def void(nparray):
    return nparray.ctypes.data, nparray.shape, nparray.dtype


class model_base (object):
    def __init_subclass__ (cls, /, kernel=None, dm=None, dmhat=None, dm2=None,
                           setV=None, **kwargs):
        """Initialize the base features of any MCT model.
        Parameters:
        -----------
        kernel, dm, dmhat, dm2, setV:
            Should be njit-functions that will be called as the
            memory kernels. These function entry points will be
            returned by the `get_kernel` and related methods.
            If either one is not set, `get_kernel` returns the
            result of a call to the model's `make_kernel` method
            (and similar for the others).
            The setV method will be called just before the first
            memory-kernel evaluation, once for each time step. This
            is mostly used by tagged-particle models.
        """
        super().__init_subclass__(**kwargs)
        cls.dtype = np.dtype(float)
        cls.mdispatch = {}
        cls.set_kernels (kernel=kernel, dm=dm, dmhat=dmhat, dm2=dm2, setV=setV)
    def __init__ (self):
        # the following fields will be populated by the respective solvers
        # to allow back-linking of a model to the solver that solved it
        # this is needed for the construction of dependency lists
        # of coupled models
        self.correlator = None
        self.nep = None
        self.ev = None
    @classmethod
    def set_kernels (cls, kernel=None, dm=None, dmhat=None, dm2=None,
                     setV=None):
        if kernel:
            cls.mdispatch['kernel'] = kernel
        if dm:
            cls.mdispatch['dm'] = dm
        if dmhat:
            cls.mdispatch['dmhat'] = dmhat
        if dm2:
            cls.mdispatch['dm2'] = dm2
        if setV:
            cls.mdispatch['setV'] = setV
    def __len__ (self):
        """The "length" of a model is the number of correlators.
        Typically, they are numbered by wave-number index `q`."""
        return 1
    def vector_dimension (self):
        """Models can define vector-valued correlators, where the index
        `q` counting towards the "length" of the model corresponds to
        a vector, where multiplication in the equation of motion is
        defined element-wise."""
        return 1
    def matrix_dimension (self):
        """Models can define matrix-valued correlators. A matrix dimension
        of `S` will typically define `S*S` elements per length index, and
        multiplication in the equation of motion is standard matrix
        multiplication for any fixed `q`."""
        return 1
    def scalar (self):
        """Return True if the model obeys scalar equations, i.e., those
        where multiplication in the equations of motion is simply
        element-wise."""
        return True
    def spatial_dimension (self):
        """Return the spatial dimension of the model. This defaults to
        three since the original development of MCT was done for it;
        this influences the calculation of the MSD and NGP."""
        return 3
    def hopping (self):
        return None
    def phi0 (self):
        """Return initial value of the correlators of this model.
        Needs to be an array of shape (M*V,) for scalar models,
        where M is the number of q indices and V the vector dimension.
        For matrix models, needs to be of shape (M*V,S,S) or similar,
        but the implemented default here then won't do."""
        return np.ones(len(self)*self.vector_dimension())
    def phi0d (self):
        """Return initial derivative values for the correlators of this
        model. See `phi0()` for the expected array shapes."""
        return np.zeros(len(self)*self.vector_dimension())
    def Wq (self):
        """Return prefactor in front of the correlator in the
        equation of motion. See `phi0()` for the expected shape."""
        return np.ones(len(self)*self.vector_dimension())
    def Aq (self):
        """Return prefactor in front of the second derivative of the
        correlator in the equation of motion. See `phi0()` for the expected
        shape. Will only be called for non-Brownian models."""
        return np.ones(len(self)*self.vector_dimension())
    def Bq (self):
        """Return prefactor in front of the first derivative of the
        correlator in the equation of motion. See `phi0()` for the expected
        shape."""
        return np.ones(len(self)*self.vector_dimension())
    def get_bases (self):
        """Return the list of base correlators needed for this one."""
        return []
    def set_base_phi (self, array):
        # old style, baking pointers into njit functions (bad, but shorter
        # call signatures if that matters at all):
        #self.phi = void(array) # then to load with nparray(self.phi)
        self.phi = array

    def kernel_extra_args (self):
        return []
    def kernel_extra_args_dm (self):
        return self.kernel_extra_args()
    def kernel_extra_args_dmhat (self):
        return self.kernel_extra_args_dm()
    def kernel_extra_args_dm2 (self):
        return self.kernel_extra_args()
    def set_V_args (self):
        return ()

    def get_kernel (self):
        if not 'kernel' in self.mdispatch:
            self.mdispatch['kernel'] = self.make_kernel()
        return self.mdispatch['kernel']
    def get_dm (self):
        if not 'dm' in self.mdispatch:
            self.mdispatch['dm'] = self.make_dm()
        return self.mdispatch['dm']
    def get_dmhat (self):
        if not 'dmhat' in self.mdispatch:
            self.mdispatch['dmhat'] = self.make_dmhat()
        return self.mdispatch['dmhat']
    def get_dm2 (self):
        if not 'dm2' in self.mdispatch:
            self.mdispatch['dm2'] = self.make_dm2()
        return self.mdispatch['dm2']
    def get_set_V (self):
        if not 'setV' in self.mdispatch:
            self.mdispatch['setV'] = self.make_set_V()
        return self.mdispatch['setV']

    def m (self, f):
        r"""Short-hand for the memory function.
        For given `f`, returns `m(f)`, which will be a newly created
        array of the same shape as `f`.

        Note that this is not intended to be called by the solvers,
        who will pre-allocate arrays for the memory kernels and fill
        them in-place. This is just a convenience wrapper.

        Also, the semantics shall be that this returns the proper
        memory kernel, while the kernel functions used in the solver
        can split out prefactors that are absorbed in the other
        prefactors of the Mori-Zwanzig equation. A typical case is the
        factor :math:`1/q^2` the is omitted there but should be
        included here.
        """
        m_ = np.zeros_like (f)
        self.get_kernel() (m_, f, 0, 0, *self.kernel_extra_args())
        return self.mrenorm(m_)
    def mrenorm (self, m):
        r"""Return that proper memory kernel from a stored memory-kernel
        array as used by the solver. This is the place where any factors
        that were taken out need to be factored in again."""
        return m

    # TODO document how the signatures should in fact be
    def make_kernel (self):
        raise NotImplementedError
    def make_dm (self):
        raise NotImplementedError
    def make_dmhat(self):
        raise NotImplementedError
    def make_dm2 (self):
        raise NotImplementedError
    # these are allowed to fail silently
    def make_set_V (self):
        return model_base_set_V
    def set_C (self, f):
        # can use this to pre-calculate stability matrix before make_dm
        return

    def dq (self):
        """Return q-grid spacing.

        Should be the values that need to be multiplied into the sum
        when an integral over the q-indices is performed."""
        return 1.0

@nb.njit(inline='always')
def model_base_set_V (i):
    return


# below some support functions, mostly for numba

def _dq (q):
    return np.diff(q, append=2*q[-1]-q[-2])

@nb.njit
def np_gradient(f,k):
    df_dk = np.zeros_like(f)
    df_dk[1:-1] = (f[2:]-f[0:-2])/(k[2:]-k[0:-2])
    df_dk[0] = (f[1]-f[0])/(k[1]-k[0])
    df_dk[-1] = (f[-1]-f[-2])/(k[-1]-k[-2])
    return df_dk

# numba does not support np.isclose with complex arrays
# we need to patch around that
# note that we also include a test like np.isclose().all()
# since we need that, and we can overload to make sure it
# also works with scalars (needed in the regula_falsi when njit'ed)

def np_isclose_all_impl(a, b, rtol, atol):
    return np.isclose(a, b, rtol=rtol, atol=atol).all()

@nb.extending.overload(np_isclose_all_impl)
def np_isclose_all_impl_overload(a, b, rtol, atol):
    if not isinstance(a, nb.types.Array):
        if not a == nb.complex128:
            def np_isclose_scalar (a, b, rtol, atol):
                return np.isclose (a, b, rtol, atol)
            return np_isclose_scalar
        else:
            def np_isclose_scalar (a, b, rtol, atol):
                return np.isclose (a.real, b.real, rtol, atol) \
                     & np.isclose (a.imag, b.imag, rtol, atol)
            return np_isclose_scalar
    if a.dtype == nb.complex128:
        def np_isclose_complex(a, b, rtol, atol):
            return (np.isclose(a.real, b.real, rtol=rtol, atol=atol) \
                  & np.isclose(a.imag, b.imag, rtol=rtol, atol=atol)).all()
        return np_isclose_complex
    else:
        return np_isclose_all_impl

@nb.njit
def np_isclose_all(a, b, rtol=1e-5, atol=1e-8):
    return np_isclose_all_impl(a, b, rtol, atol)
