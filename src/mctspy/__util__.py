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
    def __init__ (self):
        self.dtype = np.dtype(float)
    def __len__ (self):
        return 1
    def matrix_dimension (self):
        return 1
    def scalar (self):
        return True
    def hopping (self):
        return None
    def phi0 (self):
        return np.ones(len(self))
    def phi0d (self):
        return np.zeros(len(self))
    def Wq (self):
        return np.ones(len(self))
    def Aq (self):
        return np.ones(len(self))
    def Bq (self):
        return np.ones(len(self))

    def set_base (self, array):
        #self.phi = void(array)
        self.phi = array
    def cache (self):
        if 'base' in dir(self): return False
        return True

    def kernel_extra_args (self):
        return []

    def get_kernel (self):
        if not self.cache() or not '__m__' in dir(self):
            self.__m__ = self.make_kernel()
        return self.__m__
    def get_dm (self):
        if not self.cache() or not '__dm__' in dir(self):
            self.__dm__ = self.make_dm()
        return self.__dm__
    def get_dmhat (self):
        if not self.cache() or not '__dmhat__' in dir(self):
            self.__dmhat__ = self.make_dmhat()
        return self.__dmhat__
    def get_dm2 (self):
        if not self.cache() or not '__dm2__' in dir(self):
            self.__dm2__ = self.make_dm2()
        return self.__dm2__

    def make_kernel (self):
        @nb.njit
        def dummy(m, phi, i, t):
            return
        return dummy
    def make_dm (self):
        @nb.njit
        def dummy(m, phi, dphi):
            return
        return dummy
    def set_C (self, f):
        # can use this to pre-calculate stability matrix before make_dm
        return
    def make_dmhat(self):
        @nb.njit
        def dummy(m, f, ehat):
            return
        return dummy
    def make_dm2 (self):
        @nb.njit
        def dummy(m, phi, dphi):
            return
        return dummy

    def dq (self):
        return 1.0

    def h5save (self, fh):
        return

class loaded_model(model_base):
    def __init__ (self, h5data):
        model_definition = h5data['model']
        for attr,val in model_definition.attrs.items():
            self.__dict__[attr] = val
        for field,val in model_definition.items():
            self.__dict__[field] = np.array(val)
    def __len__ (self):
        return self.M

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
