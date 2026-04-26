import numpy as np
from .simple_liquid import simpleLiquidSq

from importlib import resources
from scipy.interpolate import CubicSpline

class gcm4MSA (simpleLiquidSq):
    """MSA structure factor for the GCM4 model.

    The generalized Gaussian core model GCM4 sets
    V(r) = epsilon exp(-(r/\sigma)^4).
    Within MSA the direct correlation function is directly  related
    to the Fourier transform of the potential which can be
    calculated explicitly here in terms of generalized hypergeometric
    functions. This class uses pre-calculated interpolation data
    for sigma=1.
    """
    def __init__ (self, rho, epsilon=1.0):
        """
        Parameters
        ----------
        rho : float
            Number density.
        epsilon : float, optional (default = 1.0)
            Strength of the potential, plays the role of inverse temperature.

        Notes
        -----
        This relies on a pre-calculated file to load interpolation data
        in the form (q,V(q),V'(q)).
        """
        self.rho = rho
        self.eps = epsilon
        with resources.path('mctpy','structurefactors/data/gcm4MSA.npz') as file:
            _vqdata = np.load(file)['gcm4vq']
        self._vq = CubicSpline(_vqdata[:,0], _vqdata[:,1])
        self._vqd = CubicSpline(_vqdata[:,0], _vqdata[:,2])
    def cq (self, q):
        return -self.eps * self._vq(q)
    def dcq_dq (self, q):
        return -self.eps * self._vqd(q)
