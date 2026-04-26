import numpy as np
import scipy
from .simple_liquid import simpleLiquidSq

class lgDCF (simpleLiquidSq):
    """Lorentz-gas direct correlation function.
    """
    def __init__ (self, R):
        self.R = R
        self.lowq = 0.5/R
        self.rho = 0.
    def _cq_high (self, q):
        x = q*self.R
        return 4.*np.pi*self.R**2/q * scipy.special.spherical_jn(1,x)
    def _cq_low (self, q):
        x = q*self.R
        x2 = x*x
        return 4.*np.pi*self.R**3 * (1/3. - 1./30*x2 + 1./840*x2*x2)
