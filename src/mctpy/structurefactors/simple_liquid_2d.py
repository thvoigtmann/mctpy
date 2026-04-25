import numpy as np
import scipy
from .simple_liquid import simpleLiquidSq

class hssFMT2d (simpleLiquidSq):
    """Structure factor for 2d hard disks, fundamental measure theory (FMT).

    This implements the expression derived by Thorneywork et al (2018).
    """
    def __init__ (self, eta):
        self.eta = eta
        self.lowq = np.finfo(float).eps
    def density (self):
        return self.eta*4/np.pi
    def _cq_high (self, q):
        etacmp = (1-self.eta)**2
        j0 = scipy.special.j0(q/2)
        j1 = scipy.special.j1(q/2)
        return (-(5./4)*etacmp*(q*j0)**2 \
            + (4*((self.eta-20)*self.eta+7)+(5./4)*etacmp*q**2)*j1**2 \
            + 2*(self.eta-13)*(1-self.eta)*q*j1*j0) \
            * np.pi/(6*q*q*(1-self.eta)**3)
    def _cq_low (self, q):
        return -(np.pi/4)*(4-3*self.eta+self.eta**2)/(1-self.eta)**3
    def Sq (self, q):
        """Return the structure factor and DCF.

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where S(q) and DCF should be evaluated.

        Returns
        -------
        sq : array_like
            S(q) evaluated on the given grid.
        cq : array_like
            c(q) evaluated on the given grid.
        """
        cq_ = self.cq(q)
        return 1.0 / (1.0 - self.density() * cq_), cq_
