import numpy as np
import scipy

from .simple_liquid import simpleLiquidSq
from ..__h5__ import LoadableModel

class hssFMT2d (simpleLiquidSq, LoadableModel, role='structurefactor',
                attrs=['eta'], resetable=['eta']):
    """Structure factor for 2d hard disks, fundamental measure theory (FMT).

    This implements the expression derived by Thorneywork et al (2018).

    Parameters
    ----------
    eta : float
        Volume fraction of the 2d system.
    """
    def __init__ (self, eta):
        super().__init__()
        self.eta = eta
        self.reinit()
    def reinit (self):
        self.rho = self.eta*4/np.pi
        self.lowq = np.finfo(float).eps
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
