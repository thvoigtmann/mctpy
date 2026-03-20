import numpy as np
import scipy

class hssFMT2d (object):
    """Structure factor for 2d hard disks, fundamental measure theory (FMT).

    This implements the expression derived by Thorneywork et al (2018).
    """
    def __init__ (self, eta):
        self.eta = eta
    def density (self):
        return self.eta*4/np.pi
    def cq (self, q):
        """Return direct correlation function.

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where the DCF should be evaluated.

        Returns
        -------
        cq : array_like
            DCF evaluated on the given grid.
        """
        etacmp = (1-self.eta)**2
        cq_ = np.zeros_like(q)
        regularq = q>np.finfo(float).eps
        q_ = q[regularq]
        j0 = scipy.special.j0(q_/2)
        j1 = scipy.special.j1(q_/2)
        cq_[regularq] = (-(5./4)*etacmp*(q_*j0)**2 \
            + (4*((self.eta-20)*self.eta+7)+(5./4)*etacmp*q_**2)*j1**2 \
            + 2*(self.eta-13)*(1-self.eta)*q_*j1*j0) \
            * np.pi/(6*q_*q_*(1-self.eta)**3)
        cq_[~regularq] = -(np.pi/4)*(4-3*self.eta+self.eta**2)/(1-self.eta)**3
        return cq_
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
