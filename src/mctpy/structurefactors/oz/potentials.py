import numpy as np

class Potential:
    def __call__ (self, r):
        """Return the potential evaluated at r."""
        raise NotImplementedError
    def exp (self, r, kT):
        """Return the Boltzmann factor exp(-kT U(r)) corresponding
        to the potential U(r)."""
        return np.exp(-self(r)/kT)

class HSS (Potential):
    """Hard-sphere potential.

    Parameters
    ----------
    sigma : float, default = 1.0
        Size of the hard-sphere core.
    """
    def __init__ (self, sigma=1.0):
        self.sigma = sigma
    def __call__ (self, r):
        phi = np.zeros_like(r)
        phi[r<self.sigma] = np.inf
        return phi
    def exp (self, r, kT):
        return np.array(r>self.sigma, dtype=np.float64)

class LJ (Potential):
    """Lennard-Jones potential.

    Parameters
    ----------
    sigma : float, default = 1.0
        Length scale for the potential.
    epsilon : float, default = 1.0
        Energy scale for the potential.
    """
    def __init__ (self, sigma=1.0, epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
    def __call__ (self, r):
        tmp = (self.sigma/r)**6.
        return 4.*self.epsilon * tmp * (tmp - 1)

class WCA (Potential):
    """WCA potential.

    The Weeks-Chandler-Andersen potential is a truncated and shifted
    Lennard-Jones potential.

    Parameters
    ----------
    sigma : float, default = 1.0
        Length scale for the potential.
    epsilon : float, default = 1.0
        Energy scale for the potential.
    """
    def __init__ (self, sigma=1.0, epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = 2.**(1./6) * self.sigma
    def __call__ (self, r):
        tmp = (self.sigma/r)**6.
        tmp = 4.*self.epsilon * tmp * (tmp - 1) + self.epsilon
        tmp[r>self.cutoff] = 0.
        return tmp

class softsphere (Potential):
    """Soft-sphere potential, also known as IPL (inverse-power law liquid).

    Parameters
    ----------
    sigma : float, default = 1.0
        Length scale for the potential.
    epsilon : float, default = 1.0
        Energy scale for the potential.
    n : int, default = 12
        Power-law exponent.
    """
    def __init__ (self, sigma=1.0, epsilon=1.0, n=12):
        self.sigma = sigma
        self.epsilon = epsilon
        self.n = n
    def __call__ (self, r):
        return self.epsilon * (self.sigma/r)**n

class SWS (Potential):
    """Square-Well System potential.

    Parameters
    ----------
    sigma : float, default = 1.0
        Diameter of the hard core.
    Delta : float, default = 0.1
        Width of the square well. Note that the commonly used
        `delta` is this parameter divided by the hard-core size.
    U0 : float, default = 1.0
        Depth of the square well attraction, in units of kT.
        Positive numbers correspond to negative potential values here.
    """
    def __init__ (self, sigma=1.0, Delta=0.1, U0=1.0):
        self.sigma = sigma
        self.Delta = Delta
        self.U0 = U0
    def __call__ (self, r):
        phi = np.zeros_like(r)
        phi[r<self.sigma+self.Delta] = -self.U0
        phi[r<self.sigma] = np.inf
        return phi
    def exp (self, r, kT):
        phi = np.ones_like(r)
        phi[r<self.sigma+self.Delta] = np.exp(self.U0/kT)
        phi[r<self.sigma] = 0.0
        return phi

class HSM (Potential):
    """Hard-sphere mixture potential.

    Parameters
    ----------
    sigma : array_like
        Array of hard-sphere diameters of the mixture.
    """
    def __init__ (self, sigma=[1.0]):
        self.sigma = np.array(sigma)
        self.sigmat = np.sum(np.meshgrid(self.sigma,self.sigma),axis=0)/2
    def __call__ (self, r):
        S = self.sigma.shape[0]
        phi = np.zeros((*r.shape,S,S))
        for a in range(S):
            for b in range(S):
                phi[r<self.sigmat[a,b],a,b] = np.inf
        return phi
    def exp (self, r, kT):
        S = self.sigma.shape[0]
        ephi = np.ones((*r.shape,S,S))
        for a in range(S):
            for b in range(S):
                ephi[r<self.sigmat[a,b],a,b] = 0.
        return ephi

sigmaKA = np.array([[1.0,0.8],[0.8,0.88]])
epsilonKA = np.array([[1.0, 1.5],[1.5, 0.5]])
class LJmix (Potential):
    """Lennard-Jones mixture potential.

    Parameters
    ----------
    sigma : array_like
        Length scales. Should be either of shape (S,) in which case
        the diameters will be set additively, or of shape (S,S) if you
        want to specify non-additive diameters.
    epsilon : array_like
        Energy scales. If of shape (S,S) those are the energy parameters
        of the potential (it's your responsibility to make this matrix
        symmetric if you want to stay within the range of equilibrium
        physics). If of shape (S,), the Berthelot mixing rule will be
        applied to derive the cross terms.

    Notes
    -----
    The default values are those of the Kob-Andersen potential that is
    a famous model system in glass-transition physics.
    """
    def __init__ (self, sigma=sigmaKA, epsilon=epsilonKA):
        self.sigma = np.array(sigma)
        if self.sigma.ndim == 1:
            self.sigmat = np.sum(np.meshgrid(self.sigma,self.sigma),axis=0)/2
        elif self.sigma.ndim == 2:
            self.sigmat = self.sigma
        else:
            raise
        self.epsilon = np.array(epsilon)
        if self.epsilon.ndim == 1:
            self.eps = np.sqrt(np.outer(self.epsilon,self.epsilon))
        elif self.epsilon.ndim == 2:
            self.eps = self.epsilon
        else:
            raise
    def __call__ (self, r):
        r_ = r[:,None,None]
        tmp = (self.sigmat/r_)**6.
        return 4.*self.eps * tmp * (tmp - 1)

