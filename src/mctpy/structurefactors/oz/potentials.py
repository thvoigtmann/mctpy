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
    def __init__ (self, sigma=1.0):
        self.sigma = sigma
    def __call__ (self, r):
        phi = np.zeros_like(r)
        phi[r<self.sigma] = np.inf
        return phi
    def exp (self, r, kT):
        return np.array(r>self.sigma, dtype=np.float64)

class LJ (Potential):
    def __init__ (self, sigma=1.0, epsilon=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
    def __call__ (self, r):
        tmp = (self.sigma/r)**6.
        return 4.*self.epsilon * tmp * (tmp - 1)

class WCA (Potential):
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
    def __init__ (self, sigma=1.0, epsilon=1.0, n=12):
        self.sigma = sigma
        self.epsilon = epsilon
        self.n = n
    def __call__ (self, r):
        return self.epsilon * (self.sigma/r)**n

class SWS (Potential):
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
