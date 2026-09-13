import numpy as np

class OZClosure:
    """Implementation of a closure for the Ornstein-Zernike equation.
    Needs to be callable with parameters r, ephi (the Boltzmann factor
    of the interaction potential) and Gamma(r) (r times the gamma(r)
    function from the OZ equation). Needs to also implement `dGamma` to
    return the derivative with respect to Gamma."""
    def __call__ (self, r, ephi, Gamma):
        raise NotImplementedError
    def dGamma (self, r, ephi, Gamma):
        raise NotImplementedError

class HNC (OZClosure):
    def __call__ (self, r, ephi, Gamma):
        return r*ephi*np.exp(Gamma/r) - Gamma - r
    def dGamma (self, r, ephi, Gamma):
        return ephi*np.exp(Gamma/r) - 1

class PY (OZClosure):
    def __call__ (self, r, ephi, Gamma):
        return (ephi-1)*(Gamma+r)
    def dGamma (self, r, ephi, Gamma):
        return ephi-1

class MSA (OZClosure):
    def __init__ (self, sigma=1.0):
        self.sigma = sigma
    def __call__ (self, r, ephi, Gamma):
        res = -Gamma - r
        res[r>=self.sigma] = r * np.log(ephi)
        return res
