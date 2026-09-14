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
    """Hypernetted-chain closure to OZ equation."""
    def __call__ (self, r, ephi, Gamma):
        return r*ephi*np.exp(Gamma/r) - Gamma - r
    def dGamma (self, r, ephi, Gamma):
        return ephi*np.exp(Gamma/r) - 1
class HNCmix (OZClosure):
    """Hypernetted-chain closure to OZ equation, mixture version."""
    def __call__ (self, r, ephi, Gamma):
        r_ = r[:,None,None]
        return r_*ephi*np.exp(Gamma/r_) - Gamma - r_
    def dGamma (self, r, ephi, Gamma):
        S = ephi.shape[-1]
        r_ = r[:,None,None]
        res = np.zeros((*ephi.shape,*ephi.shape[1:]))
        for a in range(S):
            for b in range(S):
                res[:,a,b,a,b] = ephi[:,a,b]*np.exp(Gamma[:,a,b]/r) - 1
        return res


class PY (OZClosure):
    """Percus-Yevick closure to OZ equation."""
    def __call__ (self, r, ephi, Gamma):
        return (ephi-1)*(Gamma+r)
    def dGamma (self, r, ephi, Gamma):
        return ephi-1
class PYmix (OZClosure):
    """Percus-Yevick closure to OZ equation, mixture version."""
    def __call__ (self, r, ephi, Gamma):
        return (ephi-1) * (Gamma + r[:,None,None])
    def dGamma (self, r, ephi, Gamma):
        S = ephi.shape[-1]
        res = np.zeros((*ephi.shape,*ephi.shape[1:]))
        for a in range(S):
            for b in range(S):
                res[:,a,b,a,b] = ephi[:,a,b]-1
        return res


class MSA (OZClosure):
    """Mean-spherical approximation for the OZ equation."""
    def __init__ (self, sigma=1.0):
        self.sigma = sigma
    def __call__ (self, r, ephi, Gamma):
        res = -Gamma - r
        res[r>=self.sigma] = r * np.log(ephi)
        return res
    def dGamma (self, r, ephi, Gamma):
        return np.array(-(r<self.sigma),dtype=np.float64)
class MSAmix (OZClosure):
    """Mean-spherical approximation for the OZ equation for mixtures."""
    def __init__ (self, sigma=[1.0]):
        self.sigma = np.array(sigma)
        self.sigmat = np.sum(np.meshgrid(self.sigma,self.sigma),axis=0)/2
    def __call__ (self, r, ephi, Gamma):
        r_ = r[:,None,None]
        res = - Gamma - r_
        S = res.shape[-1]
        for a in range(S):
            for b in range(S):
                res[r>=self.sigmat[a,b],a,b] = r * np.log(ephi[a,b])
        return res
    def dGamma (self, r, ephi, Gamma):
        res = np.zeros(*ephi.shape, *ephi.shape[1:])
        S = ephi.shape[-1]
        for a in range(S):
            for b in range(S):
                res[:,a,b,a,b] = -(r<self.sigmat[a,b])
        return res
