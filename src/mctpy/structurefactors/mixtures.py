import numpy as np

from ..__h5__ import LoadableModel

class mixtureSq (object):
    def cq (self, q):
        """Direct correlation function (DCF).

        Parameters
        ----------
        q : array_like
            List of wave number values where to evaluate the DCF.

        Returns
        -------
        cq : array_like
            List of DCF values corresponding to the given wave numbers.
            If q has shape (M) and the number of species is S, the return
            value has shape (M,S,S).
        """
        if isinstance(q,np.ndarray):
            highq = q>=self.lowq
            lowq = q<self.lowq
            q_ = q[:,None,None]
            S = self.densities.shape[0]
            res = np.zeros((q.shape[0],S,S))
            res[lowq] = self._cq_low(q_[lowq])
            res[highq] = self._cq_high(q_[highq])
        else:
            highq = q>=self.lowq
            res = self._cq_high(q) if highq else self._cq_low(q)
        return res
    def Sq (self, q):
        """Structure factor and DCF.

        Parameters
        ----------
        q : array_like
            List of wave numbers where to evaluate the structure functions,
            should have shape (M,).

        Returns
        -------
        sq : array_like
            List of S(q) values, shape (M,S,S).
        cq : array_like
            List of c(q) values, shape (M,S,S).
        """
        cq_ = self.cq(q)
        rhoa, rhob = np.meshgrid(self.densities, self.densities)
        if isinstance(q,np.ndarray):
            return np.linalg.inv(-np.sqrt(rhoa*rhob)[None,:,:]*cq_ + np.diag(np.ones_like(self.densities))[None,:,:]), cq_
        else:
            return np.linalg.inv(-np.sqrt(rhoa*rhob)*cq_ + np.diag(np.ones_like(self.densities))), cq_
    def dcq_dq (self, q):
        """Return derivative to the DCF.

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where the DCF should be evaluated.

        Returns
        -------
        dcq : array_like
            Derivative of the DCF evaluated on the given grid.
        """
        if isinstance(q,np.ndarray):
            highq = q>=self.lowq
            lowq = q<self.lowq
            q_ = q[:,None,None]
            S = self.densities.shape[0]
            res = np.zeros((q.shape[0],S,S))
            res[lowq] = self._dcq_dq_low(q_[lowq])
            res[highq] = self._dcq_dq_high(q_[highq])
        else:
            highq = q>=self.lowq
            res = self._dcq_dq_high(q) if highq else self._dcq_dq_low(q)
        return res

class mixtureTagged (LoadableModel, role='structurefactor',
                                    attrs=['s'], objects=['Sq']):
    def __init__ (self, Sq, s):
        """Take a mixture structure factor and initialize a view
        on its DCF as that of a tagged particle of a given species
        inside that mixture.

        Parameters
        ----------
        Sq : mixtureSq
            Structure factor of the underlying mixture.
        s : int
            Species index.
        """
        super().__init__()
        self.Sq = Sq
        self.s = s
    def reinit (self):
        pass
    def cq (self, q):
        return self.Sq.cq(q)[:,:,self.s]


class hsm_densities (object):
    r"""Convenience wrapper for specifications of the density of
    hard-sphere mixtures.

    Parameters:
    -----------
    sigma : array_like
        The diameters of the hard spheres in the mixture. The length of
        this array sets the number of species in the mixture.
    rho : float or None
        Total number density of the mixture.
    x : array_like, float or None
        Number fractions of the mixture species. If an array, must be
        of the same length as `sigma`. If a single number is given,
        this will determine the number fraction of the last species,
        all others assigned identical values so that the sum is unity.
    phi : float or None
        Total packing fraction of the mixture.
    xhat : array_like, float or None
        Partial volume fractions of the mixture. If an array, must be
        of the same length as `sigma`. If a single number is given,
        the volume fraction of the last species will be set to this number,
        and all other species share an identical partial volume fraction
        among themselves.

    Currently, there must be either `rho` and `x` not set to `None` only,
    or `phi` and `xhat` not set to None only. Other combinations either
    do not make sense since they would over-specify the problem, or are
    not implemented.
    If all four parameters are left `None` (the default), the object will
    remain uninitialized.

    The object acts as an iterator, so that it can be used with the
    python star-operator as an initialization argument to
    `mctpy.structurefactors.hsmPY`.
    """
    def __init__ (self, sigma, rho=None, x=None, phi=None, xhat=None):
        self.d = np.array(sigma)
        if (rho is not None) and (x is not None):
            self.partial_densities(rho, x)
        elif (phi is not None) and (xhat is not None):
            self.volume_fractions(phi, xhat)
    def partial_densities(self, rho):
        """Set state from partial number densities.

        Parameters
        ----------
        rho : array_like
            Must be an array matching the number of species in the mixture,
            and giving the partial number densities.
        """
        self.x = np.array(rho)
        self.rho = np.sum(self.x)
        self.x = self.x/self.rho
        self.phi = np.pi/6. * self.rho * np.sum(self.x*self.d**3)
        self.xhat = self.x*self.d**3/np.sum(self.x*self.d**3)
        return self
    def number_fractions(self, rho, x):
        """Set state from partial number fractions.

        Parameters
        ----------
        rho : float
            Total number density of the mixture.
        x : array_like or float
            Number fractions of the individual species; if an array, must
            match in length the number of species. If a float, this sets the
            number fraction of the last species only. The remaining fraction
            is then redistributed to all other species equally (by number).
        """
        if not isinstance(x,(list,np.ndarray)):
            self.xh = np.empty_like(self.d)
            self.x[-1] = x
            self.x[:-1] = (1.0 - x)/(self.x.shape[0]-1)
        else:
            self.x = np.array(x)
        self.rho = np.float64(rho)
        self.phi = np.pi/6. * self.rho * np.sum(self.x*self.d**3)
        self.xhat = self.x*self.d**3/np.sum(self.x*self.d**3)
        return self
    def volume_fractions(self, phi=0., xhat=0.):
        """Set state from partial volume fractions.

        Parameters
        ----------
        phi : float
            Total volume fraction (packing fraction) of the mixture.
        xhat : array_like or float
            Volume fractions of the individual species; if an array, must
            match in length the number of species. If a float, this sets the
            volume fraction of the last species only. The remaining fraction
            is then redistributed to all other species equally (by volume).
        """
        if not isinstance(xhat,(list,np.ndarray)):
            self.xhat = np.empty_like(self.d)
            self.xhat[-1] = xhat
            self.xhat[:-1] = (1.0 - xhat)/(self.xhat.shape[0]-1)
        else:
            self.xhat = np.array(xhat)
        self.phi = np.float64(phi)
        self.x = self.xhat/self.d**3 / np.sum(self.xhat/self.d**3)
        self.rho = 6./np.pi * self.phi * np.sum(self.xhat/self.d**3)
        return self
    def __call__ (self, format='partial_densities'):
        """Return the state of the mixture parameters.

        Parameters
        ----------
        format : str
            Either of 'partial_densities' (default), 'volume_fractions',
            or 'number_fractions'.

        Returns
        -------
        (partial_densities, diameters) if format == 'partial_densities'
        (volume_fraction, partial_volume_fractions, diameters)
          if format == 'volume_fractions'
        (density, number_fractions, diameters) if format == 'number_fractions'
        """
        if format == 'partial_densities':
            return self.rho*self.x, self.d
        if format == 'volume_fractions':
            return self.phi, self.xhat, self.d
        if format == 'number_fractions':
            return self.rho, self.x, self.d
    def __iter__ (self):
        yield self.rho*self.x
        yield self.d


class hsmPY (mixtureSq, LoadableModel, role='structurefactor',
                                       fields=['densities','diameters']):
    def __init__ (self, densities=None, diameters=None):
        """Hard-sphere mixture structure factor, Percus-Yevick (PY).

        Parameters
        ----------
        densities : array_like
            List of number densities of the hard-sphere species.
        diameters : array_like
            List of diameters of the hard spheres, must match shape of densities.
        """
        super().__init__()
        self.densities = np.array(densities)
        self.diameters = np.array(diameters)
        self.reinit()
    def reinit (self):
        xi1 = np.sum(self.densities*self.diameters) * np.pi/6.
        xi2 = np.sum(self.densities*self.diameters**2) * np.pi/6.
        xi3 = np.sum(self.densities*self.diameters**3) * np.pi/6.
        atmp = ((1-xi3) + 3*self.diameters*xi2) / (1-xi3)**2
        btmp = -1.5*self.diameters**2*xi2 / (1-xi3)**2
        self.a2 = np.sum(self.densities*atmp**2)
        b0 = np.sum(self.densities*np.pi*(atmp*btmp+0.5*atmp**2*self.diameters))
        self.da, self.db = np.meshgrid(self.diameters, self.diameters)
        self.dab = (self.da+self.db)/2
        self.dadb = self.da*self.db
        self.A = self.dab/(1-xi3) + 1.5*self.dadb*xi2/(1-xi3)**2
        self.B = 1./(1-xi3) - b0*self.dadb
        self.D = (6*xi2 + 12*self.dab*(xi1+3*xi2**2/(1-xi3)))/(1-xi3)**2
        self.lowq = 0.05/np.max(self.diameters)
    def density (self):
        return np.sum(self.densities)
    def partial_density (self, a):
        return self.densities[a]
    def _cq_high (self, q):
        q2 = q*q
        q3 = q2*q
        q4 = q2*q2
        Sa, Sb = np.sin(0.5*q*self.da), np.sin(0.5*q*self.db)
        Ca, Cb = np.cos(0.5*q*self.da), np.cos(0.5*q*self.db)
        return -4*np.pi * ( \
                self.A * (Sa*Sb - Ca*Cb)/q2 + self.B * (Ca*Sb + Cb*Sa)/q3 \
                + self.D * Sa*Sb/q4 \
                + np.pi/q4*self.a2 * (Ca*Cb*self.da*self.db \
                + 4*Sa*Sb/q2 - 2*(Ca*Sb*self.da + Cb*Sa*self.db)/q))
    def _cq_low (self, q):
        q2 = q*q
        q4 = q2*q2
        dab = self.da + self.db
        dadb2 = self.da**2 + self.db**2
        dadb4 = self.da**4 + self.db**4
        return -np.pi/6 * (np.pi/6 * self.a2 * self.dadb**3 + 2*self.A*dab**2 + 0.5*self.D*self.dadb**2) + np.pi/120.*(self.A * dab**4 + self.D*self.dadb**3/6 + (0.25*self.D + np.pi/12 * self.a2 * self.dadb)*self.dadb**2*dadb2)*q2 - np.pi/13440 * (self.A * dab**6 + (0.25*self.D + np.pi/12*self.a2*self.dadb) * self.dadb**2 * dadb4 + (self.D * dab**2 + 7*np.pi/10*self.a2 * self.dadb**2)*self.dadb**3/3 + (1./6)*self.D*self.dadb**4)*q4
    def _dcq_dq_high (self, q):
        q2 = q*q
        q3 = q2*q
        q4 = q2*q2
        q5 = q3*q2
        q6 = q3*q3
        Sa, Sb = np.sin(0.5*q*self.da), np.sin(0.5*q*self.db)
        Ca, Cb = np.cos(0.5*q*self.da), np.cos(0.5*q*self.db)
        return 4.*np.pi* ( \
                -self.A*self.dab*(Cb*Sa+Ca*Sb)/q2 \
                -(2*self.A+self.B*self.dab)*(Ca*Cb-Sa*Sb)/q3 \
                + (Cb*Sa*self.db*(-self.D+np.pi*self.a2*self.da**2) \
                +  Ca*Sb*self.da*(-self.D+np.pi*self.a2*self.db**2) \
                +6*self.B*(Cb*Sa+Ca*Sb))/(2*q4) \
                + (6*np.pi*self.a2*Ca*Cb*self.dadb \
                - np.pi*self.a2*Sa*Sb*(self.da**2+self.db**2) \
                + 4*self.D*Sa*Sb)/q5 \
                - 12*np.pi*self.a2*(Cb*Sa*self.db+Ca*Sb*self.da-2*Sa*Sb/q)/q6
            )
    def _dcq_dq_low (self, q):
        q2 = q*q
        q3 = q2*q
        dab = self.da + self.db
        dadb2 = self.da**2+self.db**2
        dadb4 = self.da**4+self.db**4
        return np.pi/60.*(self.A*dab**4+self.D*self.dadb**3/6. \
            +(0.25*self.D+np.pi/12.*self.a2*self.dadb)*self.dadb**2*dadb2) *q \
            - np.pi/3360.*(self.A*dab**6 \
            +(0.25*self.D+np.pi/12.*self.a2*self.dadb)*self.dadb**2*dadb4 \
            +(self.D*dab**2+7.*np.pi/10*self.dadb**2)*self.dadb**3/3.) *q3

    def contact_value (self):
        """Return contact value of the RDFs.

        This is the value matching the virial equation of state in the
        Percus-Yevick approximation."""
        xi2 = np.sum(self.densities*self.diameters**2) * np.pi/6.
        phi = np.sum(self.densities*self.diameters**3) * np.pi/6.
        return ((1-phi) + 3*xi2*self.da*self.db/(self.da+self.db)) / (1-phi)**2
