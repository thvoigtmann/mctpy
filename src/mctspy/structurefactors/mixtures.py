import numpy as np

class hsmPY (object):
    """Hard-sphere mixture structure factor, Percus-Yevick (PY).

    Parameters
    ----------
    densities : array_like
        List of number densities of the hard-sphere species.
    diameters : array_like
        List of diameters of the hard spheres, must match shape of densities.
    """
    def __init__ (self, densities, diameters):
        self.densities = np.array(densities)
        self.diameters = np.array(diameters)
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
        highq = q>=self.lowq
        lowq = q<self.lowq
        q_ = q[:,None,None]
        q2 = q_*q_
        q3 = q2*q_
        q4 = q2*q2
        Sa, Sb = np.sin(0.5*q_[highq]*self.da), np.sin(0.5*q_[highq]*self.db)
        Ca, Cb = np.cos(0.5*q_[highq]*self.da), np.cos(0.5*q_[highq]*self.db)
        res = np.zeros((q.shape[0],self.densities.shape[0],self.densities.shape[0]))
        res[highq] = -4*np.pi * ( \
            self.A * (Sa*Sb - Ca*Cb)/q2[highq] + self.B * (Ca*Sb + Cb*Sa)/q3[highq] \
            + self.D * Sa*Sb/q4[highq] \
            + np.pi/q4[highq]*self.a2 * (Ca*Cb*self.da*self.db \
                + 4*Sa*Sb/q2[highq] - 2*(Ca*Sb*self.da + Cb*Sa*self.db)/q_[highq]))
        if lowq.any():
            dab = self.da + self.db
            dadb2 = self.da**2 + self.db**2
            dadb4 = self.da**4 + self.db**4
            res[lowq] = -np.pi/6 * (np.pi/6 * self.a2 * self.dadb**3 + 2*self.A*dab**2 + 0.5*self.D*self.dadb**2) + np.pi/120.*(self.A * dab**4 + self.D*self.dadb**3/6 + (0.25*self.D + np.pi/12 * self.a2 * self.dadb)*self.dadb**2*dadb2)*q2[lowq] - np.pi/13440 * (self.A * dab**6 + (0.25*self.D + np.pi/12*self.a2*self.dadb) * self.dadb**2 * dadb4 + (self.D * dab**2 + 7*np.pi/10*self.a2 * self.dadb**2)*self.dadb**3/3 + (1./6)*self.D*self.dadb**4)*q4[lowq]
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
        return np.linalg.inv(-np.sqrt(rhoa*rhob)[None,:,:]*cq_ + np.diag(np.ones_like(self.densities))[None,:,:]), cq_
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
        highq = q>=self.lowq
        lowq = q<self.lowq
        q_ = q[:,None,None]
        q2 = q_*q_
        q3 = q2*q_
        q4 = q2[highq]*q2[highq]
        q5 = q3[highq]*q2[highq]
        q6 = q3[highq]*q3[highq]
        Sa, Sb = np.sin(0.5*q_[highq]*self.da), np.sin(0.5*q_[highq]*self.db)
        Ca, Cb = np.cos(0.5*q_[highq]*self.da), np.cos(0.5*q_[highq]*self.db)
        dcq = np.zeros((q.shape[0],self.densities.shape[0],self.densities.shape[0]))
        dcq[highq] = 4.*np.pi* ( \
            -self.A*self.dab*(Cb*Sa+Ca*Sb)/q2[highq] \
            -(2*self.A+self.B*self.dab)*(Ca*Cb-Sa*Sb)/q3[highq] \
            + (Cb*Sa*self.db*(-self.D+np.pi*self.a2*self.da**2) \
            +  Ca*Sb*self.da*(-self.D+np.pi*self.a2*self.db**2) \
               +6*self.B*(Cb*Sa+Ca*Sb))/(2*q4) \
            + (6*np.pi*self.a2*Ca*Cb*self.dadb \
              - np.pi*self.a2*Sa*Sb*(self.da**2+self.db**2) \
              + 4*self.D*Sa*Sb)/q5 \
            - 12*np.pi*self.a2*(Cb*Sa*self.db+Ca*Sb*self.da-2*Sa*Sb/q_[highq])/q6
            )
        if lowq.any():
            dab = self.da + self.db
            dadb2 = self.da**2+self.db**2
            dadb4 = self.da**4+self.db**4
            dcq[lowq] = \
                np.pi/60.*(self.A*dab**4+self.D*self.dadb**3/6. \
                +(0.25*self.D+np.pi/12.*self.a2*self.dadb)*self.dadb**2*dadb2) *q_[lowq] \
                - np.pi/3360.*(self.A*dab**6 \
                +(0.25*self.D+np.pi/12.*self.a2*self.dadb)*self.dadb**2*dadb4 \
                +(self.D*dab**2+7.*np.pi/10*self.dadb**2)*self.dadb**3/3.) *q3[lowq]
        return dcq
