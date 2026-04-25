import numpy as np
from .simple_liquid import simpleLiquidSq

# q0, q1, q2, q3: integrals \int_r0^r1 dr exp(iqr) r^alpha

# helper function to call func(q), but funclow(q) if q<qlow
# made to work with both numpy arrays and single numbers
def dispatch (func, funclow, q, qlow):
    if isinstance(q,np.ndarray):
        highq = q>=qlow
        lowq = q<qlow
        res = np.zeros_like(q,dtype=complex)
        res[lowq] = funclow(q[lowq])
        res[highq] = func(q[highq])
    else:
        highq = q>=qlow
        res = func(q) if highq else funclow(q)
    return res


class swsMSA (simpleLiquidSq):
    def __init__ (self, phi, delta=0.0, Gamma=0.0):
        """Square-well system structure factor, MSA.
    
        Parameters
        ----------
        phi : float
            Packing fraction of the hard-sphere cores.
        delta : float, default: 0.0
            Range of the square-well potential.
        Gamma : float, default: 0.0
            Attraction strength.
        """
        self.phi = phi
        self.rho = phi*6/np.pi
        self.Gamma = Gamma
        self.delta = delta 
        tau = self.Gamma*self.delta
        a0 = (1+2*phi)/(1-phi)**2-12*tau*phi/(1-phi)
        b0 = -3*phi/(2*(1-phi)**2)+6*tau*phi/(1-phi)
        #c0 = tau-a0/2-b0
        c0 = tau-1/(2*(1-phi))
        a1 = tau*(6*phi*(5*phi-2) - 72*c0*phi**2*(1-phi))/(1-phi)**2
        b1 = tau*(9*phi*(1-2*phi) + 36*c0*phi**2*(1-phi))/(1-phi)**2
        c1 = tau*(1 - 7*phi + 12*c0*phi*(1-phi))/(2*(1-phi))
        self.a  = a0+delta*a1
        self.b  = b0+delta*b1
        self.c  = c0+delta*c1
        self.c0 = c0
        self.lowq = 0.05
    def density (self):
        return self.phi*6/np.pi
    def Q0 (self, q, r0, r1):
        return dispatch (
            lambda q: (np.exp(1j*q*r0) - np.exp(1j*q*r1))*1j/q,
            lambda q: (-r0+r1) + 0.5j*q*(r1**2-r0**2) + q**2*(r0**3-r1**3)/6.,
            q, self.lowq)
    def Q1 (self, q, r0, r1):
        return dispatch (
            lambda q: ((-1+1j*q*r0)*np.cos(q*r0) + (1-1j*q*r1)*np.cos(q*r1)\
            + (- 1j - q*r0)*np.sin(q*r0) + (1j + q*r1)*np.sin(q*r1))/q**2,
            lambda q: (r1**2-r0**2)/2 + 1j*q*(r1**3-r0**3)/3 \
            + q**2*(r0**4-r1**4)/8,
            q, self.lowq)
    def Q2 (self, q, r0, r1):
        return dispatch (
            lambda q: (1j*(-2+2j*q*r0 + q**2*r0**2)*np.cos(q*r0) \
                    +(2j+2*q*r1 - 1j*q**2*r1**2)*np.cos(q*r1) \
                    -(-2+2j*q*r0 + q**2*r0**2)*np.sin(q*r0) \
                    +(-2+2j*q*r1 + q**2*r1**2)*np.sin(q*r1))/q**3,
            lambda q: (r1**3-r0**3)/3 - 1j*q*(r0**4-r1**4)/4 \
                    + q**2*(r0**5-r1**5)/10,
            q, self.lowq)
    def Q3 (self, q, r0, r1):
        return dispatch (
            lambda q: ((6-6j*q*r0 - 3*q**2*r0**2 + 1j*q**3*r0**3)*np.cos(q*r0) \
                    + (-6+6j*q*r1 + 3*q**2*r1**2 - 1j*q**3*r1**3)*np.cos(q*r1) \
                    + (6j+6*q*r0 - 3j*q**2*r0**2 - q**3*r0**3)*np.sin(q*r0) \
                    + (-6j-6*q*r1+3j*q**2*r1**2+q**3*r1**3)*np.sin(q*r1))/q**4,
            lambda q: (r1**4-r0**4)/4 - 1j*q*(r0**5-r1**5)/5 \
                    + q**2*(r0**6-r1**6)/12,
            q, self.lowq)
    def Q4 (self, q, r0, r1):
        return dispatch (
            lambda q: ((24j + 24*q*r0 - 12j*q**2*r0**2 - 4*q**3*r0**3 + 1j*q**4*r0**4)*np.cos(q*r0) \
                    + (-24j - 24*q*r1 + 12j*q**2*r1**2 + 4*q**3*r1**3 - 1j*q**4*r0**4)*np.cos(q*r1) \
                    + (-24 + 24j*q*r0 + 12*q**2*r0**2 - 4j*q**3*r0**3 - q**4*r0**4)*np.sin(q*r0) \
                    + (24 - 24j*q*r1 - 12*q**2*r1**2 + 4j*q**3*r1**3 + q**4*r1**4)*np.sin(q*r1))/q**5,
            lambda q: (r1**5-r0**5)/5 - 1j*(r0**6-r1**6)/6 \
                    + q**2*(r0**7-r1**7)/14,
            q, self.lowq)
    def Q (self, q):
        K=self.Gamma*self.delta
        Qq = 1. - 12*self.phi * (0.5*self.a*self.Q2(q,0.,1.) \
             + self.b*self.Q1(q,0.,1.) + self.c*self.Q0(q,0.,1.))
        if self.delta > 0:
            Qq -= 24*self.phi**2 * K**2 * self.delta**2 \
                * np.exp(1j*q*self.delta) * self.Q3(-q*self.delta,0.,1.)
            pre = 3*self.delta*self.phi*self.c0
            Qq -= 12*self.phi * self.delta*np.exp(1j*q) * K * \
                ((1+self.delta/2+pre)*self.Q0(q*self.delta,0.,1.) \
                - (1+2*pre)*self.Q1(q*self.delta,0.,1.) \
                + (pre - self.delta/2)*self.Q2(q*self.delta,0.,1.))
        return Qq
    def cq (self, q):
        Qq = self.Q(q)
        Sinv = (Qq * Qq.conjugate()).real
        return (1.-Sinv)/self.density()
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
        Qq = self.Q(q)
        K=self.Gamma*self.delta
        dQq = - 12j*self.phi * (0.5*self.a*self.Q3(q,0.,1.) \
             + self.b*self.Q2(q,0.,1.) + self.c*self.Q1(q,0.,1.))
        if self.delta > 0:
            dQq += 24j*self.delta*self.phi**2 * K**2 * self.delta**2 \
                * np.exp(1j*q*self.delta) * self.Q4(-q*self.delta,0.,1.)
            dQq -= 24j*self.delta*self.phi**2 * K**2 * self.delta**2 \
                * np.exp(1j*q*self.delta) * self.Q3(-q*self.delta,0.,1.)
            pre = 3*self.delta*self.phi*self.c0
            dQq -= 12j*self.delta*self.phi * self.delta*np.exp(1j*q) * K * \
                ((1+self.delta/2+pre)*self.Q1(q*self.delta,0.,1.) \
                - (1+2*pre)*self.Q2(q*self.delta,0.,1.) \
                + (pre - self.delta/2)*self.Q3(q*self.delta,0.,1.))
            dQq -= 12j* self.phi * self.delta*np.exp(1j*q) * K * \
                ((1+self.delta/2+pre)*self.Q0(q*self.delta,0.,1.) \
                - (1+2*pre)*self.Q1(q*self.delta,0.,1.) \
                + (pre - self.delta/2)*self.Q2(q*self.delta,0.,1.))
        return -(dQq*Qq.conjugate() + Qq*dQq.conjugate()).real/self.density()

