import numpy as np
from numba import njit
import scipy.special

from .standard import simple_liquid_model
from .shear import isotropically_sheared_model

class granular_model (simple_liquid_model):
    """Granular MCT model.

    Parameters
    ----------
    Sq : object
        Static structure factor.
    q : array_like
        Regular wave number grid.
    vth : float, default: 1.0
        Thermal velocity.
    restitution_coeff : float, default: 1.0
        Coefficient of restitution for the inelastic hard spheres.

    Notes
    -----
    See :py:class:`mctspy.simple_liquid_model` for restrictions on the
    wave number grid.
    """
    def __init__ (self, Sq, q, vth=1.0, restitution_coeff=1.0):
        self.epsilon = restitution_coeff
        self.vth = vth
        simple_liquid_model.__init__ (self, Sq, q, vth)
        if 'contact_value' in dir(Sq):
            self.contact_value = Sq.contact_value()
        else:
            self.contact_value = 1.0
        self.fixed_motion_type = 'damped_newtonian'
    def Aq (self):
        epsfac = (1+self.epsilon)/2 + (1-self.epsilon)/2*self.sq
        return self.sq/self.vth**2/epsfac
    def Bq (self):
        epsfac = (1+self.epsilon)/2 + (1-self.epsilon)/2*self.sq
        return self.omega_E()*(1+self.epsilon)/2 * \
            (1 - scipy.special.spherical_jn(0,self.q) \
            + 2*scipy.special.spherical_jn(2,self.q)) * \
            self.sq/self.vth**2/epsfac
    def omega_E (self):
        return 4*np.sqrt(np.pi)*self.rho*self.vth * self.contact_value
    def __init_vertices__ (self):
        simple_liquid_model.__init_vertices__(self)
        Aeps = 1. / (1. + (1.-self.epsilon)*self.sq/(1+self.epsilon))
        for n in range(9):
            self.b[n] = self.b[n] * Aeps[:,None]

class gitt_model (isotropically_sheared_model):
    """Granular ITT isotropically sheared model.

    This implements the isotropically sheared model, see
    :py:class:`mctspy.isotropically_sheared_model`, with the basis
    of the granular MCT model, :py:class:`mctspy.granular_model`.

    Parameters
    ----------
    Sq : object
        Static structure factor.
    q : array_like
        Regular wave number grid.
    vth : float, default: 1.0
        Thermal velocity.
    restitution_coeff : float, default: 1.0
        Coefficient of restitution for the inelastic hard spheres.
    gammadot : float, default: 0.0
        Shear rate.
    gammac : float, default: np.sqrt(3)
        Scaling factor for the strain.

    Notes
    -----
    See :py:class:`mctspy.simple_liquid_model` for restrictions on the
    wave number grid.
    """
    def __init__ (self, Sq, q, vth=1.0, restitution_coeff=1.0,
                  gammadot=0.0, gammac=np.sqrt(3)):
        self.epsilon = restitution_coeff
        self.vth = vth
        isotropically_sheared_model.__init__ (self, Sq, q, vth, gammadot, gammac)
        if 'contact_value' in dir(Sq):
            self.contact_value = Sq.contact_value()
        else:
            self.contact_value = 1.0
        self.fixed_motion_type = 'damped_newtonian'
    def Aq (self):
        epsfac = (1+self.epsilon)/2 + (1-self.epsilon)/2*self.sq
        return self.sq/self.vth**2/epsfac
    def Bq (self):
        epsfac = (1+self.epsilon)/2 + (1-self.epsilon)/2*self.sq
        return self.omega_E()*(1+self.epsilon)/2 * \
            (1 - scipy.special.spherical_jn(0,self.q) \
            + 2*scipy.special.spherical_jn(2,self.q)) * \
            self.sq/self.vth**2/epsfac
    def omega_E (self):
        return 4*np.sqrt(np.pi)*self.rho*self.vth * self.contact_value
    def __init_vertices__ (self):
        isotropically_sheared_model.__init_vertices__(self)
        Aeps = 1. / (1. + (1.-self.epsilon)*self.sq/(1+self.epsilon))
        for n in range(9):
            self.b[n] = self.b[n] * Aeps[:,None]
    def shear_modulus (self, phi, t):
        sig = isotropically_sheared_model.shear_modulus (self, phi, t)
        return (1+self.epsilon)/2 * sig
