==========
MCT Models
==========
.. include:: <isolat1.txt>

Simple Liquids
==============

"Simple liquid" models are the "standard models" of MCT. The liquid is
assumed to be isotropic, homogeneous, and composed of identical particles
without relevant internal degrees of freedom, so that their positions provide
full information on the configuration.
The MCT equations then take the form of scalar-valued
equations for each :math:`\phi(q,t)`, where :math:`q` is the wave number.
The different :math:`q` only couple through the memory kernel, and
for its implementation, the values are usually chosen on a regular grid
(although model implementations may deal with more general wave-number
grids).

For these models, :math:`\phi(q,t)` is usually normalized
such that :math:`\phi(q,t=0)=1`. For the memory kernel, we adopt
the convention that :math:`\hat m(q,t)=q^2m(q,t)` is the quantity that is
calculated and stored, on the basis that in equilibrium dynamics, this
quantity possesses a non-singular :math:`q\to0` limit.

The evolution equations of simple-liquid models are thus

.. math::

    q^2\omega_q^{-1}\partial_t\phi(q,t) + q^2\phi(q,t)
    + \int_0^t\hat m(q,t-t')\partial_{t'}\phi(q,t)=0


Standard scalar model of MCT
----------------------------


.. autoclass:: mctpy.simple_liquid_model
    :members:
    :inherited-members:

    For the shear modulus, the expression is

    .. math::

        \frac{G}{kT/\sigma^3}=\frac{\rho}{60\pi^2}\int dk\,
        k^4 (S_k c'_k)^2 \phi_k^2

    as for example given in [Bergenholtz1999]_.

    .. [Bergenholtz1999] J. Bergenholtz and M. Fuchs,
       Phys. Rev. E 59, 5706 (1999),
       `DOI:10.1103/PhysRevE.59.5706 <https://doi.org/10.1103/PhysRevE.59.5706>`_


Common Additions to Simple-Liquid Models
----------------------------------------

.. autoclass:: mctpy.tagged_particle_model

.. autoclass:: mctpy.tagged_particle_q0

.. autoclass:: mctpy.tagged_particle_ngp
    :members:
    :inherited-members:

    The non-Gaussian parameter :math:`\alpha_2(t)` is derived from
    the next-to-leading terms in the :math:`q\to0` limit of the MCT
    evolution equation. This was first derived and discussed in detail
    in [Fuchs1998]_.

    In essence, we define the following two memory kernels that correspond
    to those needed for the MCT equations for the NGP:

    .. math::
        :nowrap:

        \begin{align}
        m^s_0&=\frac1{6\pi^2}\int\rho S_k(c_k^s)^2k^4f_kf_k^s\,dk\\
        \tilde m^s_0&=\frac1{10\pi^2}\int\rho S_k(c_k^s)^2k^4
            (\partial_k^2f^s_k+(2/3k)\partial_kf^s_k)\,dk
        \end{align}

    Here, :math:`f^s_k` is the tagged-particle correlator, obtained from
    `self.base` which is set to `model_base.base.base` by the initialization
    method of this class; :math:`f_k` is the collective correlator,
    obtained from `model_base.base.base.base`.

    The first memory kernel is the one that also appears in the MSD,
    however we re-calculate it here. Note however that the MSD is not
    re-calculated (for the sake of modularity, since calculation of the MSD
    is kept to its separate solver).

    The equations itself are documented with
    :py:class:`mctpy.non_gaussian_parameter` with which this model is
    intended to be used.

    .. [Fuchs1998] M. Fuchs, W. G\ |ouml|\ tze, and M. R. Mayr,
       Phys. Rev. E 58, 3384 (1998),
       `DOI:10.1103/PhysRevE.58.3384 <https://doi.org/10.1103/PhysRevE.58.3384>`_

Sheared Systems
===============

Isotropically Sheared System
----------------------------

The "isotropically sheared hard-sphere model" (ISHSM) was introduced
by Fuchs and Cates [Fuchs2009]_. Its memory kernel is

.. math::

    m_q(t) = \frac1{2N}\sum_{\vec k}V^{(\dot\gamma)}_{\vec q,\vec k}(t)
    \phi_k(t)\phi_p(t)

with vertices

.. math::

    V^{(\dot\gamma)_{\vec q,\vec k}(t)=\frac{rho^2S_qS_kS_p}{q^4}
    \left[\vec q\cdot\vec k\,c_{k(t)}+\vec q\cdot\vec p\,c_{p(t)}\right]
    \left[\vec q\cdot\vec k\,c_k+\vec q\cdot\vec p\,c_p\right]

and the isotropic advected wave number
:math:`k(t)=k\sqrt{1+(\dot\gamma t/\gamma_c)^2}`.


.. autoclass:: mctpy.isotropically_sheared_model
    :members:
    :inherited-members:

    Note that the expression for the stress is [Fuchs2009]_

    .. math::

        \sigma = \frac{kT\rho\dot\gamma}{60\pi^2}\int_0^\infty dt
        \int dk\frac{k^5c_k'S'_{k(t)}}{k(t)}\phi_{k(t)}^2(t)

    In the current version of the code, we integrate however
    over :math:`\phi_k(t)` instead of the advected one; this seems
    to match more closely the numerical results shown by Fuchs in
    [Fuchs2010]_.

    .. [Fuchs2009] M. Fuchs and M. E. Cates, J. Rheol. 53, 957 (2009),
       `DOI:10.1122/1.3119084 <https://doi.org/10.1122/1.3119084>`_

    .. [Fuchs2010] M. Fuchs, Adv. Polym. Sci. 236, 55 (2010),
       `DOI:10.1007/12_2009_30 <https://doi.org/10.1007/12_2009_30>`_


Granular Systems
================

Driven Granular Fluids
----------------------

The MCT equations for driven granular fluids take a form that is very
similar to that of the simple liquids, but they naturally encode damped
Newtonian dynamics, and involve a prefactor in front of the memory kernel
that depends on the coefficient of restitution. The model has been
presented by Kranz/Sperl/Zippelius [Kranz2010]_.

.. [Kranz2010] W. T. Kranz, M. Sperl, A. Zippelius,
   Phys. Rev. Lett. 104, 225701 (2010),
   `DOI:10.1103/PhysRevLett.104.225701 <https://doi.org/10.1103/PhysRevLett.104.225701>`_


.. autoclass:: mctpy.granular.granular_model
    :members:
    :inherited-members:

    Note that the Enskog damping term requires a contact value for
    the RDF. It is expected that the structure factor implements a
    function `contact_value()`, else this value is set to unity.

On the basis of the granular MCT model, Kranz et al. [Kranz2018]_ have
developed an integration-through transients (ITT) model for sheared and
agitated granular materials.

.. [Kranz2018] W. T. Kranz, F. Frahsa, A. Zippelius, M. Fuchs,
   and M. Sperl, Phys. Rev. Lett. 121, 148002 (2018),
   `DOI:10.1103/PhysRevLett.121.148002 <https://doi.org/10.1103/PhysRevLett.121.148002>`_

.. autoclass:: mctpy.granular.gitt_model
    :members:
    :inherited-members:


Mixtures
========

Standard Model for Mixtures
---------------------------

Mixture models here refer to mixtures of simple-liquid components,
so that the only addition to the case of simple liquids is a particle-species
index. The MCT equations are thus matrix-valued evolution equations
of the form

.. math::

    q^2\boldsymbol\omega(q)^{-1}\cdot\partial_t\boldsymbol\Phi(q,t)
    +q^2\boldsymbol S(q)^{-1}\cdot\boldsymbol\Phi)(q,t)
    +\int_0^t q^2\boldsymbol M(q,t-t')\cdot\partial_{t'}\boldsymbol
    \Phi(q,t')\,dt'=0

where bold symbols indicate :math:`S\times S` matrices for a mixture
of S different species.

Besides multiplication with :math:`q^2`, the mixture memory kernel is
also multiplied with :math:`\sqrt{x_\alpha x_\beta}`, and in turn the
correlation function is divided by that factor. For a mixture where
one species has vanishing concentration, the corresponding diagonal
element of the correlator matrix then is the tagged-particle correlator.

.. autoclass:: mctpy.mixture_model
    :members:
    :inherited-members:

    This is the model studied, for example in [Goetze2003]_.

    The MCT memory-kernel expression for mixtures is given by

    .. math::

        M_{\alpha\beta}(q)=\frac{\varrho}{2q^4}\int\frac{d^3k}{(2\pi)^3}
        \sum_{\alpha'\alpha''\beta'\beta''}
        V_{\alpha\alpha'\alpha''}(\vec q,\vec k\vec p)
        V_{\beta\beta'\beta''}(\vec q,\vec k\vec p)
        F_{\alpha'\beta'}(k)F_{\alpha''\beta''}(p)

    with :math:`\vec q=\vec k+\vec p` and vertices

    .. math::

        V_{\alpha\alpha'\alpha''}(\vec q,\vec k\vec p)
        =(\vec q\vec k)\tilde c_{\alpha\alpha'}(k)\delta_{\alpha\alpha''}
        +(\vec q\vec p)\tilde c_{\alpha\alpha''}(p)\delta_{\alpha\alpha'}
        +q\sqrt{x_\alpha}\tilde c^{(3)}_{\alpha\alpha'\alpha''}(\vec q,\vec k)

    where we have introduced the modified DCF
    :math:`\tilde c_{\alpha\beta}(k)=\sqrt{x_\beta}c_{\alpha\beta}(k)`
    and :math:`\tilde c^{(3)}_{\alpha\beta\gamma}(\vec q,\vec k)
    =\sqrt{x_\beta x_\gamma}c^{(3)}_{\alpha\beta\gamma}(\vec q,\vec k)`.
    The quantities :math:`\boldsymbol F` appearing in the expression are
    assumed to be "normalized" to correspond to :math:`\boldsymbol S`, the
    static structure factor, if taken as equal-time averages.

    The three-particle static correlation function is currently not
    implemented.

    We restrict wave vectors to a finite grid, :math:`q_i = (\Delta q)(i+1/2)`
    and write the discretized integral with
    :math:`M_{\alpha\beta}(q)=(1/q^2)\hat M_{\alpha\beta}(q)`

    .. math::
        \hat M_{\alpha\beta}(q) = \frac{(\Delta q)^2}{32\pi^2q^3}\sum_{kp}
        \sum_{\alpha'\alpha''\beta'\beta''} kp
        \hat V_{\alpha\alpha'\alpha''}(q,k,p)\hat V_{\beta\beta'\beta''}(q,k,p)
        F_{\alpha'\beta'}(k)F_{\alpha''\beta''}(p)

    with

    .. math::

        \hat V_{\alpha\alpha'\alpha''}(q,k,p) = \left[
        \delta_{\alpha\alpha''}(q^2+k^2-p^2)\hat c_{\alpha\alpha'}(k)
        +\delta_{\alpha\alpha'}(q^2-k^2+p^2)\hat c_{\alpha\alpha''}(p)\right]

    and :math:`\hat c_{\alpha\beta}(k)=\sqrt{\varrho_\beta}c_{\alpha\beta}(k)`.

    In the above expression and throughout our mixture-model
    implementation, we normalize the correlation functions to have as their
    initial value the non-singular part of the structure factor, i.e. the
    one whose diagonal matrix elements approach unity for large :math:`q`.
    This differs from the common definition of a partial dynamic density
    correlation function: one has to multiply the :math:`\alpha\beta`
    matrix element of our correlation functions by
    :math:`\sqrt{x_\alpha x_\beta}`, and the memory function has to be divided
    by that expression when transforming to the common notation.

    The model also stores :math:`\hat M` instead of :math:`M`.

    These conventions are consistent throughout the rest of the code.

    .. [Goetze2003] W. G\ |ouml|\ tze and Th. Voigtmann,
       Phys. Rev. E 67, 021502 (2003),
       `DOI:10.1103/PhysRevE.67.021502 <https://doi.org/10.1103/PhysRevE.67.021502>`_


Two-dimensional MCT
===================

Two-dimensional MCT differs from the three-dimensional one only by
the dimensionality of the wave-vector integral in the memory kernel
(and by the input structure functions which have different large-wave-number
asymptotics). Especially in 2d this requires some care in how the
memory kernel is evaluated, since the usual discretization employed
for MCT introduces a variable transform with a Jacobian that in 2d
contains a square-root singularity. Specific memory-kernel algorithms
are thus implemented, which take care of better approximation, at the
cost of noticeably higher computational effort, since some of the tricks
introduced in the 3d discretization no longer work in 2d.


.. autoclass:: mctpy.simple_liquid_model_2d
    :members:
    :inherited-members:

    The MCT memory kernel implemented here is of the form

    .. math::

        m_q = \rho\int\frac{d^dk}{(4\pi)^d}\frac{S_q}{q^{d+2}}\Omega_{d-1}
              \int_0^\infty dk\,k\Phi_k\int_{|q-k|}^{q+k}dp\,p\Phi_p
              \frac{((q^2+k^2-p^2)c_k + (q^2+p^2-k^2)c_p)^2}
                   {[4q^2k^2-(q^2+k^2-p^2)^2]^{(3-d)/2}}

    specifically for :math:`d=2`. Here, :math:`\Omega_d=2\pi^{d/2}/\Gamma(d/2)`
    is the volume of the unit sphere, and :math:`\Omega_1=2`. Note that
    :math:`\Phi_k=S_k\phi_k(t)` is the non-normalized correlator.

    The code follows the paper by Caraglio et al [Caraglio2020]_.
    The kernel method calculates the outer integral following the
    trapezoidal rule,

    .. math::

        q_i^2 m_i = \frac{\rho S_i}{8\pi^2q_i^2}\sum_{j=0}^{M-2}
              \frac{q_j\Phi_jA_{ij}+q_{j+1}\Phi_{j+1}A_{i,j+1}}{2}
              (q_{j+1}-q_j)

    with the inner integral

    .. math::
        :nowrap:

        \begin{align}
        A_{ij}&=\int_{|q_i-q_j|}^{q_i+q_j}dp\,p\Phi_p
        \frac{[(q_i^2+q_j^2-p^2)c_j+(q_i^2+p^2-q_j^2)c_p]^2}
             {\sqrt{4q_i^2q_j^2-(q_i^2+q_j^2-p^2)^2}} \\
        &= 2q_i^2q_j^2\int_{-1}^1dx\,\Phi_{p(x)}
        \frac{[x(c_j-c_{p(x)})+(q_i/q_j)c_{p(x)}]^2}{\sqrt{1-x^2}}
        \end{align}

    after substitution :math:`x=(q_i^2+q_j^2-p^2)/(2q_iq_j)`.
    The key of the integration method is to perform the inner integral
    in a way that acknowledges the square-root singularity in the
    denominator: the functions of :math:`p(x)` are approximated as
    piece-wise linear given by the grid points, and the resulting integrands
    of the form :math:`x^\alpha/\sqrt{1-x^2}` are calculated analytically.
    Specifically, one needs here

    .. math::
        :nowrap:

        \begin{align}
        \int dx\,\frac1{\sqrt{1-x^2}} &= \arcsin x\\
        \int dx\,\frac{x}{\sqrt{1-x^2}} &= -\sqrt{1-x^2}\\
        \int dx\,\frac{x^2}{\sqrt{1-x^2}} &= (1/2)\arcsin x-(x/2)\sqrt{1-x^2}\\
        \int dx\,\frac{x^3}{\sqrt{1-x^2}} &= -(1/3)\sqrt{1-x^2}(2+x^2)
        \end{align}

    Care is taken to extend the integrals to the interval :math:`x\in[-1,1]`,
    except where the upper cutoff of the wave-number grid is reached.

    The idea to perform the weighted integrals over piece-wise linear
    functions analytically goes back to Filon and Tuck who used it for the
    Fourier transform.

    .. [Caraglio2020] M. Caraglio, L. Schrack, G. Jung, and T. Franosch,
       Commun. Comput. Phys. 29, 628 (2021),
       `DOI:10.4208/cicp.OA-2020-0125 <https://doi.org/10.4208/cicp.OA-2020-0125>`_


Schematic Models
================

Schematic models are ad-hoc simplifcations of the MCT equations to one or
at most a few correlation functions. They are often formulated as "dropping
the wave number dependence" from the microscopic MCT models, but should be
more generally thought of as the result of applying bifurcation theory
to MCT, and realizing that the asymptotic dynamics close to the glass
transition is described by universal laws that can be reproduced from
schematic models.

Classic schematic models
------------------------

.. autoclass:: mctpy.schematic.generic
    :members:
    :inherited-members:

.. autoclass:: mctpy.f12model
    :members:
    :inherited-members:

.. autoclass:: mctpy.sjoegren_model
    :members:
    :inherited-members:

.. autoclass:: mctpy.bosse_krieger_model
    :members:
    :inherited-members:

Schematic Models for Sheared Systems
------------------------------------

.. autoclass:: mctpy.f12gammadot_model
    :members:
    :inherited-members:

    This model builds on the F12 model and deals with a single correlator
    with memory kernel
    :math:`m[\phi(t),t] = h(\dot\gamma t)[v_1 \phi(t) + v_2 \phi(t)^2]`.
    The strain-reduction function is defined to be
    :math:`h(\gamma)=1/(1+\gamma^2)`.
    This model was introduced by Fuchs and Cates as a schematic model to
    calculate the dynamics, especially flow curves under strong shear.


.. autoclass:: mctpy.f12gammadot_tensorial_model
    :members:
    :inherited-members:

    This model is the tensorial extension of
    :py:class:`mctpy.f12gammadot_model`.
    While it also works under simple shear, it is computationally more
    expensive since it needs to calculate the deformation tensors as
    matrix-exponentials of a given shear-rate tensor.

    The model is specified by a single correlator with memory kernel

    .. math::

        m[\phi(t),t] = h_{t,0}([\boldsymbol B])[v_1 \phi(t) + v_2\phi(t)^2]

    where the strain-reduction function is modeled as a function of
    the two non-trivial invariants of the Finger tensor,
    :math:`\boldsymbol B_{tt'}=\boldsymbol F_{tt'}\cdot\boldsymbol F^T_{tt'}`
    constructed from the deformation tensor
    :math:`\boldsymbol F_{tt'}=\exp[\boldsymbol\kappa\,(t-t')]`,
    valid in homogeneous stationary flow. Specifically,

    .. math::

        h_{tt'}([\boldsymbol B])=\frac{1}{1 + (\nu I_1(\boldsymbol B_{tt'})
            + (1-\nu)I_2(\boldsymbol B_{tt'}) - 3)/\gamma_c^2}

    where :math:`I_1(\boldsymbol B)=\text{tr}\boldsymbol B` and
    :math:`I_2(\boldsymbol B)=\text{tr}\boldsymbol B^{-1}` are the two
    invariants of the Finger tensor (valid in incompressible flow).

    If the parameter `use_hhat` is set, the model invokes a modified
    evolution equation for the correlator,

    .. math::

        \partial_t\phi(t) + \phi(t) + \hat h_{t0}([\boldsymbol B])\int_0^t
        m(t-t')\partial_{t'}\phi(t')\,dt' = 0

    with setting :math:`\hat h=h`. In this form, the model implemented here
    is the one used by Brader et al. [Brader2009]_. Note that since the
    memory kernel is evaluated at time :math:`t-t'`, this model implies
    two strain reduction factors, one given by :math:`\boldsymbol B_{t,t'}`,
    and one given by :math:`\boldsymbol B_{t,0}`.

    The :py:func:`mctpy.f12gammadot_tensorial_model.shear_modulus` function
    implements both an "upper-convected" and a "lower-convected" integral
    for the stress tensor. The standard variant is the UC model,

    .. math::

        \sigma(t)=\int_0^tdt'\,[-\partial_{t'}B_{tt'}]\phi(t-t')^2

    whilte the LC model is given by

    .. math::

        \sigma(t)=\int_0^tdt'\,[-\partial_tB^{-1}_{tt'}]\phi(t-t')^2

    The derivatives of the Finger tensor and its inverse are implemented
    through the formulas (valid in homogeneous flow),

    .. math::
        :nowrap:

        \begin{align}
        \partial_tB_{tt'}&=\kappa_t\cdot B_{tt'}+B_{tt'}\cdot\kappa_t^T\\
        -\partial_tB^{-1}_{tt'}&=\kappa_t^T\cdot B^{-1}_{tt'}
                                +B^{-1}_{tt'}\cdot\kappa_t
        \end{align}


    .. [Brader2009] J. M. Brader, T. Voigtmann, M. Fuchs, R. G. Larson,
       and M. E. Cates, Proc. Natl. Acad. Sci. (USA) 106, 15186 (2009),
       `DOI:10.1073/pnas.0905330106 <https://doi.org/10.1073/pnas.0905330106>`_

SCGLE
=====

The self-consistent generalized Langevin equation (SCGLE) is not strictly
speaking MCT, but the theories share so much of the mathematical structure
that they can be treated with the same numerical methods.

The main difference in this context is that SCGLE always treats models
to contain both the collective and the tagged-particle dynamics.
While in MCT, the tagged particle correlator is fully "enslaved" to the
collective dynamics, SCGLE starts from the tagged-particle picture and
presumes that the collective dynamics is essentially the same, decorated
with the static structure factor.

For this reason, SCGLE models are typically "vectorial" models with
a vector dimension of two in our context.

The MCT-like treatment of SCGLE in this code follows [Elizondo2019]_.

.. [Elizondo2019] L. F. Elizondo-Aguilera and Th. Voigtmann,
   Phys. Rev. E 100, 042601 (2019),
   `DOI:10.1103/PhysRevE.100.042601 <https://doi.org/10.1103/PhysRevE.100.042601>`_


Simple liquid in SCGLE
----------------------

.. autoclass:: mctpy.scgle.scgle_model
   :members:
   :inherited-members:


