========================
Static Structure Factors
========================
.. include:: <isolat1.txt>

The common MCT approximations involve coupling coefficients that are
set by the equilibrium static structure of the fluid. Most prominently,
there enter the static structure factor :math:`S(q)` and the associated
direct correlation function (DCF) :math:`c(q)`, connected by the
Ornstein-Zernike equation. For a "simple liquid", i.e., a one-component
fluid of particles that are fully described by their positionts,
one has

.. math::

    S(q)=\frac1{1-\rho c(q)}


In the case of mixtures, the partial static structure factors are needed.
There are several conventions for these.
Note that the convention we use is

.. math::

    \boldsymbol S(q) = \left[1 - \boldsymbol\rho^{1/2}\cdot
        \boldsymbol c(q)\cdot\boldsymbol\rho^{1/2}\right]^{-1}

where :math:`\boldsymbol\rho` is the diagonal matrix with the partial
number densities as its entries. This implies that the structure-factor
matrix is normalized such that for large wave number it approaches the
unit matrix. The conventional static structure factor is obtained
by multiplying with :math:`\boldsymbol\rho^{1/2}` from both sides again.


Simple Liquids (3D)
===================

Hard-Sphere Percus Yevick
-------------------------

.. autoclass:: mctpy.structurefactors.hssPY
    :members:
    :inherited-members:

    The analytical expression for the direct correlation function used
    here was derived by Wertheim [Wertheim1963]_:

    .. math::
        :nowrap:

        \begin{align}
        c(q) &= 4\pi\alpha\left(\frac{\cos q}{q^2}-\frac{\sin q}{q^3}\right)
        + 2\pi\alpha\varphi\left(\frac{\cos q}{q^2}-4\frac{\sin q}{q^3}
        - 12\frac{\cos q}{q^4}+24\frac{\sin q}{q^5}-24\frac{1-\cos q}{q^6}
        \right)\\
        &+ 8\pi\beta\left(\frac12\frac{\cos q}{q^2}-\frac{\sin q}{q^3}
        + \frac{1-\cos q}{q^4}\right) \end{align}

    where :math:`\alpha=(1+2\varphi)^2/(1-\varphi)^4` and
    :math:`\beta=-6\varphi(1+\varphi/2)^2/(1-\varphi)^4`.

    For low :math:`q`, an expansion up to :math:`\mathcal O(q^4)` is used:

    .. math::

        c(q)=-\left(\pi\alpha(4+\varphi)/3+\pi\beta\right)
        + q^2\left(\pi\alpha(2/15+\varphi/24)+\pi\beta/9\right)
        - q^4\left(\pi\alpha(1/210+\varphi/600)+\pi\beta/240\right)
        + {\mathcal O}(q^6)

    The derivative of the DCF is also implemented analytically,

    .. math::
        :nowrap:

        \begin{align}
        c'(q)&=-4\pi\alpha\left(\frac{\sin q}{q^2}+\frac3q\left(
        \frac{\cos q}{q^2}-\frac{\sin q}{q^3}\right)\right)
        -2\pi\alpha\varphi\left(\frac{\sin q}{q^2}+\frac6q\left(
        \frac{\cos q}{q^2}-4\frac{\sin q}{q^3}-\frac{12\cos q}{q^4}
        -\frac{24(1-\cos q-q\sin q)}{q^6}\right)\right)\\
        &\qquad-8\pi\beta\left(\frac12\frac{\sin q}{q^2}+\frac2q\left(
        \frac{\cos q}{q^2}+2\frac{1-\cos q-q\sin q}{q^4}\right)\right)
        \end{align}

    and for low values of the wave number

    .. math::

        c'(q)=\left(\pi\alpha\left(\frac4{15}+\frac{\varphi}{12}\right)
        +\frac{2\pi\beta}{9}\right)q - \left(\pi\alpha\left(\frac{2}{105}
        +\frac{\varphi}{150}\right)+\frac{\pi\beta}{60}\right)q^3


    .. [Wertheim1963] M. S. Wertheim, Phys. Rev. Lett. 10, 321 (1963),
       `DOI:10.1103/PhysRevLett.10.321 <https://doi.org/10.1103/PhysRevLett.10.321>`_


Tagged-particle HSPY
--------------------

.. autoclass:: mctpy.structurefactors.hssPYtagged
    :members:
    :inherited-members:

    The expression implemented here is derived from the analytical result
    obtained by Baxter for the hard-sphere binary mixture, taking the limit
    of one species density to zero. In particular,

    .. math::
        :nowrap:

        \begin{align}
        c^s(q) &= 4\pi\left(A\frac{\cos(q/2)\cos(q\delta/2)-\sin(q/2)
        \sin(q\delta/2)}{q^2}-B\frac{\cos(q\delta/2)\sin(q/2)+\cos(q/2)
        \sin(q\delta/2)}{q^3}-D\frac{\sin(q/2)\sin(q\delta/2)}{q^4}\right)\\
        &-\frac{4\pi a_2}{q^6}\left(q\cos(q/2)-2\sin(q/2)\right)
        \left(q\delta\cos(q\delta/2)-2\sin(q\delta/2)\right)
        \end{align}

    with :math:`A=(1/2)(1-\varphi+\delta(1+2\varphi))/(1-\varphi)^2`,
    :math:`B=((1-\varphi)^2-3\delta\varphi(1+2\varphi))/(1-\varphi)^3`,
    :math:`D=6\varphi(2+\varphi+\delta(1+2\varphi))/(1-\varphi)^3`,
    :math:`a_2=6\varphi/\pi/(1-\varphi)^2(1+6\varphi/(1-\varphi)+9(\varphi/(1-\varphi))^2)`.

    For low :math:`q`, an expansion is used,

    .. math::
        :nowrap:

        \begin{align}
        c^s(q) &= -\frac{\pi}6\left(3A(1+\delta)^2-\frac{B}{2}(1+\delta)^3
        -\frac{D}{4}\delta(1+\delta^2)+\frac{\delta^3\varphi(1+2\varphi)}
        {(1-\varphi)^4}\right)\\
        &+\frac{\pi}{240}\left(\frac52A(1+\delta)^4-\frac{B}{4}(1+\delta)^5
        -\frac{D}{24}\delta(3+10\delta^2+3\delta^4)
        +\frac{(\delta^3+\delta^5)\varphi(1+2\varphi)^2}{(1-\varphi)^4}
        \right)q^2\\
        &-\frac{\pi}{26880}\left(\frac73A(1+\delta)^6-\frac{B}{6}(1+\delta)^7
        -\frac{D}{12}\delta(1+7\delta^2+7\delta^4+\delta^6)
        +\frac{\delta^3\varphi(5+14\delta^2+5\delta^4)(1+2\varphi)^2}
        {5(1-\varphi)^4}\right)q^4
        \end{align}

Hard-Sphere Verlet-Weis
-----------------------

.. autoclass:: mctpy.structurefactors.hssVW
    :members:
    :inherited-members:

    Verlet and Weis [Verlet1972]_ proposed to correct the Percus-Yevick (PY)
    structure factor essentially by evaluating it at an effective
    density, and on effective wave numbers. The adjustment is calculated
    on the basis of the radial distribution function, for which
    the formulas by Wertheim [Wertheim1963]_ are used. In terms of the total
    correlation function, one gets

    .. math::

        h_\text{VW}(q d,\varphi)=\frac{\varphi_\text{eff}}\varphi
        h_\text{PY}(q d_\text{eff},\varphi_\text{eff})-I(q)
        +\frac{4\pi A}{q}\frac{q^3\cos(q d)+\mu(q^2+2\mu^2)\sin(q d)}
        {q^4+4\mu^4}

    with

    .. math::

        I(q)=\frac{4\pi}{q d_\text{eff}}d_\text{eff}^3
        \int_1^{d/d_\text{eff}}\sin(q d_\text{eff}x)x
        g_\text{PY}(x,\eta_\text{eff})\,dx

    Here, the Percus-Yevick RDF :math:`g_\text{PY}(r)` is known in the relevant
    regime from Wertheim's result.
    The integral :math:`I(q)` can thus be solved in terms of elementary
    functions,

    .. math::

        I(q)=\frac{4\pi}{q d_\text{eff}}\frac{d_\text{eff}^3}
        {(1-\eta_\text{eff})^2}\sum_{i=0}^2G_ii_i(qd_\text{eff})

    where the :math:`G_i` and :math:`\mu`, :math:`A` are constants and
    the :math:`i_{0,1,2}(q)` are functions that we implement explicitly,

    .. math::
        :nowrap:

        \begin{align}
        i_0(qd_\text{eff})&=\left.e^{\gamma x}\frac{\gamma\sin(qd_\text{eff}x)
        -qd_\text{eff}\cos(qd_\text{eff}x)}{(qd_\text{eff})^2+\gamma^2}
        \right|_{x=1}^{x=d/d_\text{eff}} \\
        i_1(qd_\text{eff})&=\left.e^{-\delta x}\left(
        \frac{\cos(q_+x)(\delta\sin\kappa-q_+\cos\kappa)}{q_+^2+\delta^2}
        -\frac{\sin(q_+x)(\delta\cos\kappa+q_+\sin\kappa)}{q_+^2+\delta^2}
        -\frac{q_-\cos(q_-x+\kappa)+\delta\sin(q_-x+\kappa)}{q_-^2+\delta^2}
        \right)\right|_{x=1}^{x=d/d_\text{eff}} \\
        i_2(qd_\text{eff})&=\left.e^{-\delta x}\left(
        \frac{\cos(q_+x)(\delta\cos\kappa+q_+\sin\kappa)}{q_+^2+\delta^2}
        +\frac{\sin(q_+x)(\delta\sin\kappa-q_+\cos\kappa)}{q_+^2+\delta^2}
        +\frac{q_-\sin(q_-x+\kappa)-\delta\cos(q_-x+\kappa)}{q_-^2+\delta^2}
        \right)\right|_{x=1}^{x=d/d_\text{eff}}
        \end{align}

    The structure factor and the direct correlation function are thus
    obtained from the total correlation function,

    .. math::
        :nowrap:

        \begin{align} c(q)&=\frac{h(q)}{1+\rho h(q)} \\
        S(q) &= 1 + \rho h(q) \end{align}

    The derivatives are calculated from

    .. math::

        c'(q)=\frac{h'(q)}{(1+\rho h(q))*2}

    where the derivative of the total correlation function is obtained
    from simple numerical differentiation (although this could in principle
    be also calculated analytically).

    .. [Verlet1972] L. Verlet and J.-J. Weis, Phys. Rev. A 5, 939 (1972),
       `DOI:10.1103/PhysRevA.2.939 <https://doi.org/10.1103/PhysRevA.2.939>`_


Square Well System (MSA)
------------------------

.. autoclass:: mctpy.structurefactors.swsMSA
    :members:
    :inherited-members:

    The expression is valid to first order in small well widths and
    was derived in [Dawson2001]_.
    The current implementation directly uses the Baxter functions as outlined
    in the paper; one writes :math:`S(q)^{-1}=Q_qQ_q^*` with

    .. math::

        Q_q=1-2\pi\rho\int_0^\infty dr\,\exp[iqr]Q(r)

    and the approximation consists in specifying :math:`Q(r)`, which is
    expected to be short-ranged. In the MSA for the square-well system,
    the function :math:`Q(r)` in fact is non-zero only over the ranges
    :math:`[0,\delta]`, :math:`[\delta,1]`, and :math:`[1,1+\delta]`,
    and specified by polynomials of at most degree 3. The Fourier integral
    over these polynomials is implemented analytically.

    Likewise, for the derivative of the direct correlation function,
    one uses that :math:`\partial_qQ_q` can be expressed by similar integrals,
    up to one order higher in the polynomials in :math:`Q(r)`.


    .. [Dawson2001] K. Dawson, G. Foffi, M. Fuchs, W. G\ |ouml|\ tze,
       F. Sciortino, M. Sperl, P. Tartaglia, Th. Voigtmann, and
       E. Zaccarelli, Phys. Rev. E 63, 011401 (2000),
       `DOI:10.1103/PhysRevE.63.011401 <https://doi.org/10.1103/PhysRevE.63.011401>`_

Generalized Gaussian core model (GCM4)
--------------------------------------

.. autoclass:: mctpy.structurefactors.gcm4MSA
    :members:
    :inherited-members:

    The generalized Gaussian core model GCM4 is

    .. math::

        V(r) = \epsilon e^{-(r/\sigma)^4}

    and in the mean-spherical approximation (MSA), we directly have
    :math:`c(q) = -\beta V(q)`. In three spatial dimensions, the
    Fourier transform of the GCM4 pair potential can be evaluated in terms
    of generalized hypergeometric functions.

    .. math::

        V(q) = \frac{4\pi}{q} \int_0^\infty r\,\sin(qr)e^{-r^4}\,dr

    This expression has been evaluated to high numerical precision
    with Wolfram Mathematica and tabulated, so that the structure factor
    class can interpolate to the desired values. Also :math:`V'(q)`
    can be evaluated in terms of generalized hypergeometric functions.


Lorentz-Gas direct correlation function
---------------------------------------

.. autoclass:: mctpy.structurefactors.lgDCF
    :members:
    :inherited-members:

    The Lorentz gas of overlapping hard spheres consists of randomly
    distributed, fixed hard-sphere obstacle, through which a point obstacle
    moves. The hard spheres are of radius :math:`R` obstacle and may overlap;
    in essence, :math:`R` is the distance of closest possible approach between
    the point particle and the center of any scattering sphere.

    For the purpose of MCT calculations, the mixted static structure factor,
    :math:`S^s(q)=\sum_i\langle\exp(-i\vec q\cdot\vec r_s)
    \exp(i\vec q\cdot\vec r_i)\rangle` (where :math:`\vec r_i` are the
    scatterer positions), plays the role of the direct correlation function
    (once divided by the density). Direct evaluation leads to set

    .. math::

        c^s(q) = \frac{4\pi}{q}R^2j_1(qR)

    where :math:`j_1(x)` denotes the spherical Bessel function of order 1.
    The expression approaches a constant value of
    :math:`(4\pi/3)R^3` as :math:`q\to0`.

    This object is to implement this simple expression, essentially as a
    wrapper around library-provided code for the spherical Bessel function,
    so that the Lorentz-gas expression can be used in appropriate
    (tagged-particle) MCT models.


Simple Liquids (2D)
===================

Hard Spheres (FMT)
-----------------

.. autoclass:: mctpy.structurefactors.hssFMT2d
    :members:
    :inherited-members:

    The expression implemented here was given by
    Thorneywork et al. [Thorneywork2018]_

    .. math::
        :nowrap:

        \begin{align}
        c(q) &= \frac{\pi}{6(1-\varphi)^3}q^2\left[
        -\frac54(1-\varphi)^2q^2J_0(q/2)^2
        \right. \\ &\left.
        +\left(4((\varphi-20)\varphi+7)+\frac54(1-\varphi)^2q^2\right)
        J_1(q/2)^2 \right.\\ &\left.
        +2(\varphi-13)(1-\varphi)qJ_1(q/2)J_0(q/2)\right]
        \end{align}

    with the Bessel functions :math:`J_n(x)`. For :math:`q=0` (technically,
    any input value less than machine-epsilon), the analytical limit is
    returned,

    .. math::

        c(0) = -\frac{\pi}{4}\frac{4-3\varphi+\varphi*2}{1-\varphi)^3}

    which is consistent with :math:`S(0)=(1-\varphi)^3/(1+\varphi)`, the
    equation of state from scaled-particle theory.

 
    .. [Thorneywork2018] A. L. Thorneywork, S. K. Schnyer, D. G. A. L. Aarts,
           J. Horbach, R. Roth, R. P. A. Dullens,
           Molec. Phys. 116, 3245 (2018), `DOI:10.1080/00268976.2018.1492745 <https://doi.org/10.1080/00268976.2018.1492745>`_


Mixtures (3D)
=============

Hard-Sphere Mixtures
--------------------

.. autoclass:: mctpy.structurefactors.hsmPY
    :members:
    :inherited-members:

    This implements the Percus-Yevick (PY) approximation for the
    structure factors of hard-sphere mixtures with a user-determined
    number of species, specified through their number densities
    :math:`\rho_\alpha` and their diameters :math:`d_\alpha`
    (where Greek indices label the species).

    Note that in contrast to the one-component HSS-PY structure factor,
    :py:class:`mctpy.structurefactors.hssPY`, we do not work with
    packing fractions here, but with the number densities.

    The implementation is based on analytic expressions that follow from
    Fourier transforming the real-space expressions derived by Baxter
    [Baxter1970]_ using a Wiener-Hopf factorization method.
    This gives the DCF :math:`c_{\alpha\beta}(q)` directly, so that the
    structure factor matrix follows from the Ornstein-Zernike equation.
    For small wave numbers, these analytic expressions are expanded to
    yield numerically more stable results

    In detail, the PY approximation for the hard-sphere mixture is thus
    given by

    .. math::
        :nowrap:

        \begin{align} c_{\alpha\beta}(q) &= -4\pi\left(
        A_{\alpha\beta}\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)-
        \cos(qd_\alpha/2)\cos(qd_\beta/2)}{q^2}
        \right.\\ &\left.
        +B_{\alpha\beta}\frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)+
        \cos(qd_\beta/2)\sin(qd_\alpha/2)}{q^3}
        \right.\\ &\left.
        +D_{\alpha\beta}\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)}{q^4}
        \right.\\ &\left.
        +\frac{a_2\pi}{q^4}\left(
        \cos(qd_\alpha/2)\cos(qd_\beta/2)d_\alpha d_\beta
        +\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)}{q^2}
        -\frac{2(\cos(qd_\alpha/2)\sin(qd_\beta/2)d_\alpha
        +\cos(qd_\beta/2)\sin(qd_\alpha/2)d_\beta)}{q}\right)\right)
        \end{align}

    with parameters

    .. math::
        :nowrap:

        \begin{align} A_{\alpha\beta}&=\frac{d_{\alpha\beta}(1-\xi_3)
        +(3/2)d_\alpha d_\beta\xi_2}{(1-\xi_3)^2} \\
        B_{\alpha\beta}&=\frac1{1-\xi_3}-\beta_0 d_\alpha d_\beta \\
        D_{\alpha\beta}&=\frac{6\xi_2}{(1-\xi_3)^2}+4d_{\alpha\beta}\beta_0 \\
        a_2&=\frac6\pi\frac{\xi_0}{(1-\xi_3)^2}+\frac{36}\pi\frac{\xi_1\xi_2}
        {(1-\xi_3)^3}+\frac{54}\pi\frac{\xi_2^3}{(1-\xi_3)^4} \\
        \beta_0&=(9\xi_2^2+3\xi_1(1-\xi_3))/(1-\xi_3)^3\end{align}

    and :math:`d_{\alpha\beta}=(d_\alpha+d_\beta)/2`,
    :math:`\xi_n=(\pi/6)\sum_\gamma\varrho_\gamma d_\gamma^n`.

    For low :math:`q`, the implementation here uses an analytical expansion
    to avoid cancellation errors. Specifically,

    .. math::
        :nowrap:

        \begin{align} c_{\alpha\beta}(q)&=c_{\alpha\beta}^{(0)}
        +\frac12c_{\alpha\beta}^{(2)}q^2+\frac1{4!}c_{\alpha\beta}^{(4)}q^4
        +{\mathcal O}(q^6) \\
        c_{\alpha\beta}^{(0)}&=-\frac\pi6\left(
        \frac\pi6a_2(d_\alpha d_\beta)^3+2(d_\alpha+d_\beta)^2A_{\alpha\beta}
        +\frac12(d_\alpha d_\beta)^2D_{\alpha\beta}\right)\\
        c_{\alpha\beta}^{(2)}&=\frac\pi{60}\left(
        A_{\alpha\beta}(d_\alpha+d_\beta)^4+\frac16D_{\alpha\beta}
        (d_\alpha d_\beta)^3+\left(\frac14D_{\alpha\beta}+\frac\pi{12}a_2
        d_\alpha d_\beta\right)(d_\alpha d_\beta)^2(d_\alpha^2+d_\beta^2)
        \right)\\
        c_{\alpha\beta}^{(4)}&=-\frac\pi{560}\left(
        A_{\alpha\beta}(d_\alpha+d_\beta)^6+\left(\frac14D_{\alpha\beta}
        +\frac\pi{12}a_2d_\alpha d_\beta\right)(d_\alpha d_\beta)^2
        (d_\alpha^4+d_\beta^4)+\frac13\left(D_{\alpha\beta}(d_\alpha
        +d_\beta)^2+\frac{7\pi}{10}a_2(d_\alpha d_\beta)^2\right)
        (d_\alpha d_\beta)^3+\frac16D_{\alpha\beta}(d_\alpha d_\beta)^4
        \right)\end{align}

    Also the derivative of the DCF is implemented analytically,

    .. math::
        :nowrap:

        \begin{align} c'_{\alpha\beta}(q)&=4\pi\left(
        -A_{\alpha\beta}\frac{d_\alpha+d_\beta}2
        \frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)
        +\cos(qd_\beta/2)\sin(qd_\alpha/2)}{q^2}
        \right. \\ &\left.
        -\left(2A_{\alpha\beta}+B_{\alpha\beta}\frac{d_\alpha+d_\beta}2\right)
        \frac{\cos(qd_\alpha/2)\cos(qd_\beta/2)
        -\sin(qd_\alpha/2)\sin(qd_\beta/2)}{q^3}
        +3B_{\alpha\beta}\frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)
        +\cos(qd_\beta/2)\sin(qd_\alpha/2)}{q^4}
        \right. \\ &\left.
        -D_{\alpha\beta}\frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)d_\alpha
        +\cos(qd_\beta/2)\sin(qd_\alpha/2)d_\beta}{2q^4}
        +D_{\alpha\beta}\frac{4\sin(qd_\alpha/2)\sin(qd_\beta/2)}{q^5}
        \right. \\ &\left.
        +\pi a_2\frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)d_\alpha d_\beta^2
        +\cos(qd_\beta/2)\sin(qd_\alpha/2)d_\beta d_\alpha^2}{2q^4}
        +6\pi a_2\frac{\cos(qd_\alpha/2)\cos(qd_\beta/2)d_\alpha d_\beta}{q^5}
        \right. \\ &\left.
        -\pi a_2\frac{\sin(qd_\alpha/2)\sin(qd_\beta/2)(d_\alpha^2+d_\beta^2)}
          {q^5}
        -12\pi a_2\frac{\cos(qd_\alpha/2)\sin(qd_\beta/2)d_\alpha
         +\cos(qd_\beta/2)\sin(qd_\alpha/2)d_\beta-2\sin(qd_\alpha/2)
         \sin(qd_\beta/2)/q}{q^6}
         \right)
         \end{align}

    with its low-wave-number expansion

    .. math::
        :nowrap:

        \begin{multline}
        c'_{\alpha\beta}(q)=\frac{\pi}{60}\left(
        A_{\alpha\beta}(d_\alpha+d_\beta)^4+\frac16D_{\alpha\beta}
        (d_\alpha d_\beta)^3+\left(\frac14D_{\alpha\beta}+\frac{a_2\pi}{12}
        (d_\alpha d_\beta)\right)
        (d_\alpha d_\beta)^2(d_\alpha^2+d_\beta^2)\right)q\\
        -\frac1{3!}\frac{\pi}{560}\left(
        A_{\alpha\beta}(d_\alpha+d_\beta)^6+\left(\frac14D_{\alpha\beta}
        +\frac{\pi}{12}a_2(d_\alpha d_\beta)\right)
        (d_\alpha d_\beta)^2(d_\alpha^4+d_\beta^4)
        +\frac13\left(D_{\alpha\beta}(d_\alpha+d_\beta)^2+\frac{7\pi}{10}a_2
        (d_\alpha d_\beta)^2\right) (d_\alpha d_\beta)^3
        +\frac16 D_{\alpha\beta}(d_\alpha d_\beta)^4
        \right)q^3+{\mathcal O}(q^5)
        \end{multline}


    .. [Baxter1970] R. J. Baxter, J. Chem. Phys. 52, 4559 (1970),
       `DOI:10.1063/1.1673684 <https://doi.org/10.1063/1.1673684>`_

