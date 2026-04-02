.. include:: <isolat1.txt>

Implementation Details
======================

Implementing Models
-------------------

Models that should work with :py:class:`mctpy.correlator` or
derived time-domain solvers need to implement the following
methods:

* :py:func:`mctpy.model_base.Aq` to specify the prefactor :math:`A_q`
    in front of the second time derivative in the evolution equation.
    For models implementing Newtonian dynamics, this will be
    For models implementing Brownian dynamics, this should return the
    symbol `None`.
* :py:func:`mctpy.model_base.Bq` to specify the prefactor :math:`B_q`
    in front of the first time derivative in the evolution equation.
    For models implementing Brownian dynamics, this will be
* Optionally, :py:func:`mctpy.model_base.Bqinv` can be implemented
    to avoid numerical inversion of :math:`B_q`.
* :py:func:`mctpy.model_base.Wq` to specify the prefactor :math:`W_q`
    in front of the correlator itself in the evolution equation.
* Optionally, :py:func:`mctpy.model_base.WqSq` can be implemented
    to return the product :math:`W_q \phi_{q,0}` directly.
* :py:func:`mctpy.model_base.phi0` to return the initial values of
    the function to solve for.
* Optionally, :py:func:`mctpy.model_base.phi0d` can return the
    initial derivative of the function to solve for. This is ignored
    if :math:`A_q` is set to zero (Brownian dynamics), since then
    the derivative is fixed by the other parameters.

Standard "Moment" Solver
------------------------

The commonly used solver for MCT equations (:py:class:`mctpy.correlator`)
is based on an algorithm first proposed by Fuchs and Hofacker [Fuchs1991]_
called the "moment algorithm". The key idea is the splitting of a
convolution integral over slowly varying functions at some half-way
time point according to

.. math::

    \begin{align}
    I&=\frac{d}{dt}\int_0^tm(t-t')F(t')dt'\\
    &=F(\bar t)m(\bar t)+\int_0^{\bar t}\dot m(t-t')F(t')\,dt'
    +\int_0^{t-\bar t}\dot F(t-t')m(t')\,dt'
    \end{align}
  
and then treating the integrals of the form

.. math::

    \int_{t_1}^{t_2}\dot A(t-t')B(t')\,dt'
    \approx [A(t-t_1)-A(t-t_2)]\frac1{t_2-t_1}\int_{t_1}^{t_2}B(t')\,dt'

where the second part of this expression is referred to as the
"moment" of the function :math:`B`.

The MCT equations are typically of the form

.. math::

    \begin{align}
    A_q\frac{d^2}{dt^2}\phi_q(t) &+B_q\frac{d}{dt}\phi_q(t)
        +W_q\phi_q(t)\\ &+\int_0^t m_q(t-t')\left[\frac{d}{dt'}\phi_q(t')
        +X_q\phi_q(t')\right]\,dt'=0
    \end{align}

with specific choices of coefficients and initial values defined by
a given MCT model.

Specifically, one writes the above convolution integral containing
the n-th derivative of :math:`\phi_q` as

.. math::

    \begin{align}
    I&=\int_0^{\bar t}\left[\partial_t^{(n)}m(t-t')\right]\phi(t')\,dt'\\
      &+\int_0^{t-\bar t}\left[\partial_t^{(n)}\phi(t-t')\right]m(t')\,dt'
       +B^{(n)}(t,\bar t)
    \end{align}


where boundary terms
:math:`B^{(1)}(t,\bar t)=m(t-\bar t)\phi(\bar t)-m(t)\phi(0)`
arise only for :math:`$n=1`, and
:math:`\bar t=\bar\imath h_d` with :math:`\bar\imath=\lfloor i/2\rfloor`.
Under some assumptions (that are not always fulfilled if oscillations are
present in the integrand, but are approximately true if
:math:`h_d` is small compared to any oscillation period) an extended
mean-value theorem can be used to rewrite this as

.. math::

    \sum_{k=1}^{\bar\imath}\left[h_d\partial_t^{(n)}m(t-t_k^{m*})\right]
        d\Phi_k+\sum_{k=1}^{i-\bar\imath}\left[h_d\partial_t^{(n)}
        \phi(t-t_k^*)\right]dM_k+B^{(n)}(t,\bar t)

where "moments" have been introduced,

.. math::

    dF_k:=\frac1{h_d}\int_{t_{k-1}}^{t_k}f(t')\,dt'

The unknown midvalue points :math:`t_k^*` and `t_k^{m*}` are then
approximated by one of the boundaries of the interval they are in,
and a simple two-point finite difference scheme for the derivatives
leads to

.. math::

    \sum_{k=1}^{\bar\imath}(m_{i-k+1}-m_{i-k})d\Phi_k
       +\sum_{k=1}^{i-\bar\imath}(\phi_{i-k+1}-\phi_{i-k})dM_k
       +m_{i-\bar\imath}\phi_{\bar\imath}-m_i\phi_0

for :math:`n=1`, and

.. math::

    \sum_{k=1}^{\bar\imath}(h_d/2)(m_{i-k+1}+m_{i-k})d\Phi_k
       +\sum_{k=1}^{i-\bar\imath}(h_d/2)(\phi_{i-k+1}+\phi_{i-k})dM_k

for :math:`n=0`. These sums then are the representation of the
convolution integral in the moment algorithm. Since :math:`m_i` depends
on :math:`\phi_i` itself in general,  extracting all terms containing either
value leads to an implicit equation that has to be solved for every grid
point :math:`i`, using all previously calculated values for grid points
:math:`j<i`.

In detail, the equation to solve reads

.. math::

    \tilde A\phi_i = \tilde Bm_i[\phi_i] - \tilde C_i

with coefficients

.. math::

    \begin{align}
      \tilde A &= W + \delta_d^+dM_1 + D_d\\
      \tilde B &= \phi_0 - \delta_d^+d\Phi_1\\
      \tilde C_i &= \Sigma_i + m_{i-\bar\imath}\phi_{\bar\imath}
        -\delta_d^-(m_{i-1}d\Phi_1+\phi_{i-1}dM_1)+\bar D_i\\
      \Sigma_i &= \sum_{k=2}^{\bar\imath}
        (\delta_d^+m_{i-k+1}-\delta_d^-m_{i-k})d\Phi_k
        +\sum_{k=2}^{i-\bar\imath}(\delta_d^+\phi_{i-k+1}-\delta_d^-\phi_{i-k})
        dM_k\\
      \delta_d^\pm &= 1\pm(h_d/2)X
    \end{align}

and :math:`D_d` and :math:`\bar D_i` taken from a discretization of the time
derivatives,

.. math::

    \begin{align}D_d&=2A/h_d^2+3B/(2h_d)\\
      \bar D_i&=(A/h_d^2)(-\phi_{i-3}+4\phi_{i-2}-5\phi_{i-1})
              +(B/h_d)(\phi_{i-2}/2-2\phi_{i-1})
    \end{align}

The finite-difference schemes employed for the derivatives need a few
previous grid points to exist always, and thus the very first grid
points on the first block are calculated from a direct short-time
expansion of the equations of motion.

The decimation procedure shows that moments can be transferred to
a coarser grid without loosing further accuracy. This makes clear the gist
of the moment algorithm: handling integrals over small times, where
functions may vary quickly, on coarse grids is difficult, but moments
allow us to keep the accuracy of the finest grid for such integrals.
The drawback is that in turn we have to accept an error coming from
the approximation of the midpoint value (which however should be reasonably
small), and that the algorithm uses more memory.




.. [Fuchs1991] M. Fuchs, W. G\ |ouml|\ tze, I. Hofacker, and A. Latz,
   J. Phys.: Condens. Matter 3, 5047 (1991)
   `DOI:10.1088/0953-8984/3/26/022 <https://doi.org/10.1088/0953-8984/3/26/022>`_

beta-Scaling Solvers
--------------------

The :py:class:`mctpy.beta_scaling_function` solver uses the same
idea of algorithm as used for the correlator, with a few modifications.
First, there is no separate memory kernel; second, the discretized
equation leads to a quadratic equation to be solved at each time step:

.. math::

    -\lambda g_i^2 + 2\,dG_1\,g_i + C = 0

The standard iteration scheme does not work well for this equation,
especially when :math:`g_i` becomes negative. Older versions of the
MCT solver code used a regula falsi scheme (this is still implemented
in the code, but not currently used). It turns out, that a direct
solution of the quadratic equation seems to work well.

In the case of SBR, the equation is modified to

.. math::
    -\lambda g_{x,i}^2 + 2(dG_{x,1} + d\alpha/dx^2)\,g_i
    +(C-\alpha/dx^2\,(g_{x+dx,i}-g_{x-dx,i})=0

We solve this equation by semi-implicit iteration: For each lattice point
:math:`x`, the quadratic equation is solved, assuming the terms
referring to different lattice points known and fixed; this is repeated
for all lattice points until convergence is reached.

The :math:`d`-dimensional lattice is treated with periodic boundary conditions,
and in higher dimensions the finite-difference laplacian is taken over
all unit vectors,

.. math::

    \Delta g\mapsto\frac1{dx^2}\left(
    \sum_{\alpha=1}^d(g_{x+e_\alpha dx}+g_{x-e_\alpha dx})-2dg_{x}\right)

The stencil is currently only implemented for one-, two-, and
three-dimensional lattices.

The initial values of the :math:`\beta`-correlator are given by

.. math::

    g(t)=(t/t_0)^{-a} + A_1\sigma\,(t/t_0)^a

where :math:`A_1 = 1/(2 (a\pi / \sin(a\pi) - \lambda))`. The solver initializes
with this formula, omitting the singular initial value; the initial moments
are calculated from their analytical formula which can include the integral
over the singularity at :math:`t=0`. Note that the correction term is
relevant to obtain the correct fluctuations in SBR.

Matrix-Valued Models
--------------------

The conventions for matrix-valued models follow those explained in detail
in [Voigtmann2003phd]_ for the MCT for mixtures. Essentially, nothing is
normalized to the static structure factors.

Bengtzelius trick
^^^^^^^^^^^^^^^^^

For the memory-kernel integration in the standard mixture model
:py:class:`mctpy.mixture_model`, we use an extension of the Bengtzelius
trick. We have

    .. math::

        \begin{align}
        \hat M_{q,\alpha\beta}[f,g]&=\nu
        \sum_{n=1}^3\Big[\phantom+\,
        \sum_{\alpha'\beta'}
        \sum_k a_{\alpha\alpha'\beta\beta'}^{(n)}(k)f_{k,\alpha'\beta'}
        \sum_{p(q,k)} b^{(n)}(q,p)g_{p,\alpha\beta}\\ 
        &\phantom{=p\sum_{n=1}^3\Big[}+
        \sum_{\alpha''\beta''}
        \sum_k a^{(n)}(k)f_{k,\alpha\beta}
        \sum_{p(q,k)}
          b_{\alpha\alpha''\beta\beta''}^{(n)}(q,p)g_{p,\alpha''\beta''}
        \\
        &\phantom{=\nu\sum_{n=1}^3\Big[}+
        \sum_{\alpha'\beta''}
        \sum_k a_{\alpha\alpha'}^{(n)}(k)f_{k,\alpha'\beta}
        \sum_{p(q,k)} b_{\beta\beta''}^{(n)}(q,p)g_{p,\alpha\beta''}\\
        &\phantom{=\nu\sum_{n=1}^3\Big[}+
        \sum_{\alpha''\beta'}
        \sum_k a_{\beta\beta'}^{(n)}(k)f_{k,\alpha\beta'}
        \sum_{p(q,k)} b_{\alpha\alpha''}^{(n)}(q,p)g_{p,\alpha''\beta}
        \;\phantom+\Big]
        \end{align}

    with :math:`\nu = (\Delta q)^2/(32\pi^2q^3)` as a prefactor. This
    expression does not assume symmetry under swapping :math:`f` with
    :math:`g` and is used for the calculation of the stability matrix.
    For the case of the memory kernel itself, the symmetry allows to further
    reduce to

    .. math::

        \begin{align}
        \hat M_{q,\alpha\beta}&=2\nu\sum_{n=1}^3
        \sum_{\gamma\delta}\Big[\phantom+\,
        \sum_k a_{\alpha\gamma\beta\delta}^{(n)}(k)\Phi_{k,\gamma\delta}
        \sum_{p(q,k)} b^{(n)}(q,p)\Phi_{p,\alpha\beta}\\ \nonumber
        &\phantom{=2\nu\sum_{n=1}^3\sum_{\gamma\delta}\Big[}+
        \sum_k a_{\alpha\gamma}^{(n)}(k)\Phi_{k,\gamma\beta}
        \sum_{p(q,k)} b_{\beta\delta}^{(n)}(q,p)\Phi_{p,\alpha\delta}
        \;\phantom+\Big]
        \end{align}

    In the above expressions, the Bengtzelius coefficients are

    .. math::

        \begin{align}
        a^{(1)}(k)&=k^5
        & a_{\alpha\beta}^{(1)}(k)&=k\tilde c_{\alpha\beta}(k)
        & a_{\alpha\beta\gamma\delta}^{(1)}(k)&=k^5\tilde c_{\alpha\beta}(k)
                                               \tilde c_{\gamma\delta}(k)\\
        a^{(2)}(k)&=-2k^3
        & a_{\alpha\beta}^{(2)}(k)&=-k^5\tilde c_{\alpha\beta}(k)
        & a_{\alpha\beta\gamma\delta}^{(2)}(k)&=2k^3\tilde c_{\alpha\beta}(k)
                                                \tilde c_{\gamma\delta}(k)\\
        a^{(3)}(k)&=k
        & a_{\alpha\beta}^{(3)}(k)&=2k^3\tilde c_{\alpha\beta}(k)
        & a_{\alpha\beta\gamma\delta}^{(3)}(k)&=k\tilde c_{\alpha\beta}(k)
                                             \tilde c_{\gamma\delta}(k)
        \\
        b^{(1)}(q,p)&=p
        & b_{\alpha\beta}^{(1)}(q,p)&=p(q^4\!-\!p^4)\tilde c_{\alpha\beta}(p)
        & b_{\alpha\beta\gamma\delta}^{(1)}(q,p)&=p\tilde c_{\alpha\beta}(p)
                                               \tilde c_{\gamma\delta}(p)\\
        b^{(2)}(q,p)&=p(q^2\!-\!p^2)
        & b_{\alpha\beta}^{(2)}(q,p)&=p\tilde c_{\alpha\beta}(p)
        & b_{\alpha\beta\gamma\delta}^{(2)}(q,p)&=p(q^2\!+\!p^2)
                      \tilde c_{\alpha\beta}(p)\tilde c_{\gamma\delta}(p)\\
        b^{(3)}(q,p)&=p(q^2\!-\!p^2)^2
        & b_{\alpha\beta}^{(3)}(q,p)&=p^3\tilde c_{\alpha\beta}(p)
        & b_{\alpha\beta\gamma\delta}^{(3)}(q,p)&=p(q^2\!+\!p^2)^2
                      \tilde c_{\alpha\beta}(p)\tilde c_{\gamma\delta}(p)
        \end{align}

    As in the scalar case, the performance gain known as Bengtzelius' trick
    comes from realizing that the inner sum can be reduced to a
    :math:`\mathcal O(1)` operation when its value for the previous
    outer wave number is known.



Critical amplitudes
^^^^^^^^^^^^^^^^^^^

In :py:func:`mctpy.mixture_model.make_dm` we implement

.. math::

    C_{\alpha\beta,q}[\boldsymbol H]=\frac{(\Delta q)^2}{32\pi^2q^5}
    \sum_{kp}\sum_{\alpha'\alpha''\beta'\beta''}
    kp\hat V_{\alpha\alpha'\alpha''}(q,k,p)
    \hat V_{\beta\beta'\beta''}(q,k,p)
    H_{\alpha'\beta'}(k)F_{\alpha''\beta''}(p)

The corresponding left-eigenvector uses the implementation in
:py:func:`mctpy.mixture_model.make_dmhat`,

.. math::

    \frac{(\Delta q)^2}{32\pi^2q^5}
    \sum_{qp}\sum_{\alpha\alpha''\beta\beta''}
    kp\hat V_{\alpha\alpha'\alpha''}(q,k,p)
    \hat V_{\beta\beta'\beta''}(q,k,p)
    \hat H_{\alpha\beta}(q)F_{\alpha''\beta''}(p)

This works together with the implementation of the eigenvalue solver for
matrices in :py:func:`mctpy.eigenvalue.solve`. There, we solve the
eigenvalue equation by iterating

.. math::

    \boldsymbol H^{(n+1)}=2(\boldsymbol S-\boldsymbol F)
    \boldsymbol M[\boldsymbol F,\boldsymbol H^{(n)}]
    (\boldsymbol S-\boldsymbol F)

where :math:`2\boldsymbol M[\boldsymbol F,\boldsymbol H]` is what needs
to be returned by the function returned by
:py:func:`mctpy.mixture_model.make_dm`.

Normalization of the matrix-valued eigenvectors follows the convention

.. math::

    \begin{align} \left(\hat{\boldsymbol H}_q,
    \boldsymbol H_q(\boldsymbol S_q-\boldsymbol F_q)^{-1}\boldsymbol H_q
    \right) &=1 \\
    \left(\hat{\boldsymbol H}_q,\boldsymbol H_q\right)&=1
    \end{align}

This implies that the comparison to the eigenvectors of the one-component
model should be done according to :math:`e_q(1-f^c_q)^2 = H_q/S_q` and
:math:`\hat e_q/(1-f^c_q)^2 = \hat H_q/S_q`.

With this normalization, the exponent parameter is given by

.. math::

    \lambda = \left(\hat{\boldsymbol H}_q,
    (\boldsymbol S_q-\boldsymbol F_q)\boldsymbol M_q[\boldsymbol H]
    (\boldsymbol S_q-\boldsymbol F_q)\right)

where for a bi-linear memory kernel, the same expression is valid for the
second variation as for the kernel itself (but note that we need to
take care of dividing by :math:`q^2` here since our implementation of the
memory kernel implicitly multiplies by that factor).



.. [Voigtmann2003phd] Th. Voigtmann, PhD thesis, TU Munich (2003),
   https://mediatum.ub.tum.de/603008
