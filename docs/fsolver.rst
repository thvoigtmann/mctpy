==========================================
Non-Ergodicity Parameters and Eigenvectors
==========================================

The analysis of the long-time limit of the standard MCT equation leads
to an algebraic equation for the non-ergodicity parameter,

.. math::

    m[f](q) = \frac{f(q)}{1-f(q)}

where :math:`m[f](q)` is the value of the memory kernel evaluated at
the non-erdociity parameter. This non-linear implicit equation has potentially
many roots, and the largest (in some appropriate sense) of it is the
long-time limit of the correlation function. As the parameters entering
the memory kernel are changed, bifurcations appear in this equation,
and these are the MCT glass transition points. The proper analysis of these
bifurcations is via the analysis of the critical eigenvector of the stability
matrix of the equation.

We provide a number of functions to solve for the non-ergodicity parameter,
the associated eigenvalues, and related quantities.

.. include:: <isolat1.txt>

Nonergodicity Parameter
=======================

.. autoclass:: mctpy.nonergodicity_parameter
    :members:
    :inherited-members:

    In the case of scalar-valued models, the solver performs the
    iteration

    .. math::

        f_q^{(n+1)} = \frac{m_q[f^{(n)}]}{W_q + m_q[f^{(n)}]} \phi_{q,0}

    initialized with :math:`f_q^{(0)} = 1`, where :math:`W_q` and
    :math:`\phi_{q,0}` are model-specific (usually either both unity,
    or :math:`W_q=q^2` with an appropriate definition of the memory
    kernel). This iteration is guaranteed to converge towards the correct
    long-time limit of the correlation functions in standard MCT [1]_.


    In the case of matrix-valued correlators, the iteration is performed
    as

    .. math::

        \boldsymbol F_q^{(n+1)} = \boldsymbol\Phi_{q,0}
        - \left[\boldsymbol W_q + \boldsymbol M_q[\boldsymbol F^{(n)}]
        \right]^{-1}\cdot\boldsymbol W_q\cdot\boldsymbol\Phi_{q,0}

    where the product :math:`\boldsymbol W_q\cdot\boldsymbol\Phi_{q,0}`
    is supplied by the model separately. The background is that usually
    this term is a multiple of the unit matrix, so that the iteration
    written in the above form manifestly preserves symmetry of the solutions.

    A drawback of this iteration scheme is that the zero solution is not
    numerically obtained. The solver tries to detect such near-zero solutions
    and then switches to the iteration scheme that more closely matches
    the scalar case (but is not manifestly symmetric)

    .. math::

        \boldsymbol F_q^{(n+1)} =
        \left[\boldsymbol W_q + \boldsymbol M_q[\boldsymbol F^{(n)}]\right]^{-1}
        \cdot\boldsymbol M_q[\boldsymbol F^{(n)}] \cdot
        \boldsymbol\Phi_{q,0}

    This is implemented in the hope that true liquid solutions are
    represented as true zeros in memory, which is nice to have in case
    one calculates the eigenvalues to find liquid-glass bifurcations.

    As in the scalar case, it can be shown [2]_ that under certain plausible
    assumptions on the form of the MCT vertices appearing in the standard
    MCT memory kernel, the iteration converges to the correct (positive
    definite) solution describing the long-time limit of the correlation
    function.

    .. [1] W. G\ |ouml|\ tze and L. Sj\ |ouml|\ gren, J. Math. Analysis Appl. 195, 230 (1995), `DOI:10.1006/jmaa.1995.1352 <https://doi.org/10.1006/jmaa.1995.1352>`_

    .. [2] T. Franosch and Th. Voigtmann, J. Stat. Phys. 109, 237 (2002), `DOI:10.1023/A:1019991729106 <https://doi.org/10.1023/A:1019991729106>`_


Critical Eigenvector and -value
===============================

.. autoclass:: mctpy.eigenvalue
   :members:
   :inherited-members:


Distance parameter
==================

The MCT distance parameter :math:`\sigma` is calculated from the vertex
difference of a non-critical point to a given critical point. It is
defined as, for a scalar model

.. math::

   \sigma = \frac{\left(\hat e_q,(m_q[f^c] - m^c_q[f^c])\right)}
   {\left(\hat e_q,e_q(1-f^c_q)e_q\right)}

The method `mctpy.eigenvalue.sigma` allows to calculate that sum.
