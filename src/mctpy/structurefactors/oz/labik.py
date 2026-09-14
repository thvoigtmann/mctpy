import scipy
import numpy as np

from ..interpolation import dataSq

class Sq (dataSq):
    """Static structure factor from numerical solution of OZ equation.

    This implements an iterative solution scheme for the OZ equation
    given a closure and a potential, as proposed by Labík, Malijevský,
    and Voňka [Molecular Physics 56, 709 (1985)], with an optional
    improved convergence scheme by Ng [J. Chem. Phys. 61, 2680 (1974)].

    Handle with care, the convergence of OZ solutions is famously tricky
    and will depend on a judicious choice of parameters and initial guesses.
    The scheme might converge to unphysical solutions."""
    def __init__ (self):
        pass
    def solve (self, closure, potential, rho, kT=1.0, Gr0=None, Gq0=None,
               N=2048, M=40, Rmax=50.0, maxiter=10000, maxiter2=2000, mix=0.5,
               mix2=0.5, accuracy=1e-5, use_ng=False, progress_bar=None):
        """Solve the OZ equation.

        Parameters
        ----------
        closure : Closure object
            Should implement r c(r) for the OZ closure, and
            the derivative of that closure with respect to r gamma(r).
        potential : Potential object
            Should implement the Boltzmann factor corresponding to the
            potential.
        rho : float
            Number density.
        kT : float, default 1.0
            Temperature.
        Gr0, Gq0 : array_like
            Initial guesses for Gamma(r) and Gamma(q). If not given,
            the initial guess is zero, which is good for reasonably low
            densities / high temperatures. A typical approach for
            high-density / low-temperature regions, where convergence is
            poor, is some sort of annealing, where the result from a
            previous iteration at higher temperature is fed here to start
            the iteration at lower temperature.
        N : int, default 2048
            Number of discretization points. Since the scheme relies on
            repeated FFT, should be a power of two.
        M : int, default 40
            Number of small-r points for which the solution will be
            treated with a nonlinear matrix equation in direct iteration,
            as outlined by Labík et al.
        Rmax : float, default 50.0
            Cutoff in r-space, units depend on the potential.
        maxiter : int, default 10000
            Maximum number of FFT-based iterations to perform.
        maxiter2 : int, default 2000
            Maximum number of direct-matrix iterations to perform for
            each FFT iteration.
        mix : float, default 0.5
            Mix-in factor for the FFT-based iteration scheme. Only `mix`
            fraction of the new solution will be used to update the
            previous guess. A common trick to stabilize badly converging
            iterations.
        mix2 : float, default 0.5
            Mix-in factor for the internal matrix iteration.
        accuracy : float, default 1e-5
            Target accuracy for the solution.
        use_ng : bool, default False
            Whether to use the update scheme by Ng.
        progress_bar : callable, optional
            Designed to be used with `tqdm` to provide visual feedback
            on the iteration progress.

        Returns
        -------
        Gr, Gq : array_like
            Resulting Gamma(r) and Gamma(q), can be used to continue
            the iterative solution at a lower temperature.
        """
        Gr = np.empty(N)
        Gq = np.empty(N)
        dr = Rmax/N
        dq = np.pi/N/dr
        r = (np.arange(N) + 0.5) * dr
        q = (np.arange(N) + 0.5) * dq
        if use_ng:
            f_n = np.empty((3,Gr.shape[0]))
            g_n = np.empty((3,Gr.shape[0]))

        Im = np.diag(np.ones(M))

        # Labik step 1: setup
        ephi = potential.exp(r, kT)
        if Gr0 is None:
            Gr[:] = 0.0
        else:
            Gr[:] = Gr0
        if Gq0 is None:
            Gq[:] = 0.0
        else:
            Gq[:] = Gq0

        CC = np.empty((M,M))
        rmix = mix
        converged = False
        iterations = 0
        self.errlog = np.ones(maxiter)+np.nan
        while (not converged and iterations < maxiter):
            if progress_bar is not None:
                progress_bar.update(1)
            rmix = mix

            # Labik step 2: calculate Cr, phi, Cq

            # Gamma^(n)(r) -> C^(n)(r)
            Cr = closure(r, ephi, Gr)

            # C^(n)(r) -> C^(n)(q)
            Cq = scipy.fft.dst(Cr, type=4) * 2*np.pi * dr

            # Labik step 3: choose M
            # could choose M differently in each iteration step?

            # Labik step 4: calculate CC matrix (M x M)
            # Labik equation (13)

            phi0 = closure.dGamma(r, ephi, Gr)
            phi0hat = scipy.fft.dct(phi0, type=2)

            for i in range(M):
                for j in range(M):
                    if j>i:
                        CC[i,j] = (phi0hat[j-i] - phi0hat[i+j+1])/(2*N)
                    else:
                        CC[i,j] = (phi0hat[i-j] - phi0hat[i+j+1])/(2*N)

            # setup matrix equation
            # Labik Eq. (14)
            tmp = rho*Cq[:M] / (q[:M] - rho*Cq[:M])
            A = Im - (tmp*(2+tmp))[:,None] * CC

            # Labik step 6, 7: inner iteration
            inner_converged = False
            iterations2 = 0
            while (not inner_converged and iterations2 < maxiter2):
                b = tmp*Cq[:M] - Gq[:M]

                ## solve dGq from A.dG = b
                dGq = np.linalg.solve(A, b)

                Gq[:M] += mix2 * dGq

                iterations2 += 1
                ierr = np.linalg.norm(dGq) # TODO normalize by M?
                if ierr < accuracy:
                    inner_converged = True

            # Labik step 8
            # OZ equation in Fourier transform for large r
            if ierr < 1e-2:
                Gq[M:] = (rho*Cq*Cq/(q - rho*Cq))[M:]
            else:
                # if inner-iteration error is too high: use FT for all r
                # and mix in only very little for next iteration
                Gq = (rho*Cq*Cq/(q - rho*Cq))
                rmix = 0.05

            # inverse FT
            Grn = scipy.fft.dst(Gq, type=4) / (4*np.pi * N * dr)

            err = np.linalg.norm(Gr-Grn)

            if use_ng:
                f_n[0,:] = f_n[1,:]
                f_n[1,:] = f_n[2,:]
                f_n[2,:] = Gr
                g_n[0,:] = g_n[1,:]
                g_n[1,:] = g_n[2,:]
                g_n[2,:] = Grn
            if use_ng and iterations > 2:
                d_n = g_n-f_n
                d_01 = d_n[2] - d_n[1]
                d_02 = d_n[2] - d_n[0]
                d01 = np.sum(d_01**2)
                d02 = np.sum(d_02**2)
                dn01 = np.sum(d_n[2]*d_01)
                dn02 = np.sum(d_n[2]*d_02)
                d0102 = np.sum(d_01*d_02)
                det = d01*d02 - d0102**2
                c1 = (d02*dn01 - d0102*dn02)/det
                c2 = (d01*dn02 - d0102*dn01)/det
                Gr = (1.0-c1-c2)*g_n[2] + c1*g_n[1] + c2*g_n[0]
            else:
                Gr = (1-rmix)*Gr + rmix*Grn
            self.errlog[iterations] = err
            iterations += 1
            if err < accuracy:
                converged = True

        # check consistency: Gamma(q) and C(q) should solve OZ equation
        self.oz_err = np.linalg.norm(Gq - rho*Cq*Cq/(q - rho*Cq))
        # Gamma(r) and C(r) should solve the closure
        self.cl_err = np.linalg.norm(Cr - closure(r,ephi,Gr))
        self.converged = (self.oz_err < accuracy) and (self.cl_err < accuracy)
        self.accuracy = accuracy

        self._r_data = r
        self._gr_data = (Gr+Cr)/r + 1
        self._q_data = q
        self._sq_data = (Gq+Cq)/q*rho + 1
        self.rho = rho
        #self._cq_data = rho*(Gq+Cq)/(q + rho*(Gq+Cq)) # not needed

        self._interpolator = scipy.interpolate.CubicSpline(self._q_data, self._sq_data)
        self._gr_interpolator = scipy.interpolate.CubicSpline(self._r_data, self._gr_data)
        return Gr, Gq

    def gr (self, r):
        return self.interpolate_sq (r, self._gr_interpolator, minval=0.0)
