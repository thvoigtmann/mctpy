import scipy
import numpy as np

from ..interpolation import dataMixtureSq

class mixtureSq (dataMixtureSq):
    """Static structure factor from numerical solution of OZ equation.

    This implements the matrix-valued version of an iterative solution
    scheme for the OZ equation given a closure and a potential. The basis
    is the :py:class:`mctpy.structurefactors.oz.Sq` implementation that
    follows Labík, Malijevský, and Voňka [Molecular Physics 56, 709 (1985)],
    with an optional improved convergence scheme by
    Ng [J. Chem. Phys. 61, 2680 (1974)].

    The extension from the original code to matrices is essentially
    done by extending from indices i,j for the q points to meta-indices
    (iab),(jgd) in the direct-matrix iteration.

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
        rho : array_like
            Partial number densities.
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
        dr = Rmax/N
        dq = np.pi/N/dr
        r = (np.arange(N) + 0.5) * dr
        q = (np.arange(N) + 0.5) * dq
        ephi = potential.exp(r, kT)
        rhomat = np.diag(np.array(rho))
        S = rhomat.shape[-1]
        #x = np.array(rho)/np.sum(np.array(rho))

        # Labik step 1: setup

        if Gr0 is None:
            Gr = np.zeros_like(ephi)
        else:
            Gr = Gr0.copy()
        if Gq0 is None:
            Gq = np.zeros_like(ephi)
        else:
            Gq = Gq0.copy()
        if use_ng:
            f_n = np.empty((3,*Gr.shape))
            g_n = np.empty((3,*Gr.shape))

        Is = np.diag(np.ones(S))
        Im = np.diag(np.ones(M))
        # I_{ij,ambn} = delta_{ij}delta_{ab}delta_{mn}
        #I = np.transpose((Im[...,None,None] * Is)[...,None,None] * Is,
        #                 axes=(0,1,2,4,3,5))
        # I_{ij,abmn} = delta_{ij}delta_{ab}delta_{mn}
        #I = (Im[...,None,None] * Is)[...,None,None] * Is
        # I_{iab,jmn} = delta_{ij}delta_{ab}delta_{mn}
        I = Im[:,None,None,:,None,None] \
                * Is[None,:,None,None,:,None] * Is[None,None,:,None,None,:]

        qI = q[:,None,None] * Is

        CC = np.empty((M,S,S,M,S,S))
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
            Cq = np.zeros_like(Cr)
            for a in range(S):
                for b in range(S):
                    Cq[:,a,b] = scipy.fft.dst(Cr[:,a,b], type=4) * 2*np.pi * dr

            # Labik step 3: choose M
            # could choose M differently in each iteration step?

            # Labik step 4: calculate CC matrix (M x M)
            # Labik equation (13)

            # ephi, Gr are (N,S,S)
            # phi0 should have shape (N,S,S,S,S)
            # and represent phi0_{i,abgd} = dF_{i,ab}/dGamma_{i,gd}
            phi0 = closure.dGamma(r, ephi, Gr)
            phi0hat = np.zeros_like(phi0)
            for a in range(S):
                for b in range(S):
                    for c in range(S):
                        for d in range(S):
                            phi0hat[:,a,b,c,d] = scipy.fft.dct(
                                    phi0[:,a,b,c,d], type=2)

            for i in range(M):
                for j in range(M):
                    if j>i:
                        CC[i,:,:,j,:,:] = (phi0hat[j-i] - phi0hat[i+j+1])/(2*N)
                    else:
                        CC[i,:,:,j,:,:] = (phi0hat[i-j] - phi0hat[i+j+1])/(2*N)

            # setup matrix equation
            # Labik Eq. (14)
            denom = np.linalg.inv(qI[:M] - Cq[:M] @ rhomat)
            C2 = Cq[:M] @ rhomat @ Cq[:M]
            tmp = denom @ C2
            # tensors need to be (q,a,b,m,n) = (q,(ab),(mn))
            rhoCq = rhomat @ Cq[:M]
            shape = (M*S*S)
            shape2 = (M*S*S,M*S*S)
            shape0 = (M,S,S)
            A = I - np.einsum('iag,igdjmn,jdb->iabjmn',denom,CC,rhoCq[:M]) \
                  - np.einsum('iag,igd,dd,idbjmn->iabjmn',denom,Cq[:M],rhomat,CC) \
                  - np.einsum('iag,gg,igljmn,ilb',denom,rhomat,CC,tmp)

            # Labik step 6, 7: inner iteration
            inner_converged = False
            iterations2 = 0
            while (not inner_converged and iterations2 < maxiter2):
                b = denom @ C2[:M] - Gq[:M]

                ## solve dGq from A.dG = b
                dGq = np.linalg.solve(A.reshape(shape2), b.reshape(shape)).reshape(shape0)

                Gq[:M] += mix2 * dGq

                iterations2 += 1
                ierr = np.linalg.norm(dGq) # TODO normalize by M?
                if ierr < accuracy:
                    inner_converged = True

            # Labik step 8
            # OZ equation in Fourier transform for large r
            if ierr < 1e-2:
                denom = np.linalg.inv(qI[M:] - Cq[M:]@rhomat)
                Gq[M:] = denom @ Cq[M:] @ rhomat @ Cq[M:]
            else:
                # if inner-iteration error is too high: use FT for all r
                # and mix in only very little for next iteration
                # TODO FIXME matrix
                denom = np.linalg.inv(qI - Cq[:]@rhomat)
                Gq[:] = denom @ Cq @ rhomat @ Cq
                rmix = 0.05

            # inverse FT
            Grn = np.zeros_like(Gr)
            for a in range(S):
                for b in range(S):
                    Grn[:,a,b] = scipy.fft.dst(Gq[:,a,b], type=4) / (4*np.pi * N * dr)

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
        self.rho = np.sum(rho)
        self.densities = rho
        denom = np.linalg.inv(qI - Cq@rhomat)
        self.oz_err = np.linalg.norm(Gq - denom @ Cq @ rhomat @ Cq)
        # Gamma(r) and C(r) should solve the closure
        self.cl_err = np.linalg.norm(Cr - closure(r,ephi,Gr))
        self.converged = (self.oz_err < accuracy) and (self.cl_err < accuracy)
        self.accuracy = accuracy

        self._r_data = r
        self._gr_data = (Gr+Cr)/r[:,None,None] + Is
        self._q_data = q
        self._sq_data = np.sqrt(rhomat) @ (Gq+Cq)/q[:,None,None] @ np.sqrt(rhomat) + Is

        self._interpolator = scipy.interpolate.CubicSpline(self._q_data, self._sq_data)
        self._gr_interpolator = scipy.interpolate.CubicSpline(self._r_data, self._gr_data)
        return Gr, Gq

    def gr (self, r):
        return self.interpolate_sq (r, self._gr_interpolator, minval=0.0)
