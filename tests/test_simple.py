import mctpy as mct
import numpy as np
import h5py, io

class TestSimpleLiquid:

    def test_nep(self):
        qgrid = np.linspace(0.2,39.8,100)
        sq = mct.structurefactors.hssPY(0.5)
        model = mct.simple_liquid_model(sq, qgrid)
        nep = mct.nonergodicity_parameter (model)
        nep.solve()
        assert (nep.f==0).all()
        sq = mct.structurefactors.hssPY(0.516)
        model.set(Sq=sq)
        nep.solve()
        assert (nep.f>0).all()
        ev = mct.eigenvalue(nep)
        ev.solve()
        assert ev.lam > 0.5 and ev.lam < 1.0
        G = model.shear_modulus(nep.f,None)
        assert np.isclose(G,19.52,atol=1e-3)

    def test_correlator(self):
        qgrid = np.linspace(0.2,39.8,100)
        sq = mct.structurefactors.hssPY(0.5)
        model = mct.simple_liquid_model(sq, qgrid)
        phi = mct.correlator (model=model, store=True)
        phi.solve_all()

        buf = io.BytesIO()
        with h5py.File(buf,'w') as f:
            phi.save(f)

        phi = mct.h5.load_file(buf)

        cs = mct.structurefactors.hssPYtagged(phi=phi.model.Sq.phi, delta=1)
        model_s = mct.tagged_particle_model (model, Sq=cs)
        phi_s = mct.correlator (model=model_s, store=True)
        model_s0 = mct.tagged_particle_q0 (model_s)
        msd = mct.mean_squared_displacement (model=model_s0, store=True)
        phi_s.solve_all()
        msd.solve_all()

        # MSD should be linear at long times, thus dr^2/t ~ const
        # should have a low standard deviation for the last time points
        D_t = msd.phi[1:,0] / msd.t[1:]
        assert np.std(D_t[-100:]) < 1e-12

if __name__ == '__main__':
    TestSimpleLiquid().test_nep()
