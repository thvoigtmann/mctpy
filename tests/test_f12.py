import mctpy as mct
import numpy as np

class TestLambda:
    def test_exponents(self):
        """Test calculation of MCT exponents."""
        l = 0.74
        a, b = mct.util.exponents(l)
        assert np.abs(mct.util.lambda_func(a)-l) < 1e-8
        assert np.abs(mct.util.lambda_func(-b)-l) < 1e-8

class TestF12:
    def test_f12statics(self):
        """Run F12 model and solve for nonergodicity parameter."""
        model = mct.f12model (0,4)
        f = mct.nonergodicity_parameter (model)
        f.solve()
        assert np.abs(f.f-0.5)<1e-6
        assert np.abs(f.m-1.0)<1e-5
    def test_f12dynamics(self):
        """Run F12/Sjoegren model, test re-attaching a Sjoegren
        correlator to a pre-solved F12 correlator."""
        model = mct.f12model (0,3.95)
        model_s = mct.sjoegren_model(15.0,model)
        phi = mct.correlator (model=model, store=True)
        phi_s = mct.correlator (model=model_s, store=True)
        correlators = mct.CorrelatorStack([phi,phi_s])
        correlators.solve_all()
        model_s2 = mct.sjoegren_model(15.0,model)
        phi_s2 = mct.correlator (model=model_s2, store=True)
        correlators2 = mct.CorrelatorStack([phi,phi_s2])
        correlators2.solve_all()
        assert (phi_s2.phi==phi_s.phi).all()
    def test_shorttime(self):
        """Tests the short-time behavior of the dynamics.
        Should be exp(-t/tau) for Brownian, cos(omega t) for Newtonian.
        For damped-ND we currently use as initial condition that the
        first derivative is zero!?"""
        model = mct.f12model (0,0,tau=0.5,Omega=2.)
        phi = mct.correlator (model=model, store=True, Tend=2., blocksize=1024,
                              quiet=True)
        phi.solve_all()
        assert np.isclose(phi.phi[:,0],np.exp(-phi.t/model.tau),atol=1e-6).all()
        phi = mct.correlator (model=model, store=True, Tend=2., blocksize=1024,
                              motion_type = "newtonian", quiet=True)
        phi.solve_all()
        assert np.isclose(phi.phi[:,0],np.cos(model.Omega*phi.t),atol=1e-4).all()
        phi = mct.correlator (model=model, store=True, Tend=2., blocksize=1024,
                              motion_type = "damped_newtonian", quiet=True)
        phi.solve_all()
        #import matplotlib.pyplot as plt
        #plt.plot(phi.t,phi.phi[:,0])
        om = np.sqrt(model.Omega**2-1./model.tau**2/4)
        damped_ho_phid0 = np.exp(-phi.t/model.tau/2) * (
                np.cos(om*phi.t) + np.sin(om*phi.t)/(2*om*model.tau))
        #plt.plot(phi.t, damped_ho_phid0)
        assert np.isclose(phi.phi[:,0],damped_ho_phid0,atol=1e-4).all()
        #plt.show()

if __name__ == '__main__':
    for key,tst in TestF12.__dict__.items():
        if key.startswith('test_'):
            print("running",key)
            tst(TestF12())
