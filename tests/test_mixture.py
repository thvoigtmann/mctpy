import mctpy as mct
import numpy as np

class TestMixtureModel:
    def test_resetSq(self):
        """Test resetting S(q) in a mixture model."""
        qgrid = np.linspace(0.2,39.8,100)
        sq = mct.structurefactors.hsmPY([0.25,0.25],[1.0,0.6])
        model = mct.mixture_model (sq, qgrid)
        nep = mct.nonergodicity_parameter(model)
        nep.solve()

        sq2 = mct.structurefactors.hsmPY([0.84,0.97],[1.0,0.6])
        model.set(Sq=sq2)
        nep = mct.nonergodicity_parameter(model)
        nep.solve(accuracy=1e-8)

        model2 = mct.mixture_model (sq2, qgrid)
        nep2 = mct.nonergodicity_parameter(model2)
        nep2.solve(accuracy=1e-8)
        assert (nep2.f == nep.f).all()


