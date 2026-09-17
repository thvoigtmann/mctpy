import mctpy as mct
import numpy as np
import h5py, io

class TestSqInterpolation:
    def test_interpolatedSq(self):
        """Test interpolation of two S(q) objects."""
        qgrid = np.linspace(0.2,39.8,100)
        gammalist = [0.5,0.501]
        sqlist = []
        for gamma in gammalist:
            sqlist.append (mct.structurefactors.hssPY(gamma))
        sq = mct.structurefactors.interpolatedSq(gammalist, sqlist)
        gamma = np.mean(gammalist)
        sqmid = mct.structurefactors.hssPY(gamma)
        cqmid = sqmid.cq(qgrid)
        cqlin = sq.set_gamma(gamma).cq(qgrid)
        assert (cqmid - cqlin < 1e-3).all()
        buf = io.BytesIO()
        with h5py.File(buf,'w') as f:
            sq.h5save(f)
        res = mct.h5.load_file(buf)
        assert (res['structurefactor'].cq(qgrid) == sq.cq(qgrid)).all()
