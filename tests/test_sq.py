import mctpy as mct
import numpy as np
import h5py, io

class TestSqInterpolation:
    def test_dataSq(self):
        hssSq = mct.structurefactors.hssPY(0.4)
        qtab = np.linspace(0,50,400)
        hsstab,_ = hssSq.Sq(qtab)
        dataSq = mct.structurefactors.dataSq(qtab,hsstab,density=hssSq.density())
        q = np.linspace(0.2,39.8,100)
        sq, cq = hssSq.Sq(q)
        sqi, cqi = dataSq.Sq(q)
        assert np.isclose(sq, sqi, atol=5e-5).all()
        assert np.isclose(cq, cqi, atol=5e-5).all()
    def test_dataMixtureSq(self):
        hsmSq = mct.structurefactors.hsmPY(densities=[0.5,0.4,0.1],
                                           diameters=[1.0,0.8,0.6])
        qtab = np.linspace(0,50,400)
        hsmtab,_ = hsmSq.Sq(qtab)
        hsmtab_triu = np.triu(hsmtab).\
                reshape(hsmtab.shape[0],-1)[:,[0,1,2,4,5,8]]
        hsmtab_tril = np.tril(hsmtab).\
                reshape(hsmtab.shape[0],-1)[:,[0,3,4,6,7,8]]
        Sqmix = mct.structurefactors.dataMixtureSq(qtab, hsmtab,
                    density=hsmSq.densities,
                    normalization='unity', matrix_format='full')
        Sqmix_triu = mct.structurefactors.dataMixtureSq(qtab, hsmtab_triu,
                    density=hsmSq.densities,
                    normalization='unity', matrix_format='upper_triangular')
        Sqmix_tril = mct.structurefactors.dataMixtureSq(qtab, hsmtab_tril,
                    density=hsmSq.densities,
                    normalization='unity', matrix_format='lower_triangular')
        q = np.linspace(0.2,39.8,100)
        sqhsm, cqhsm = hsmSq.Sq(q)
        sqmix, cqmix = Sqmix.Sq(q)
        sqmix_triu, cqmix_triu = Sqmix_triu.Sq(q)
        sqmix_tril, cqmix_tril = Sqmix_tril.Sq(q)
        assert np.isclose(sqmix, sqmix_triu).all()
        assert np.isclose(sqmix, sqmix_tril).all()
        assert np.isclose(sqhsm, sqmix, atol=1e-5).all()
        assert np.isclose(hsmSq.dcq_dq(q), Sqmix.dcq_dq(q), atol=5e-4).all()
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

if __name__ == '__main__':
    TestSqInterpolation().test_dataSq()
    TestSqInterpolation().test_dataMixtureSq()
