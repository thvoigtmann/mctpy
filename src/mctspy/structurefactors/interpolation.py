import scipy

class dataSq:
    def __init__ (self, q, sq, density=1.0):
        self.rho = density
        self._q_data, self._sq_data = q, sq
        self._interpolator = scipy.interpolate.CubicSpline(self._q_data, self._sq_data)
    def density (self):
        return self.rho
    def interpolate_sq (self, q):
        #return scipy.interpolate.interpn(self._q_data, self._sq_data, q)
        return self._interpolator(q)
    def _cq_from_sq (self, sq):
        return ((1.-1./sq)/self.density())
    def cq(self, q):
        return self._cq_from_sq (self.interpolate_sq(q))
    def Sq(self, q):
        sq_ = self.interpolate_sq(q)
        return sq_, self._cq_from_sq (sq_)


