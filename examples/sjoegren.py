import numpy as np
import argparse
import matplotlib.pyplot as plt

import mctspy as mct
from numba import njit


parser = argparse.ArgumentParser()
parser.add_argument ('-v1',metavar='v1',help='vertex v1 of F12 model',
                     type=float, default=0.)
parser.add_argument ('-v2',metavar='v2',help='vertex v2 of F12 model',
                     type=float, default=3.95)
parser.add_argument ('-vs',metavar='vs',help='vertex vs of Sjoegren model',
                     type=float, default=15.0)
args = parser.parse_args()

class f12_with_sjoegren (mct.model_base):
    def __init__ (self, v1, v2, vs):
        self.v1 = v1
        self.v2 = v2
        self.vs = vs
    def __len__ (self):
        return 2
    def make_kernel (self, m, phi, i, t):
        v1, v2, vs = self.v1, self.v2, self.vs
        @njit
        def ker(m, phi, i, t):
            m[0] = v1 * phi[0] + v2 * phi[0]*phi[0]
            m[1] = vs * phi[0] * phi[1]
        return ker
    def make_dm (self, m, phi, dphi):
        v1, v2, vs = self.v1, self.v2, self.vs
        @njit
        def dm (m, phi, dphi):
            m[0] = (1-phi[0]) * (v1 + 2*v2 * phi[0])*dphi[0] * (1-phi[0])
            m[1] = (1-phi[1]) * vs * (phi[0]*dphi[1] + phi[1]*dphi[0]) * (1-phi[1])
        return dm
    def make_dmhat (self, m, f, ehat):
        v1, v2, vs = self.v1, self.v2, self.vs
        @njit
        def dm2(m, f, ehat):
            m[0] = (1-f[0]) * (v1 + 2*v2*f[0]) * ehat[0] * (1-f[0]) \
                 + (1-f[1]) * vs*f[1] * ehat[1] * (1-f[1])
            m[1] = (1-f[1])*  vs*f[0] * ehat[1] * (1-f[1])
        return dm2
    def make_dm2 (self, m, phi, dphi):
        v1, v2, vs = self.v1, self.v2, self.vs
        @njit
        def dm2(m, phi, dphi):
            m[0] = (1-phi[0]) * v2 * dphi[0]*dphi[0] * (1-phi[0])
            m[1] = (1-phi[1]) * vs * dphi[0]*dphi[1] * (1-phi[1])
        return dm2

model = f12_with_sjoegren (args.v1, args.v2, args.vs)
phi = mct.correlator (model = model, maxiter=1000000, blocksize=256, accuracy=1e-10, store=True)
correlators = [phi]

def output (d, istart, iend, correlator_array):
    print ("block",d,"\r",end='')

f = mct.nonergodicity_parameter (model)
f.solve()
print(f.f,f.m)

ev = mct.eigenvalue (f)
ev.solve()
print ("# eigenvalue = {:f} (check ehat: {:f})".format(ev.eval,ev.eval2))
print ("# e = {}".format(ev.e))
print ("# ehat = {}".format(ev.ehat))
print ("# lambda = {:f}".format(ev.lam))


phi.solve_all(correlators, callback=output)
print("")

plt.plot(phi.t,phi.phi[:,0])
plt.plot(phi.t,phi.phi[:,1])
plt.xscale('log')
plt.show()
