import numpy as np
import argparse
import matplotlib.pyplot as plt

import mctspy as mct
from numba import njit

# try -v1 2.9411 -v2 0.130327 -v3 45 to test critical point
# Fig 9 of Goetze and Sperl PRE 2002

parser = argparse.ArgumentParser()
parser.add_argument ('-v1',metavar='v1',help='vertex v1 of BK model',
                     type=float, default=2.9254)
parser.add_argument ('-v2',metavar='v2',help='vertex v2 of BK model',
                     type=float, default=0.1292)
parser.add_argument ('-v3',metavar='v3',help='vertex v3 of BK model',
                     type=float, default=45.0)
args = parser.parse_args()

model = mct.bosse_krieger_model (args.v1, args.v2, args.v3)
phi = mct.correlator (model = model, maxiter=1000000, blocksize=256, accuracy=1e-10, store=True)
correlators = [phi]

def output (d, istart, iend, correlator_array, filter=0):
    first = correlator_array[0]
    for i in range(istart,iend):
        if not (filter and (i%filter)):
            print ("{t:.15f} ".format(t=i*first.h),end='')
            for correlator in correlator_array:
                for q in range(len(correlator.phi[i])):
                    print ("{phi:.15f} ".format(phi=correlator.phi[i][q]),end='')
                for q in range(len(correlator.m[i])):
                    print ("{m:.15f} ".format(m=correlator.m[i][q]),end='')
            print ("#")
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
print ("# (ehat,e) = {}".format(np.dot(ev.ehat,ev.e)))
print ("# (ehat,e*e*(1-f)) = {}".format(np.dot(ev.ehat,ev.e*ev.e*(1-f.f[0]))))
print ("# lambda = {:f}".format(ev.lam))


phi.solve_all(correlators, callback=output)
print("")

plt.plot(phi.t,phi.phi[:,0])
plt.plot(phi.t,phi.phi[:,1])
plt.xscale('log')
plt.show()
