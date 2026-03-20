import numpy as np
import matplotlib.pyplot as plt
import argparse

import mctspy as mct

parser = argparse.ArgumentParser()
parser.add_argument ('-phi',metavar='phi',help='packing fraction',
                     type=float, nargs='+', default=[0.5])
parser.add_argument ('-delta',metavar='delta',help='tagged-particle size',
                     type=float, default=1.0)
args = parser.parse_args()

def output (d, istart, iend, correlator_array):
    print ("block",d,"\r",end='')

qgrid = np.linspace(0.2,39.8,100)


sqlist = []
cslist = []
for packing_fraction in args.phi:
    sq = mct.structurefactors.hssPY(phi=packing_fraction)
    plt.plot(qgrid, sq.Sq(qgrid)[0])
    cs = mct.structurefactors.hssPYtagged(phi=packing_fraction,delta=args.delta)
    sqlist.append(sq)
    cslist.append(cs)
plt.show()


phi_list = []
phi_s_list = []
msd_list = []
ngp_list = []
for packing_fraction,sq,cs in zip(args.phi,sqlist,cslist):
    # base model "simple liquid"
    model = mct.simple_liquid_model (sq, qgrid)
    phi = mct.correlator (model = model, store = True, blocks=50)
    correlators = mct.CorrelatorStack([phi])

    # tagged-particle model
    model_s = mct.tagged_particle_model (model, cs=cs, D0s=1/args.delta)
    phi_s = mct.correlator (model = model_s, base=phi, store = True)
    correlators.append(phi_s)

    # q=0 limit of tagged particle to determine the MSD
    model_s0 = mct.tagged_particle_q0 (model_s)
    msd = mct.mean_squared_displacement (model = model_s0, base=phi_s, store=True)
    correlators.append(msd)

    # non-Gaussian parameter
    model_ngp = mct.tagged_particle_ngp (model_s0)
    ngp = mct.non_gaussian_parameter (model = model_ngp, base=msd, store=True)
    correlators.append(ngp)

    if False:
        f = mct.nonergodicity_parameter (model = model)
        f.solve()
        fs = mct.nonergodicity_parameter (model = model_s)
        fs.solve()
        plt.plot(qgrid, f.f)
        plt.plot(qgrid, fs.f)
        plt.show()
    
        ev = mct.eigenvalue (f)
        ev.solve()
        print ("eigenvalue",ev.eval,ev.eval2)
        print ("lambda",ev.lam)
        plt.plot(qgrid,ev.e*(1-f.f)**2)
        plt.plot(qgrid,ev.ehat)
        plt.show()


    print("packing fraction",packing_fraction)
    correlators.solve_all(callback=output)
    print("")

    # save:
    #phi.save("/tmp/phi.h5")

    # restore:
    #phi2 = mct.correlator.load('/tmp/phi.h5')
    #correlators2 = mct.CorrelatorStack([phi2])
    #model_s2 = mct.tagged_particle_model (phi2.model, cs=sq)
    #phi_s2 = mct.correlator (model = model_s2, base=phi2, store = True)
    #correlators2.append(phi_s2)
    #correlators2.solve_all(callback=output)
    #print("")

    phi_list.append(phi)
    phi_s_list.append(phi_s)
    msd_list.append(msd)
    ngp_list.append(ngp)


qval = 7.4
qi = np.nonzero(np.isclose(qgrid,qval))[0][0]
for packing_fraction,phi,phi_s in zip(args.phi,phi_list,phi_s_list):
    plt.plot(phi.t, phi.phi[:,qi], label='phi(t), phi={}'.format(packing_fraction))
    plt.plot(phi_s.t, phi_s.phi[:,qi], label='phi_s(t), phi={}'.format(packing_fraction))
    plt.plot(phi.t, np.exp(-qval**2*phi.t),color='black',linestyle='dashed')
plt.xscale('log')
plt.legend()
plt.show()

for packing_fraction,phi,phi_s in zip(args.phi,phi_list,phi_s_list):
    for corr in [phi,phi_s]:
        tau_indices = corr.phi.shape[0] - np.sum(corr.phi<=0.1,axis=0)
        tau_indices[tau_indices>=corr.phi.shape[0]] = -1
        tau = corr.t[tau_indices]
        plt.plot(model.q,tau)
plt.yscale('log')
plt.show()

for packing_fraction,msd in zip(args.phi,msd_list):
    plt.plot(msd.t, msd.phi[:,0])
    plt.plot(msd.t, 6*msd.t/args.delta, color='black', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.show()

for packing_fraction,ngp in zip(args.phi,ngp_list):
    plt.plot(ngp.t[ngp.t>1e-5], (ngp.phi[:,0][ngp.t>1e-5]/ngp.phi[:,1][ngp.t>1e-5]**2 - 1), label='NGP phi={}'.format(packing_fraction))
plt.xscale('log')
plt.legend()
plt.show()
