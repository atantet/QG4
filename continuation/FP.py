import numpy as np
import scipy.optimize

dim = 4

def field(x, sigma, ci, li):
    f = np.array([ci[0]*x[0]*x[1] + ci[1]*x[1]*x[2] + ci[2]*x[2]*x[3] - li[0]*x[0],
                  ci[3]*x[1]*x[3] + ci[4]*x[0]*x[2] - ci[0]*x[0]**2 \
                  - li[1]*x[1] + ci[6] * sigma,
                  ci[5]*x[0]*x[3] - (ci[1]+ci[4])*x[0]*x[1] - li[2]*x[2],
                  -ci[3]*x[1]**2 - (ci[2]+ci[5])*x[0]*x[2] - li[3]*x[3]])
    return f

def Jacobian(x, sigma, ci, li):
    J = np.array([[ci[0]*x[1] - li[0], ci[0]*x[0] + ci[1]*x[2],
                   ci[1]*x[1] + ci[2]*x[3], ci[2]*x[2]],
                  [ci[4]*x[2] - 2*ci[0]*x[0], ci[3]*x[3] - li[1],
                   ci[4]*x[0], ci[3]*x[1]],
                  [ci[5]*x[3] - (ci[1]+ci[4])*x[1], -(ci[1]+ci[4])*x[0],
                   -li[2], ci[5]*x[0]],
                  [-(ci[2]+ci[5])*x[2], -2*ci[3]*x[1], -(ci[2]+ci[5])*x[0], -li[3]]])
    return J


sigmaRng = np.arange(0, 1, 0.001)
ci = np.array([0.020736, 0.018337, 0.015617, 0.031977, 0.036673, 0.046850, 0.314802])
li = np.array([0.0128616, 0.0211107, 0.0318615, 0.0427787])
#a = 0.01
a = 0.
#a = -0.01
x0 = np.array([-a, 0., a, 0.])
xRng = np.empty((sigmaRng.shape[0], dim))
eigValRng = np.empty((sigmaRng.shape[0], dim), dtype=complex)

for s in np.arange(sigmaRng.shape[0]):
    sigma = sigmaRng[s]
    (x, info, ier, mesg) = scipy.optimize.fsolve(field, x0, args=(sigma, ci, li),
                                                 fprime=Jacobian, full_output=True)
    if (ier != 1):
        print mesg
        print 'Not converged at ', s, ' for ', sigma
        xRng[s] = np.nan
        eigValRng[s] = np.nan
    else:
        xRng[s] = x
        J = Jacobian(x, sigma, ci, li)
        (eigVal, eigVec) = numpy.linalg.eig(J)
        isort = np.argsort(-eigVal.real)
        eigVal = eigVal[isort]
        eigValRng[s] = eigVal

fig = plt.figure()
ax = fig.add_subplot(111)
obs = xRng[:, 0] + xRng[:, 2]
ax.plot(sigmaRng[eigValRng[:, 0].real <= 0], obs[eigValRng[:, 0].real <= 0],
        'ok', markersize=3)
ax.plot(sigmaRng[eigValRng[:, 0].real > 0], obs[eigValRng[:, 0].real > 0],
        '+r', markersize=3)
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$A_1 + A_3$')
fig.savefig('bifurcationDiagram.eps', dpi=300, bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(sigmaRng, np.zeros((sigmaRng.shape[0],)), '--k')
for d in np.arange(dim):
    ax1.plot(sigmaRng, eigValRng[:, d].real, linewidth=2)
    ax2.plot(sigmaRng, eigValRng[:, d].imag, linewidth=2)
ax1.set_ylabel(r'$\Re(\lambda)$')
ax2.set_ylabel(r'$\Im(\lambda)$')
ax2.set_xlabel(r'$\sigma$')
#ax1.set_ylim(-0.2, 0.2)
#ax2.set_ylim(-1, 1)
fig.savefig('JacobianEigenvalues.eps', dpi=300, bbox_inches='tight')
