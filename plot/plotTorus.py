import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2

figFormat = 'eps'

def field(x, p):
    sigma, ci, li = p
    f = np.array([ci[0]*x[0]*x[1] + ci[1]*x[1]*x[2] + ci[2]*x[2]*x[3] - li[0]*x[0],
                  ci[3]*x[1]*x[3] + ci[4]*x[0]*x[2] - ci[0]*x[0]**2 \
                  - li[1]*x[1] + ci[6] * sigma,
                  ci[5]*x[0]*x[3] - (ci[1]+ci[4])*x[0]*x[1] - li[2]*x[2],
                  -ci[3]*x[1]**2 - (ci[2]+ci[5])*x[0]*x[2] - li[3]*x[3]])
    return f

def Jacobian(x, p):
    sigma, ci, li = p
    J = np.array([[ci[0]*x[1] - li[0], ci[0]*x[0] + ci[1]*x[2],
                   ci[1]*x[1] + ci[2]*x[3], ci[2]*x[2]],
                  [ci[4]*x[2] - 2*ci[0]*x[0], ci[3]*x[3] - li[1],
                   ci[4]*x[0], ci[3]*x[1]],
                  [ci[5]*x[3] - (ci[1]+ci[4])*x[1], -(ci[1]+ci[4])*x[0],
                   -li[2], ci[5]*x[0]],
                  [-(ci[2]+ci[5])*x[2], -2*ci[3]*x[1], -(ci[2]+ci[5])*x[0], -li[3]]])
    return J

def propagateRK4(x0, field, p, dt, nt):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    xt = np.empty((nt, x0.shape[0]))
    xt[0] = x0.copy()
    for t in np.arange(1, nt):
        # Step solution forward
        k1 = field(xt[t-1], p) * dt
        tmp = k1 * 0.5 + xt[t-1]

        k2 = field(tmp, p) * dt
        tmp = k2 * 0.5 + xt[t-1]

        k3 = field(tmp, p) * dt
        tmp = k3 + xt[t-1]

        k4 = field(tmp, p) * dt
        tmp = (k1 + 2*k2 + 2*k3 + k4) / 6
        
        xt[t] = xt[t-1] + tmp

    return xt

fs_latex = 'xx-large'
fs_xticklabels = 'large'
fs_yticklabels = fs_xticklabels

configFile = '../cfg/QG4.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

dim = cfg.model.dim
L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

# List of continuations to plot
#initContRng = [[1.903806, 0.893549, -3.889728, -0.097373, 0.8, 75.],
#               [1.903806, 0.893549, -3.889728, -0.097373, 0.8, 75.]]
#sigmaStepRng = [0.001, -0.001]
# initContRng = [[-0.389461, 1.940367, 2.03401 ,-2.047458, 0.64, 73.]]
# sigmaStepRng = [-0.0001]
initContRng = [[1.903806, 0.893549, -3.889728, -0.097373, 0.8, 75.],
               [1.903806, 0.893549, -3.889728, -0.097373, 0.8, 75.]]
sigmaStepRng = [-0.001, 0.001]
dtRng = [1.e-3, 1.e-3]
nCont = len(initContRng)

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

# Prepare plot
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(111)
poL = []
FloquetExpL = []
sigmaL = []
TRngL = []
for k in np.arange(nCont):
    initCont = initContRng[k]
    sigmaStep = sigmaStepRng[k]
    
    sigmaAbs = sqrt(sigmaStep*sigmaStep)
    sign = sigmaStep / sigmaAbs
    exp = np.log10(sigmaAbs)
    mantis = sign * np.exp(np.log(sigmaAbs) / exp)
    dstPostfix = "%s_sigma%04d_sigmaStep%de%d_dt%d_numShoot%d" \
                 % (srcPostfix, int(initCont[dim] * 1000 + 0.1), int(mantis*1.01),
                    (int(exp*1.01)), -np.round(np.log10(dtRng[k])),
                    cfg.continuation.numShoot)
    poFileName = '%s/poCont%s.txt' % (contDir, dstPostfix)
    FloquetExpFileName = '%s/poExpCont%s.txt' % (contDir, dstPostfix)

    # Read fixed point and sigma
    state = np.loadtxt(poFileName).reshape(-1, dim+2)
    # Read FloquetExpenvalues
    FloquetExp = np.loadtxt(FloquetExpFileName)
    FloquetExp = (FloquetExp[:, 0] + 1j * FloquetExp[:, 1]).reshape(-1, dim)
    # Remove last
    state = state[:-1]
    FloquetExp = FloquetExp[:-1]
    
    po = state[:, :dim]
    TRng = state[:, dim+1]
    sigmaRng = state[:, dim]


    # Reorder Floquet exp
    for t in np.arange(1, sigmaRng.shape[0]):
        tmp = FloquetExp[t].tolist()
        for exp in np.arange(dim):
            idx = np.argmin(np.abs(tmp - FloquetExp[t-1, exp]))
            FloquetExp[t, exp] = tmp[idx]
            tmp.pop(idx)

    poL.append(po)
    FloquetExpL.append(FloquetExp)
    sigmaL.append(sigmaRng)
    TRngL.append(TRng)
    
    isStable = np.max(FloquetExp.real, 1) < 1.e-6
    A13 = po[:, 0] + po[:, 2]

    # Plot period
    ax.plot(sigmaRng[isStable], TRng[isStable], '-k', linewidth=2)
    ax.plot(sigmaRng[~isStable], TRng[~isStable], '--k', linewidth=2)

ax.set_ylabel(r'$T$', fontsize=fs_latex)
ax.set_xlim(0., 2.)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
ax.set_xlabel(r'$\sigma$', fontsize=fs_latex)

# k = 0
# amin = np.argmin(sigmaL[k])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(sigmaL[k][:amin], FloquetExpL[k][:amin, :].real)
# ax.set_ylim(0., 0.2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(sigmaL[k][amin:], FloquetExpL[k][amin:].real)

# for k in np.arange(nCont):
#     sigmaRng = sigmaL[k]
#     TRng = TRngL[k]
#     t = 0
#     dt = 0.01
    
#     T = TRng[t] * 1
#     nt = int(np.ceil(T / dt))
#     sigma = sigmaRng[t]
#     p = [sigma, cfg.model.ci, cfg.model.li]

#     xt = propagateRK4(poL[k][t], field, p, dt, nt)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(xt[:, 0] + xt[:, 2], xt[:, 1], xt[:, 3])
#     ax.set_xlabel(r'$A_1 + A_3$')
#     ax.set_ylabel(r'$A_2$')
#     ax.set_zlabel(r'$A_4$')

#     tendency = np.sqrt(np.sum((xt[1:] - xt[:-1])**2, 1)) / dt
#     plt.figure()
#     plt.plot(tendency)

# k = 1
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(sigmaL[k], FloquetExpL[k].real)

T = 1000
nt = int(np.ceil(T / dt))
sigma = 1.
p = [sigma, cfg.model.ci, cfg.model.li]

xt = propagateRK4(poL[0][0], field, p, dt, nt)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt[:, 0] + xt[:, 2], xt[:, 1], xt[:, 3])
ax.set_xlabel(r'$A_1 + A_3$')
ax.set_ylabel(r'$A_2$')
ax.set_zlabel(r'$A_4$')

plt.figure()
plt.plot(xt[:, 0] + xt[:, 2])

