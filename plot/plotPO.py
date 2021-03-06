import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2

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

def propagateEuler(x0, field, p, dt, nt):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    xt = np.empty((nt, x0.shape[0]))
    xt[0] = x0.copy()
    for t in np.arange(1, nt):
        # Step solution forward
        xt[t] = xt[t-1] + dt * field(xt[t-1], p)

    return xt

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
initContRng = [[0.043845, 2.654927, 2.46688 , -2.329673, 0.79, 86.14],
               [0.043845, 2.654927, 2.46688 , -2.329673, 0.79, 86.14],
               [1.903806, 0.893549, -3.889728, -0.097373, 0.8, 75.]]
sigmaStepRng = [0.00001, -0.00001, -0.001]
dtRng = [1.e-2, 1.e-2, 1.e-3]
nCont = len(initContRng)

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

# Prepare plot
fig = plt.figure(figsize=(8, 10))
ax = []
#nPan = 100*(1+2*nCont) + 10 + 1
nPan = 100*(1+2*1) + 10 + 1
ax.append(fig.add_subplot(nPan))
#for k in np.arange(nCont):
for k in np.arange(1):
    nPan += 1
    ax.append(fig.add_subplot(nPan))
    nPan += 1
    ax.append(fig.add_subplot(nPan))

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
    ax[0].plot(sigmaRng[isStable], TRng[isStable], '-k', linewidth=2)
    ax[0].plot(sigmaRng[~isStable], TRng[~isStable], '--k', linewidth=2)

    # Plot real parts
    k = 0
    ax[1+2*k].plot(sigmaRng, np.zeros((sigmaRng.shape[0],)), '--k')
    ax[1+2*k].plot(sigmaRng, FloquetExp.real, linewidth=2)
    ax[1+2*k].set_ylabel(r'$\Re(\lambda_i)$', fontsize=fs_latex)
    plt.setp(ax[1+2*k].get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax[1+2*k].get_yticklabels(), fontsize=fs_yticklabels)
    ax[1+2*k].set_xlim(cfg.continuation.sigmaMin, cfg.continuation.sigmaMax)

    # Plot imaginary parts
    ax[1+2*k+1].plot(sigmaRng, FloquetExp.imag, linewidth=2)
    ax[1+2*k+1].set_ylabel(r'$\Im(\lambda_i)$', fontsize=fs_latex)
    plt.setp(ax[1+2*k+1].get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax[1+2*k+1].get_yticklabels(), fontsize=fs_yticklabels)
    ax[1+2*k+1].set_xlim(cfg.continuation.sigmaMin, cfg.continuation.sigmaMax)
ax[0].set_ylabel(r'$T$', fontsize=fs_latex)
ax[0].set_xlim(cfg.continuation.sigmaMin, cfg.continuation.sigmaMax)
plt.setp(ax[0].get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax[0].get_yticklabels(), fontsize=fs_yticklabels)
ax[-1].set_xlabel(r'$\sigma$', fontsize=fs_latex)

plt.savefig('%s/continuation/poCont%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')


# Fixed point
initContRngFP = [[0., 0., 0., 0., 0.],
                 [-0.1, 1.4, 0.43, -1.4, 0.3],
                 [-0.1, 1.4, 0.43, -1.4, 0.3]]
sigmaStepRngFP = [0.001, 0.001, -0.001]

fig = plt.figure()
ax = fig.add_subplot(111)
dist = []
for k in np.arange(len(initContRngFP)):
    initContFP = initContRngFP[k]
    sigmaStepFP = sigmaStepRngFP[k]
    sigmaAbs = sqrt(sigmaStepFP*sigmaStepFP)
    sign = sigmaStepFP / sigmaAbs
    exp = np.log10(sigmaAbs)
    mantis = sign * np.exp(np.log(sigmaAbs) / exp)
    dstPostfixFP = "%s_sigma%04d_sigmaStep%de%d" \
                   % (srcPostfix, int(initContFP[dim] * 1000 + 0.1),
                      int(mantis*1.01), (int(exp*1.01)))
    fpFileName = '%s/fpCont%s.txt' % (contDir, dstPostfixFP)
    eigFileName = '%s/fpEigCont%s.txt' % (contDir, dstPostfixFP)

    # Read fixed point and sigma
    state = np.loadtxt(fpFileName).reshape(-1, dim+1)
    fp = state[:, :dim]
    sigmaRngFP = state[:, dim]
    eig = np.loadtxt(eigFileName)
    eig = (eig[:, 0] + 1j * eig[:, 1]).reshape(-1, dim)
    # Bound
    isig = sigmaRngFP < 1.
    sigmaRngFP = sigmaRngFP[isig]
    fp = fp[isig]
    eig = eig[isig]

    isStable = np.max(eig.real, 1) < 0

    A13 = fp[:, 0] + fp[:, 2]
    plt.plot(A13[isStable], fp[isStable, 1], '-k', linewidth=2)
    plt.plot(A13[~isStable], fp[~isStable, 1], '--k', linewidth=2)

sampOrbitRng = [2000, 2000, 250]
sampInit = [0, 0, 400]
for k in np.arange(nCont):
    sampOrbit = sampOrbitRng[k]
    po = poL[k]
    FloquetExp = FloquetExpL[k]
    sigmaRng = sigmaL[k]
    TRng = TRngL[k]
    isStable = np.max(FloquetExp.real, 1) < 1.e-6
    
    for t in np.arange(sampInit[k], sigmaRng.shape[0], sampOrbit):
        sigma = sigmaRng[t]
        T = TRng[t]
        print 'Propagating orbit of period ', T, ' at sigma = ', sigma, \
            ' from x(0) = ', po[t]
        print 'Floquet = ', FloquetExp[t]

        nt = int(np.ceil(T / dtRng[k]))
        # propagate
        p = [sigma, cfg.model.ci, cfg.model.li]
        xt = propagateRK4(po[t], field, p, dtRng[k]*10, nt/10)
        if isStable[t]:
            ls = '-'
        else:
            ls = '--'
        #plt.plot(xt[:, 0] + xt[:, 2], xt[:, 1], xt[:, 3],
        #         linestyle=ls, linewidth=2)
        plt.plot(xt[:, 0] + xt[:, 2], xt[:, 1], linestyle=ls, linewidth=2)
        

        # Last one
    t = -1
    sigma = sigmaRng[t]
    T = TRng[t]
    print 'Propagating orbit of period ', T, ' at sigma = ', sigma, \
        ' from x(0) = ', po[t]
    print 'Floquet = ', FloquetExp[t]
    nt = int(np.ceil(T / dtRng[k]))
    # propagate
    p = [sigma, cfg.model.ci, cfg.model.li]
    xt = propagateRK4(po[t], field, p, dtRng[k]*10, nt/10)
    if isStable[t]:
        ls = '-'
    else:
        ls = '--'
    #plt.plot(xt[:, 0] + xt[:, 2], xt[:, 1], xt[:, 3], linestyle=ls, linewidth=2)
    plt.plot(xt[:, 0] + xt[:, 2], xt[:, 1], linestyle=ls, linewidth=2)

# # Homoclinic orbit
# print 'Propagating homoclinic orbit.'
# Th = 100.
# nt = int(np.ceil(Th*10 / cfg.simulation.dt))
# # propagate
# p = [0.79, cfg.model.ci, cfg.model.li]
# #p = [0.789, cfg.model.ci, cfg.model.li]
# x0 = np.array([0.075126, 2.68851, 2.455366, -2.314435])
# #x0 = np.array([0.075126, 2.68851, 2.455366, -2.314435])
# xt = propagateRK4(x0, field, p, cfg.simulation.dt, nt)
# spinup = int(Th*6 / cfg.simulation.dt)
# xt = xt[spinup:]
# plt.plot(xt[:, 0] + xt[:, 2], xt[:, 1], '--k')

ax.set_xlabel(r'$A_1 + A_3$', fontsize=fs_latex)
ax.set_ylabel(r'$A_2$', fontsize=fs_latex)
# ax.set_xlim(-2.8, 2.8)
# ax.set_ylim(1.2, 2.8)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
plt.savefig('%s/continuation/poContOrbits%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')
