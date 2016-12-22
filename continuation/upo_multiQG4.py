"""
Continuation of the Hopf fixed points and periodic orbit
"""
import numpy as np

dim = 4

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
    xt = x0.copy()
    for t in np.arange(nt):
        # Step solution forward
        xt += dt * field(xt, p)

    return xt

def propagateLinearEuler(x0, M0, field, Jacobian, p, dt, nt):
    '''Propagate solution and fundamental matrix \
    according to the vector field field and Jacobian matrix Jacobian \
    with Euler scheme from x0 and M0 for nt time steps of size dt.'''
    xt = x0.copy()
    Mt = M0.copy()
    for t in np.arange(nt):
        # Step fundamental forward
        Mt += dt * np.dot(Jacobian(xt, p), Mt)
        # Step solution forward
        xt += dt * field(xt, p)

    return (xt, Mt)

def getLinearSystem(x0, M0, xt, Mt, field, p):
    '''Get matrix N and column vector B of linear system N * DY = B.'''
    dim = x0.shape[0]
    
    # Define matrix to inverse
    N = np.zeros((dim+1, dim+1))
    N[:dim, :dim] = Mt - M0
    N[-1, :-1] = field(x0, p)
    N[:-1, -1] = field(xt, p)
    
    # Define target
    B = np.zeros((dim+1, 1))
    B[:-1, 0] = x0 - xt

    return (N, B)
    
def getLinearSystemMulti(x0, M0, xt, Mt, field, p):
    '''Get matrix N and column vector B of linear system N * DY = B.'''
    dim = x0[0].shape[0]
    nShoot = len(x0)
    
    # Define matrix to inverse and target
    N = np.zeros((dim*nShoot+1, dim*nShoot+1))
    B = np.zeros((dim*nShoot+1, 1))
    for s in np.arange(nShoot):
        N[-1, s*dim:(s+1)*dim] = field(x0[s], p)
        N[s*dim:(s+1)*dim, -1] = field(xt[s], p)
        N[s*dim:(s+1)*dim, s*dim:(s+1)*dim] = Mt[s]
    for s in np.arange(nShoot-1):
        N[s*dim:(s+1)*dim, (s+1)*dim:(s+2)*dim] = -M0[s+1]
        B[s*dim:(s+1)*dim, 0] = x0[s+1] - xt[s]
    N[(nShoot-1)*dim:nShoot*dim, :dim] += -M0[0]
    B[(nShoot-1)*dim:nShoot*dim, 0] = x0[0] - xt[nShoot-1]

    return (N, B)
    
def NewtonStep(x0, M0, T0, field, Jacobian, p, dt0):
    ''' Perform one Newton step.'''
    dim = x0.shape[0]
    # Adapt time step to period
    nt = np.int(np.ceil(T0 / dt0))
    dt = T0 / nt
    
    # Get solution and fundamental matrix
    (xt, Mt) = propagateLinearEuler(x0, M0, field, Jacobian, p,
                                    dt, nt)

    # Get matrix and target vector of linear system N * DY = B
    (N, B) = getLinearSystem(x0, M0, xt, Mt, field, p)

    # Get solution to linear system
    DY = np.dot(np.linalg.inv(N), B)
        
    # Calculate distance between points and step size
    errDist = np.sqrt(np.sum((xt - x0)**2))        
    errDelta = np.sqrt(np.sum(DY**2))
    
    return (DY, errDist, errDelta, xt, Mt)

def NewtonStepMulti(x0, M0, T0, field, Jacobian, p, dt0):
    ''' Perform one Newton multiple step from a list of \
    initial states and fundamental matrices.'''
    dim = x0[0].shape[0]
    nShoot = len(x0)
    
    # Adapt time step to period
    ntt = np.int(np.ceil(T0 / dt0))
    dt = T0 / ntt
    nt = np.ones((nShoot,), dtype=int) * (ntt / nShoot)
    nt[-1] += np.mod(ntt, nShoot)

    # Get propagated state and fundamental matrix for each shoot
    xt = []
    Mt = []
    for s in np.arange(nShoot):
        # Get solution and fundamental matrix
        (xts, Mts) = propagateLinearEuler(x0[s], M0[s], field, Jacobian, p,
                                          dt, nt[s])
        xt.append(xts)
        Mt.append(Mts)

    # Get matrix and target vector of linear system N * DY = B
    (N, B) = getLinearSystemMulti(x0, M0, xt, Mt, field, p)

    # Get solution to linear system
    DY = np.dot(np.linalg.inv(N), B)
        
    # Calculate distance between points and step size
    errDist = np.sum((xt[-1] - x0[0])**2)
    for s in np.arange(nShoot-1):
        errDist += np.sum((xt[s] - x0[s+1])**2)
    errDist = np.sqrt(errDist)
    errDelta = np.sqrt(np.sum(DY**2))
    
    return (DY, errDist, errDelta, xt, Mt)

def NewtonUpdate(x0, T, step, damping):
    '''Update state after Newton step with damping.'''
    x0 += damping * step[:-1, 0]
    T += damping * step[-1, 0]

    return T

def NewtonUpdateMulti(x0, T, step, damping):
    '''Update state after Newton step with damping.'''
    nShoot = len(x0)
    for s in np.arange(nShoot):
        x0[s] += damping * step[s*dim:(s+1)*dim, 0]
    T += damping * step[-1, 0]

    return T

def getQuality(x0, T0, field, p, dt0):
    '''Evaluate quality function H.T * H at state x0.'''
    dim = x0.shape[0]
    # Adapt time step to period
    nt = np.int(np.ceil(T0 / dt0))
    dt = T0 / nt

    # Get solution at t
    xt = propagateEuler(x0, field, p, dt, nt)

    # Get quality
    H = np.zeros((dim+1,))
    H[:-1] = x0 - xt
    H2 = np.sum(H**2) / 2

    return H2
    
def getQualityMulti(x0, T0, field, p, dt0):
    '''Evaluate quality function H.T * H at state x0.'''
    nShoot = len(x0)
    dim = x0[0].shape[0]
    # Adapt time step to period
    ntt = np.int(np.ceil(T0 / dt0))
    dt = T0 / ntt
    nt = np.ones((nShoot,), dtype=int) * (ntt / nShoot)
    nt[-1] += np.mod(ntt, nShoot)

    # Get solution at t
    xt = []
    for s in np.arange(nShoot):
        xt.append(propagateEuler(x0[s], field, p, dt, nt[s]))

    # Get quality
    H = np.zeros((nShoot*dim+1,))
    for s in np.arange(nShoot-1):    
        H[s*dim:(s+1)*dim] = x0[s+1] - xt[s]
    H[(nShoot-1)*dim:nShoot*dim] = x0[0] - xt[nShoot-1]
    H2 = np.sum(H**2) / 2

    return H2
    
def getDampingFloquet(Mt, T):
    '''Calculate damping based on largest Floquet multiplier.'''
    (multipliers, eigVec) = np.linalg.eig(Mt)
    isort = np.argsort(np.abs(multipliers))

    return multipliers[isort][-2]

def getDampingLineSearch(x0, M0, T0, step, field, p, dt0, alpha):
    '''Get damping for Newton method by backtracking line search.'''
    damping1 = 1.
    dampMin = 0.1

    # Get initial quality
    H20 = getQuality(x0, T0, field, p, dt0)

    # Get update with standard damping
    xn = x0.copy()
    Tn = NewtonUpdate(xn, T0, step, damping1)
    H21 = getQuality(xn, T0, field, p, dt0)
    
    # Get matrix and target vector of linear system N * DY = B
    (N, B) = getLinearSystem(x0, M0, xt, Mt, field, p)
    # Get g'(0) with g(damping) = H2(x0 + damping * DY)
    gp0 = np.dot(np.dot(B.T, N), step)

    # If the decrease of the quality function is not good enough,
    # then backtrack.
    damping = damping1
    print 'H21 = ', H21
    print 'H20 = ', H20
    if (H21 > H20 + alpha * gp0):
        # Get damping from quadratic approximation of g(damping)
        damping2 = -gp0 / (2 * (H21 - H20 - gp0))
        
        # Guard from to strong damping
        damping = np.max([damping2, damping1/10])

        # Get update with new damping
        Tn = NewtonUpdate(xn, T0, step, damping)
        H22 = getQuality(xn, T0, field, p, dt0)

        # Subsequent steps are cubic
        while (H22 > H20 + alpha * gp0):
            # Get coefficients a and b
            ab = 1./(damping1 - damping2) * \
                 np.dot(np.array([[1./damping1**2,
                                   -1./damping2**2],
                                  [-damping2/damping1**2,
                                   damping1/damping2**2]]),
                        np.array([H21 - gp0*damping1 - H20,
                                  H22, - gp0*damping2 - H20]))
            # Update dampings
            damping1 = damping2
            damping2 = (-ab[1] + np.sqrt(ab[1]**2 - 3*ab[0]*gp0)) / (3*ab[0])

            # Guard damping damping1/10 < damping < damping1/2
            damping = np.max([np.min([damping2, damping1/2]), damping1/10])

            # Get update with new damping
            H21 = H22
            Tn = NewtonUpdate(xn, T0, step, damping)
            H22 = getQuality(xn, T0, field, p, dt)

    return damping
                     

def getDampingLineSearchMulti(x0, M0, T0, step, field, p, dt0, alpha):
    '''Get damping for Newton method by backtracking line search.'''
    damping1 = 1.
    dampMin = 0.1

    # Get initial quality
    H20 = getQualityMulti(x0, T0, field, p, dt0)

    # Get update with standard damping
    xn = copy(x0)
    Tn = NewtonUpdateMulti(xn, T0, step, damping1)
    H21 = getQualityMulti(xn, T0, field, p, dt0)
    
    # Get matrix and target vector of linear system N * DY = B
    (N, B) = getLinearSystemMulti(x0, M0, xt, Mt, field, p)
    # Get g'(0) with g(damping) = H2(x0 + damping * DY)
    gp0 = np.dot(np.dot(B.T, N), step)[0, 0]

    # If the decrease of the quality function is not good enough,
    # then backtrack.
    damping = damping1
    print 'H21 = ', H21
    print 'H20 = ', H20
    if (H21 > H20 + alpha * gp0):
        # Get damping from quadratic approximation of g(damping)
        damping2 = -gp0 / (2 * (H21 - H20 - gp0))
        
        # Guard from to strong damping
        damping = np.max([damping2, damping1/10])

        # Get update with new damping
        Tn = NewtonUpdateMulti(xn, T0, step, damping)
        H22 = getQualityMulti(xn, T0, field, p, dt0)

        # Subsequent steps are cubic
        while (H22 > H20 + alpha * gp0):
            # Get coefficients a and b
            ab = 1./(damping1 - damping2) * \
                 np.dot(np.array([[1./damping1**2,
                                   -1./damping2**2],
                                  [-damping2/damping1**2,
                                   damping1/damping2**2]]),
                        np.array([H21 - gp0*damping1 - H20,
                                  H22 - gp0*damping2 - H20]))
            # Update dampings
            damping1 = damping2
            damping2 = (-ab[1] + np.sqrt(ab[1]**2 - 3*ab[0]*gp0)) / (3*ab[0])

            # Guard damping damping1/10 < damping < damping1/2
            damping = np.max([np.min([damping2, damping1/2]), damping1/10])

            # Get update with new damping
            H21 = H22
            Tn = NewtonUpdateMulti(xn, T0, step, damping)
            H22 = getQualityMulti(xn, T0, field, p, dt0)

    return damping
    

# Phase space dimension
dim = 4
# Parameters
sigma = 0.8
ci = np.array([0.020736, 0.018337, 0.015617, 0.031977, 0.036673, 0.046850, 0.314802])
li = np.array([0.0128616, 0.0211107, 0.0318615, 0.0427787])
p = [sigma, ci, li]

print '------------------------'
print '--- Model parameters ---'
print 'dim = ', dim
print 'p = ', p

# Plot
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111)


# Time step
dt0 = 1.e-3
# Distance before to stop and maximum number of iterations
eps = 1.e-6
maxIter = 1000
alpha = 1.e-4
nShoot = 1

print '-------------------------'
print '--- Scheme parameters ---'
print 'dt0 = ', dt0
print 'eps = ', eps
print 'maxIter = ', maxIter

# Initial state
pert = 0.1
T0 = 75.
x0 = [np.array([1.903806,  0.893549, -3.889728, -0.097373])]
M0 = [np.eye(dim)]

# Plot
msize = 60

# niter = 20
xn = x0
Tn = T0
errDist, errDelta = 1.e27, 1e27

#for k in np.arange(niter):
nIter = 0
damp = 1.

# Get first error
# Adapt time step to period
nt = np.int(np.ceil(T0 / dt0))
dt = T0 / nt

# Test distance between periodic points, size of Newton step
# and maximum number of iterations
marker = ['+', 'x', 'o', '^']
print '---------------'
while ((errDist > eps) or (errDelta > eps)) and (nIter < maxIter):
    # Integrate solution Euler forward
    print '---Iteration ', nIter, '---'
    print 'x0 = ', x0[0]
    print 'T0 = ', T0
    #    for s in np.arange(nShoot):
    #        plt.scatter(x0[s][0], x0[s][1], s=msize, c='k', marker=marker[s])
    plt.scatter(x0[0][0], x0[0][1], s=msize, c='k', marker='+')

    # Perform Newton step
    (step, errDist, errDelta, xt, Mt) \
        = NewtonStepMulti(x0, M0, T0, field, Jacobian, p, dt0)

    print 'x(t) = ', xt[-1]
    #    for s in np.arange(nShoot):
    #        plt.scatter(xt[s][0], xt[s][1], s=msize, c='r', marker=marker[s])
    plt.scatter(xt[-1][0], xt[-1][1], s=msize, c='r', marker='x')

    # Get damping
    # Get quality function
    # damp = getDampingFloquet(Mt, T0)
    # damp = getDampingLineSearchMulti(x0, M0, T0, step, field, p, dt0, alpha)
    # print 'damping = ', damp

    # Update Newton
    T0 = NewtonUpdateMulti(x0, T0, step, damp)
    
    print '------------'
    print '---Errors---'
    print 'd(x(t), x0) = ', errDist
    print '|dy| = ', errDelta

    nIter += 1

# Get Floquet multipliers
# Adapt time step to period
nt = np.int(np.ceil(T0 / dt0))
dt = T0 / nt
(xT, MT) = propagateLinearEuler(x0[0], M0[0], field, Jacobian, p,
                                dt, nt)
(multipliers, eigVec) = np.linalg.eig(MT)
exponents = np.log(multipliers) / T0
print '-------------'
print '---Floquet---'
print 'Floquet multipliers = ', multipliers
print 'Floquet exponents = ', exponents

print '-----------------'
print '---Convergence---'
if nIter == maxIter:
    print 'Did not converge!'
else:
    print 'Converged.'
