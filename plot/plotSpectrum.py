import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

#ergoPlot.dpi = 2000

configFile = '../cfg/QG4.cfg'
compName1 = 'A_1'
compName2 = 'A_2'
compName3 = 'A_3'
compName3 = 'A_4'

cfg = pylibconfig2.Config()
cfg.read_file(configFile)

L = cfg.simulation.LCut + cfg.simulation.spinup
tau = L
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
sigma = cfg.model.sigma
dim = cfg.model.dim
dimObs = dim

N = np.prod(np.array(cfg.grid.nx))
gridPostfix = ""
for d in np.arange(dimObs):
    if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.nSTDLow[d],
                                        cfg.grid.nSTDHigh[d])
    else:
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.sprinkle.minInitState[d],
                                        cfg.sprinkle.maxInitState[d])
gridPostfix = "_%s%s" % (caseName, gridPostfix)
srcPostfixSim = "%s_sigma%04d_L%d_dt%d_nTraj%d" \
                % (gridPostfix, int(sigma * 1000 + 0.1), int(L * 1000 + 0.1),
                   -np.round(np.log10(cfg.simulation.dt)), cfg.sprinkle.nTraj)

xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))


# Define file names
postfix = "%s_tau%03d" % (srcPostfixSim, tau * 1000)
eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                       cfg.general.fileFormat)
eigValBackwardFile = '%s/eigval/eigvalBackward_nev%d%s.%s' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                    cfg.general.fileFormat)

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and backward eigenvectors:
print 'Readig spectrum for tau = %.3f...' % tau
(eigValForward,) = ergoPlot.readSpectrum(eigValForwardFile,
                                         fileFormat=cfg.general.fileFormat)

# Get generator eigenvalues (using the complex logarithm)
eigValGen = np.log(eigValForward) / tau

# Plot eigenvectors of transfer operator
alpha = 0.0
os.system('mkdir %s/spectrum/eigvec 2> /dev/null' % cfg.general.plotDir)
os.system('mkdir %s/spectrum/reconstruction 2> /dev/null' % cfg.general.plotDir)
for ev in np.arange(cfg.spectrum.nEigVecPlot):
    print 'Plotting real part of eigenvector %d...' % (ev + 1,)
    if dimObs == 2:
        ergoPlot.plot2D(X, Y, eigVecForward[ev].real,
                        ev_xlabel, ev_ylabel, alpha)
    elif dimObs == 3:
        ergoPlot.plot3D(X, Y, Z, eigVecForward[ev].real, mask,
                        ev_xlabel, ev_ylabel, ev_zlabel, alpha)
    dstFile = '%s/spectrum/eigvec/eigvecForwardReal_ev%03d%s.%s' \
              % (cfg.general.plotDir, ev + 1, postfix, ergoPlot.figFormat)
    plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
    if cfg.spectrum.plotImag & (eigValForward[ev].imag != 0):
        print 'Plotting imaginary  part of eigenvector %d...' % (ev + 1,)
        if dimObs == 2:
            ergoPlot.plot2D(X, Y, eigVecForward[ev].imag,
                            ev_xlabel, ev_ylabel, alpha)
        elif dimObs == 3:
            ergoPlot.plot3D(X, Y, Z, eigVecForward[ev].imag, mask,
                            ev_xlabel, ev_ylabel, ev_zlabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecForwardImag_ev%03d%s.%s' \
                  % (cfg.general.plotDir, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches,
                    dpi=ergoPlot.dpi)
    
    # Plot eigenvectors of backward operator
    if cfg.spectrum.plotBackward:
        print 'Plotting real part of backward eigenvector %d...' % (ev + 1,)
        if dimObs == 2:
            ergoPlot.plot2D(X, Y, eigVecBackward[ev].real,
                            ev_xlabel, ev_ylabel, alpha)
        elif dimObs == 3:
            ergoPlot.plot3D(X, Y, Z, eigVecBackward[ev].real, mask,
                            ev_xlabel, ev_ylabel, ev_zlabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecBackwardReal_ev%03d%s.%s' \
                  % (cfg.general.plotDir, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches,
                    dpi=ergoPlot.dpi)
        
        if cfg.spectrum.plotImag & (eigValForward[ev].imag != 0):
            print 'Plotting imaginary  part of backward eigenvector %d...' \
                % (ev + 1,)
            if dimObs == 2:
                ergoPlot.plot2D(X, Y, eigVecBackward[ev].imag,
                                ev_xlabel, ev_ylabel, alpha)
            elif dimObs == 3:
                ergoPlot.plot3D(X, Y, Z, eigVecBackward[ev].imag, mask,
                                ev_xlabel, ev_ylabel, ev_zlabel, alpha)
            dstFile = '%s/spectrum/eigvec/eigvecBackwardImag_ev%03d%s.%s' \
                      % (cfg.general.plotDir, ev + 1, postfix,
                         ergoPlot.figFormat)
            plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches,
                        dpi=ergoPlot.dpi)

            
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'

ergoPlot.plotEig(eigValGen, xlabel=realLabel, ylabel=imagLabel)
plt.savefig('%s/spectrum/eigVal/eigVal%s.%s'\
            % (cfg.general.plotDir, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

