
"""
runs a power analysis for both the basic PRF model and the Ratio model.  
For a range of true values of the selection coefficient g
and array contains a series of values from the chi-squared distribution with 1 degree of freedom
the script estimates the proportion of time that a True Positive occurs. 

User must set theta 

takes a couple hours to run 

"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
import  SFS_functions
import math
import warnings
warnings.filterwarnings("ignore")

pvalues = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
plabels = ["{:.3f}".format(p) for p in pvalues]
x2thresholds = [2.70554, 3.84146, 5.41189, 6.6349, 7.87944, 9.54954, 10.8276]
np.random.seed(1)
dobasicPRF = True
doratioPRF = True
n=40
theta = 100
dofolded = False
foldstring = "folded" if dofolded else "unfolded"
gvals = [5,2.5,2,1.5,1,0.5,0.2,0.1,0.05,0.0,-0.05,-0.1,-0.2,-0.5,-1,-1.5,-2,-2.5,-5]
gvals.reverse()
ntrialsperg = 1000
if dobasicPRF:
    plotfilename = 'basicPRF_power_theta{}_n{}_{}.pdf'.format(theta,n,foldstring)
    # x2thresholds.reverse()
    results = [[0]*len(pvalues) for i in range(len(gvals))] #results[i][j] has the count of significant results for gval[i] and pval[j]
    for gj,g in enumerate(gvals):
        for i in range(ntrialsperg):
            sfs,sfsfolded = SFS_functions.simsfs(theta,g,n,None)
            thetastart = 100.0
            gstart = -1.0
            thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            g0result = minimize_scalar(SFS_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,sfsfolded),method='Brent')
            thetagdelta = 2*(-thetagresult.fun + g0result.fun)
            for ti,t in enumerate(x2thresholds):
                if thetagdelta > t:
                    results[gj][ti] += 1
    for ti,t in enumerate(x2thresholds):
        for gj,g in enumerate(gvals):
            results[gj][ti]/= ntrialsperg
    for ti,t in enumerate(x2thresholds):
        temp = []
        for gj,g in enumerate(gvals):
            temp.append(results[gj][ti])
        plt.plot(gvals,temp,label=plabels[ti])
    plt.xlabel("g (selection)")
    plt.ylabel("Probability of rejecting Null")
    plt.legend(title='p')
    plt.savefig(plotfilename)
    # plt.show()
if doratioPRF:
    plotfilename = 'ratioPRF_power_theta{}_n{}_{}.pdf'.format(theta,n,foldstring)
    # x2thresholds.reverse()
    results = [[0]*len(pvalues) for i in range(len(gvals))] #results[i][j] has the count of significant results for gval[i] and pval[j]
    for gj,g in enumerate(gvals):
        for i in range(ntrialsperg):
            nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,n, None, dofolded, g=g)
            thetastart = 100.0
            gstart = -1.0
            ratiothetagresult =  minimize(SFS_functions.NegL_SFSRATIO_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            ratiothetag0result =  minimize_scalar(SFS_functions.NegL_SFSRATIO_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,ratios,True),method='Brent')        
            thetagdelta = 2*(-ratiothetagresult.fun + ratiothetag0result.fun)
            for ti,t in enumerate(x2thresholds):
                if thetagdelta > t:
                    results[gj][ti] += 1
    for ti,t in enumerate(x2thresholds):
        for gj,g in enumerate(gvals):
            results[gj][ti]/= ntrialsperg
    for ti,t in enumerate(x2thresholds):
        temp = []
        for gj,g in enumerate(gvals):
            temp.append(results[gj][ti])
        plt.plot(gvals,temp,label=plabels[ti])
    plt.xlabel("g (selection)")
    plt.ylabel("Probability of rejecting Null")
    plt.legend(title='p')
    plt.savefig(plotfilename)
    # plt.show()

