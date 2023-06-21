
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
dobasicPRF = False #True
doratioPRF = True
n=40
theta = 25
gvals = [5,2.5,2,1.5,1,0.5,0.2,0.1,0.05,0.0,-0.05,-0.1,-0.2,-0.5,-1,-1.5,-2,-2.5,-5]
gvals.reverse()
ntrialsperg = 1000
if dobasicPRF:
    plotfilename = 'basicPRF_power_theta{}.pdf'.format(theta)
    # x2thresholds.reverse()
    results = [[0]*len(pvalues) for i in range(len(gvals))] #results[i][j] has the count of significant results for gval[i] and pval[j]
    for gj,g in enumerate(gvals):
        for i in range(ntrialsperg):
            sfs,sfsfolded = SFS_functions.simsfs(theta,g,n)
            thetastart = 100.0
            gstart = -1.0
            maxi = n//2 - 1 # not used at the moment
            thetagresult = minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            g0result = minimize_scalar(SFS_functions.negLfw,bracket=(thetastart/10,thetastart*10),args = (maxi,n,sfsfolded),method='Brent')
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
    plotfilename = 'ratioPRF_power_theta{}.pdf'.format(theta)
    # x2thresholds.reverse()
    results = [[0]*len(pvalues) for i in range(len(gvals))] #results[i][j] has the count of significant results for gval[i] and pval[j]
    for gj,g in enumerate(gvals):
        for i in range(ntrialsperg):
            nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,g,n)
            thetastart = 100.0
            gstart = -1.0
            maxi = n//2 - 1 # not used at the moment
            ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            ratiothetag0result =  minimize_scalar(SFS_functions.negL_ratio,bracket=(thetastart/10,thetastart*10),args = (maxi,n,ratios,True),method='Brent')        
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
    plt.show()

