
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

np.random.seed(1)
dobasicPRF = True
doratioPRF = True
n=40
theta = 100
# gvals = [5,2.5,2,1.5,1,0.5,0.2,0.1,0.05,0.0,-0.05,-0.1,-0.2,-0.5,-1,-1.5,-2,-2.5,-5]
gvals = [-10,-5,-2,-1,0,1,2,5,10]
# gvals.reverse()
ntrialsperg = 200
if dobasicPRF:
    plotfilename = 'basicPRF_estimator_bias_var_theta{}.pdf'.format(theta)
    results = []
    for gi,g in enumerate(gvals):
        results.append([])
        for i in range(ntrialsperg):
            sfs,sfsfolded = SFS_functions.simsfs(theta,g,n)
            thetastart = 100.0
            gstart = -1.0
            maxi = n//2 - 1 # not used at the moment
            thetagresult = minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
            results[gi].append(thetagresult.x[1])            

    
    fig, ax = plt.subplots()
    ax.boxplot(results,showmeans=True,sym='',positions=gvals)
    plt.xlabel("g (selection)")
    plt.ylabel("Estimates")
    plt.plot(gvals,gvals)
    plt.savefig(plotfilename)
    # plt.show()
    plt.clf
if doratioPRF:
    plotfilename = 'ratioPRF_estimator_bias_var_theta{}.pdf'.format(theta)
    results = []
    for gi,g in enumerate(gvals):
        results.append([])
        for i in range(ntrialsperg):
            nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,g,n)
            thetastart = 100.0
            gstart = -1.0
            maxi = n//2 - 1 # not used at the moment
            ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
            # ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios,False),method="Nelder-Mead",bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
            results[gi].append(ratiothetagresult.x[1])            
   
    fig, ax = plt.subplots()
    ax.boxplot(results,showmeans=True,sym='',positions=gvals)
    plt.xlabel("g (selection)")
    plt.ylabel("Estimates")
    plt.plot(gvals,gvals)
    plt.savefig(plotfilename)
    # plt.show()
    plt.clf
