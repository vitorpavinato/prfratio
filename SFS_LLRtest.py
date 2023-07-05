"""
    check the log likelihood ratio (LLR) statistic against chi^2 distributions
    if checkPRFnull is True:
        SFS simulations and a figure are generated for the conventional LLR test of a PRF Fisher Wright model
        the results will be saved in a file  chisq_check_PRF_theta{}_g{}.pdf 
        The comparison is mostly only relevant when the null model (i.e. g=0) is true 
    if checkRATIOnull is True:
        simulations and a figure are generated for the LLR test or a ratio of two SFSs 
        two SFS simulations are done (numerator and denominator), and so there are two thetas (thetaS and thetaN)
        the results will be saved in a file  chisq_check_RATIO_thetaN{}_thetaS{}_g{}.pdf 
        As with PRF case, the comparison is mostly only relevant when the null model (i.e. g=0) is true        
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
import  SFS_functions

checkPRFnull = False #True
checkRationull = True

np.random.seed(1)
ntrials = 100
n=40
g = 0.0 
theta = 25#500


if checkPRFnull: # regular PRF LLR stuff,  no use of ratio 
    figfn = "chisq_check_PRF_theta{}_g{}.pdf".format(theta,g)
    thetagcumobsv = []
    #check chi^2 approximation of LLR for basic FW PRF 
    for i in range(ntrials):
        sfs,sfsfolded = SFS_functions.simsfs(theta,g,n)
        thetastart = 100.0
        gstart = -1.0
        maxi = n//2 - 1 # not used
        thetagresult = minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
        # ratioresult =  minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
        # check straight-up use of watterson's estimator
        # wtheta,wL = SFS_functions.watterson_L(n,sfs,folded=False)
        # wthetafolded,wLfolded = SFS_functions.watterson_L(n,sfsfolded,folded=True)
        g0result = minimize_scalar(SFS_functions.negLfw,bracket=(thetastart/10,thetastart*10),args = (maxi,n,sfsfolded),method='Brent')
        thetagdelta = -thetagresult.fun + g0result.fun
        thetagcumobsv.append(2*thetagdelta)
    thetagcumobsv.sort()
    thetagcumobsv = np.array(thetagcumobsv)
    cprob = np.array([i/len(thetagcumobsv) for i in range(1,len(thetagcumobsv)+1)])
    x = np.linspace(0, max(thetagcumobsv), 200)
    plt.plot(thetagcumobsv,cprob)
    plt.plot(x, chi2.cdf(x, df=1))
    plt.savefig(figfn)
    plt.show()
    plt.clf()

ntrials = 100
g = 0.0 
thetaN = 25 #500
thetaS = 5 #500
thetaNequalthetaS = thetaN==thetaS
if thetaNequalthetaS:
    theta = thetaN
n=40
numnegdeltas = 0
if checkRationull: #check chi^2 approximation of LLR for ratio of PRF, uses a single theta
    figfn = "chisq_check_RATIO_thetaN{}_thetaS{}_g{}.pdf".format(thetaN,thetaS,g)
    ratiothetacumobsv = []
    for i in range(ntrials):
        nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,g,n)
        thetastart = 100.0
        gstart = -1.0
        maxi = n//2 - 1 # not used at the moment
        # fw2thetag0result = minimize_scalar(SFS_functions.negLfw2,bracket=(thetastart/10,thetastart*10),args = (maxi,n,nsfsfolded,ssfsfolded,True),method='Brent')        
        if thetaNequalthetaS:
            ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            ratiothetag0result =  minimize_scalar(SFS_functions.negL_ratio,bracket=(thetastart/10,thetastart*10),args = (maxi,n,ratios,True),method='Brent')        
        else:
            ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,thetastart,gstart]),args=(maxi,n,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
            ratiothetag0result = minimize(SFS_functions.negL_ratio,np.array([thetastart,thetastart]),args=(maxi,n,ratios,True),method="Powell",bounds=[(thetastart/10,thetastart*10),(thetastart/10,thetastart*10)])
        thetagdelta = -ratiothetagresult.fun + ratiothetag0result.fun
        if thetagdelta < 0:
            thetagdelta = 0.0
            numnegdeltas +=1
        ratiothetacumobsv.append(2*thetagdelta)
    ratiothetacumobsv.sort()
    ratiothetacumobsv = np.array(ratiothetacumobsv)
    cprob = np.array([i/len(ratiothetacumobsv) for i in range(1,len(ratiothetacumobsv)+1)])
    x = np.linspace(0, max(ratiothetacumobsv), 200)
    plt.plot(ratiothetacumobsv,cprob)
    plt.plot(x, chi2.cdf(x, df=1))
    plt.savefig(figfn)
    # print(numnegdeltas)
    plt.show()
    
