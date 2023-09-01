"""
    check the log likelihood ratio (LLR) statistic against chi^2 distributions
    if dobasicPRF is True:
        SFS simulations and a figure are generated for the conventional LLR test of a PRF Fisher Wright model
        the results will be saved in a file  chisq_check_PRF_theta{}_g{}.pdf 
        The comparison is mostly only relevant when the null model (i.e. g=0) is true 

        as of 8/24/2023,  this only seems to work for folded distributions
    if doratioPRF is True:
        simulations and a figure are generated for the LLR test or a ratio of two SFSs 
        two SFS simulations are done (numerator and denominator), and so there are two thetas (thetaS and thetaN)
        It is assumed that thetaS==thetaN,  so there is only one theta to be estimated 
        the results will be saved in a file  chisq_check_RATIO_thetaN{}_thetaS{}_g{}.pdf 
        As with PRF case, the comparison is mostly only relevant when the null model (i.e. g=0) is true        
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
import  SFS_functions

dobasicPRF = True
doratioPRF = True 
theta = 100
n=50

np.random.seed(1)
ntrials = 100
g = 0.0 # fixed at 0,  we assume the null model is true 
dofolded = True # False

foldstring = "folded" if dofolded else "unfolded"
if dobasicPRF: # regular PRF LLR stuff
    figfn = "chisq_check_PRF_theta{}_g{}_n{}_{}.pdf".format(theta,g,n,foldstring)
    thetagcumobsv = []
    numnegdeltas = 0
    #check chi^2 approximation of LLR for basic FW PRF 
    for i in range(ntrials):
        sfs,sfsfolded = SFS_functions.simsfs(theta,g,n,None,False)
        sfs = sfsfolded if dofolded else sfs
        thetastart = 100.0
        gstart = -1.0
        thetagresult = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,sfs),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
        g0result = minimize_scalar(SFS_functions.NegL_SFS_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,sfs),method='Brent')

        thetagdelta = -thetagresult.fun + g0result.fun
        if thetagdelta < 0:
            thetagdelta = 0.0
            numnegdeltas +=1        
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




if doratioPRF: #check chi^2 approximation of LLR for ratio of PRF, uses a single theta
    SFS_functions.SSFconstant_dokuethe = False # see SFS_functions
    figfn = "chisq_check_RATIO_thetaN{}_thetaS{}_g{}_n{}_{}.pdf".format(theta,theta,g,n,foldstring)
    ratiothetacumobsv = []
    numnegdeltas = 0
    for i in range(ntrials):
        nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,1.0,n,None,dofolded,None,g,False)
        thetastart = 100.0
        gstart = -1.0
        ratiothetagresult =  minimize(SFS_functions.NegL_SFSRATIO_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
        ratiothetag0result =  minimize_scalar(SFS_functions.NegL_SFSRATIO_Theta_Ns,bracket=(thetastart/10,thetastart*10),args = (n,dofolded,ratios,True),method='Brent')        
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
    
