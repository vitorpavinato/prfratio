

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
import  SFS_functions
import math
import warnings
warnings.filterwarnings("ignore")

def valstoscreen(fn,thresholds,pvalues,TPrate,FPrate):
    print("{}\nT\tp\tTPrate\tFPrate".format(fn))
    for i in range(len(thresholds)):
        print("{:.2g}\t{:.5f}\t{:.4f}\t{:.4f}".format(thresholds[i],pvalues[i],TPrate[i],FPrate[i]))

pvalues = [1,0.999, 0.995, 0.99, 0.9, 0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 
    0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001,0]
x2thresholds = [0.0,1.5708e-6, 0.0000392704, 0.000157088, 0.0157908, 0.454936, 
        2.70554, 3.84146, 5.41189, 6.6349, 7.87944, 9.54954, 10.8276, 
        12.1157, 13.8311, 15.1367, 16.4481, 18.1893, 19.5114,math.inf]
x2thresholds.reverse()
pvalues.reverse()
np.random.seed(1)
dobasicPRFROC = True
doratioPRFROC = True
theta = 25
gmax = 100
ntrials = 1000
if dobasicPRFROC:
    n=40
    numg0 = ntrials//2
    rocfilename = 'basicPRF_ROC_theta{}_gmax{}.pdf'.format(theta,gmax)
    # x2thresholds.reverse()
    results = [[0.0,0] for i in range(numg0)]
    for i in range(ntrials-numg0):
        results.append([-np.random.random()*gmax,1])
    for i in range(ntrials):
        g = results[i][0]
        sfs,sfsfolded = SFS_functions.simsfs(theta,g,n)
        thetastart = 100.0
        gstart = -1.0
        maxi = n//2 - 1 # not used at the moment
        thetagresult = minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,sfsfolded),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
        g0result = minimize_scalar(SFS_functions.negLfw,bracket=(thetastart/10,thetastart*10),args = (maxi,n,sfsfolded),method='Brent')
        thetagdelta = 2*(-thetagresult.fun + g0result.fun)
        for t in x2thresholds:
            if thetagdelta > t:
                results[i].append(1)
            else:
                results[i].append(0)
    TPrate = []
    FPrate = []
    for ti,t in enumerate(x2thresholds):
        tc = 0
        tpc = 0
        fc = 0
        fpc = 0
        for r in results:
            tc += r[1] == 1
            tpc += r[1]==1 and r[2+ti]==1
            fc += r[1] == 0
            fpc += r[1] == 0 and r[2+ti] == 1
        TPrate.append(tpc/tc)
        FPrate.append(fpc/fc)
    AUC = sum([(FPrate[i+1]-FPrate[i])*(TPrate[i+1]+TPrate[i])/2 for i in range(len(TPrate)-1) ])
    plt.plot(FPrate,TPrate)
    plt.title("Poisson Random Field Likelihood Ratio Test ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.text(0.1,0.1,"AUC = {:.3f}".format(AUC),fontsize=14)
    # plt.show()
    # exit()
    plt.savefig(rocfilename)
    plt.clf()
    valstoscreen(rocfilename,x2thresholds,pvalues,TPrate,FPrate)
if doratioPRFROC:
    n=40
    numg0 = ntrials//2
    rocfilename = 'ratioPRF_ROC_theta{}_gmax{}.pdf'.format(theta,gmax)
    # x2thresholds.reverse()
    results = [[0.0,0] for i in range(numg0)]
    for i in range(ntrials-numg0):
        results.append([-np.random.random()*gmax,1])
    for i in range(ntrials):
        g = results[i][0]
        nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,g,n)
        thetastart = 100.0
        gstart = -1.0
        maxi = n//2 - 1 # not used at the moment
        ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios,False),method="Powell",bounds=[(thetastart/10,thetastart*10),(10*gstart,-10*gstart)])
        ratiothetag0result =  minimize_scalar(SFS_functions.negL_ratio,bracket=(thetastart/10,thetastart*10),args = (maxi,n,ratios,True),method='Brent')        
        thetagdelta = 2*(-ratiothetagresult.fun + ratiothetag0result.fun)
        for t in x2thresholds:
            if thetagdelta > t:
                results[i].append(1)
            else:
                results[i].append(0)
    TPrate = []
    FPrate = []
    for ti,t in enumerate(x2thresholds):
        tc = 0
        tpc = 0
        fc = 0
        fpc = 0
        for r in results:
            tc += r[1] == 1
            tpc += r[1]==1 and r[2+ti]==1
            fc += r[1] == 0
            fpc += r[1] == 0 and r[2+ti] == 1
        TPrate.append(tpc/tc)
        FPrate.append(fpc/fc)
    AUC = sum([(FPrate[i+1]-FPrate[i])*(TPrate[i+1]+TPrate[i])/2 for i in range(len(TPrate)-1) ])
    plt.plot(FPrate,TPrate)
    plt.title("Site Frequency Ratio Likelihood Ratio Test ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.text(0.1,0.1,"AUC = {:.3f}".format(AUC),fontsize=14)
    plt.savefig(rocfilename)
    valstoscreen(rocfilename,x2thresholds,pvalues,TPrate,FPrate)
    # plt.show()

