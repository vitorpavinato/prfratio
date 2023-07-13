
"""
runs an analysis of estimator bias and variance .  
For a range of true values of the selection coefficient g generate boxplots of estimates 

User must set theta 

"""
import os
import time
import sys
import argparse
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2
import  SFS_functions
import math
import warnings
warnings.filterwarnings("ignore")

#---FUNCTION DEFINITION---#

# Load SFS(s) from a file
def loadSFSs(filename, path, proportions=False, subsample=0):
    file = (path + "/" + filename + ".txt")
    
    sfss = []

    for line in open(file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                # convert a list of strings to a list o ints
                counts_sfs = [eval(i) for i in line.strip().split()]
                counts_sfs.insert(0,0)
                if (proportions):
                    total = sum(counts_sfs)
                    arr = np.array(counts_sfs)
                    freq_sfs = np.divide(arr, total).tolist()
                    sfss.append(freq_sfs)
                else:
                    if (subsample > 0):
                        sub = np.divide(counts_sfs, subsample).tolist()
                        sub = [int(i) for i in sub]
                        sfss.append(sub)
                    else:
                        sfss.append(counts_sfs)
                        
            except:
                print(sfss)
                pass

    return sfss

# Calculate the ratio of two SFSs
def sfsratios(basename, path, proportions=False):
    sfsfs = loadSFSs(filename = (basename + "_csfs_selected"), path=path, proportions=proportions, subsample=0)
    nfsfs = loadSFSs(filename = (basename + "_csfs_neutral"), path=path, proportions=proportions, subsample=2.5)

    if len(sfsfs) != len(nfsfs):
        raise Exception("Sorry, the number of folded SFSs differ between sets!!!")
    
    fsfs_ratios = []
    el = 0
    for i in range(0,len(sfsfs)):
        if len(sfsfs[i]) != len(nfsfs[i]):
            raise Exception("Sorry, the number of bins in folded SFSs # " + el +  "differ between sets!!!")
        el += 1
        ratio = np.divide(sfsfs[i],nfsfs[i]).tolist()
        ratio = [0 if math.isnan(x) else x for x in ratio]
        fsfs_ratios.append(ratio)

    return nfsfs, sfsfs, fsfs_ratios


# Define the command line arguments
def parseargs():
    parser = argparse.ArgumentParser("python run_slim.py",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", help="number of simulations",
                        dest="nsimulations", default = 100, type=int)
    parser.add_argument("-L", help="ThetaL value",
                        dest="thetaL", default=10, type=int)
    parser.add_argument("-f", help="Number of sequences",
                        dest="nSeqs", default=200, type=int)
    parser.add_argument("-n", help="Sample size",
                        dest="sampleSize", default=40, type=int)
    parser.add_argument("-m", help="Model",
                        dest="model",required = True,type = str)
    parser.add_argument("-d", help="Parent directory path to import results files from",
                        dest="parent_dir",default = "SLiM/results/prfratio",type = str)
    parser.add_argument("-o", help="Parent directory path to export results files from",
                        dest="out_dir",default = "results/SLiM",type = str)
    return parser


# argv = "-r 100 -L 10 -f 200 -n 40 -m constant -d results/prfration -o results"
# sys.argv = argv.split()
# print(sys.argv)

def main(argv):

    starttime = time.time()
    parser = parseargs()
    if argv[-1] =='':
        argv = argv[0:-1]
    args = parser.parse_args(argv)
    
    #CMD arguments
    # ntrialsperg = 100
    # thetaL = 10
    # nSeqs = 200
    # sampleSize = 40
    # model = "constant"
    # parent_dir = "SLiM/results/prfratio"
    # out_dir = "results/SLiM"
      
    ntrialsperg = args.nsimulations
    thetaL = args.thetaL
    nSeqs = args.nSeqs
    sampleSize = args.sampleSize
    model = args.model
    parent_dir = args.parent_dir
    out_dir = args.out_dir

    # Define file path
    path = os.path.join(parent_dir, model)

    # Create out dir if it doesn't exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #np.random.seed(1)
    dobasicPRF = True
    doratioPRF = True
    n=2*sampleSize
    theta = thetaL*nSeqs
    # gvals = [5,2.5,2,1.5,1,0.5,0.2,0.1,0.05,0.0,-0.05,-0.1,-0.2,-0.5,-1,-1.5,-2,-2.5,-5]
    gvals = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    # gvals.reverse()
    ntrialsperg = ntrialsperg
    
    if dobasicPRF:
        plotfilename = (out_dir + '/' + 'basicPRF_estimator_bias_var_theta{}.pdf'.format(theta))
        results = []
        for gi,g in enumerate(gvals):
            results.append([])
            nsfs, ssfs, ratios = sfsratios(basename=(model + "_" + str(float(g))), path=path, proportions=False)
            for i in range(ntrialsperg):
                #sfs,sfsfolded = SFS_functions.simsfs(theta,g,n)
                thetastart = 100.0
                gstart = -1.0
                maxi = n//2 - 1 # not used at the moment
                thetagresult = minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,ssfs[i]),method="Powell",bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
                results[gi].append(thetagresult.x[1])            

        # Make a barplot
        fig, ax = plt.subplots()
        ax.boxplot(results,showmeans=True,sym='',positions=gvals)
        plt.xlabel("g (selection)")
        plt.ylabel("Estimates")
        plt.plot(gvals,gvals)
        plt.savefig(plotfilename)
        # plt.show()
        plt.clf

    if doratioPRF:
        plotfilename = (out_dir + '/' + 'ratioPRF_estimator_bias_var_theta{}.pdf'.format(theta))
        results = []
        for gi,g in enumerate(gvals):
            results.append([])
            nsfs, ssfs, ratios = sfsratios(basename=(model + "_" + str(float(g))), path=path, proportions=False)
            for i in range(ntrialsperg):
                #nsfsfolded,ssfsfolded,ratios = SFS_functions.simsfsratio(theta,theta,g,n)
                thetastart = 100.0
                gstart = -1.0
                maxi = n//2 - 1 # not used at the moment
                ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios[i],False),method="Powell",bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
                # ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,gstart]),args=(maxi,n,ratios,False),method="Nelder-Mead",bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
                results[gi].append(ratiothetagresult.x[1])            
        
        # Make a barplot
        fig, ax = plt.subplots()
        ax.boxplot(results,showmeans=True,sym='',positions=gvals)
        plt.xlabel("g (selection)")
        plt.ylabel("Estimates")
        plt.plot(gvals,gvals)
        plt.savefig(plotfilename)
        # plt.show()
        plt.clf

if __name__ == "__main__":

    if len(sys.argv) < 2:
        main(['-h'])
    else:
        main(sys.argv[1:])