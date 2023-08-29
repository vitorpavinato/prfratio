
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
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2
import  SFS_functions
import math
import warnings
warnings.filterwarnings("ignore")

#---FUNCTION DEFINITION---#

# Load SFS(s) from a file
def loadSFSs(filename, path):
    file = (path + "/" + filename + ".txt")
    
    sfss = []
    for line in open(file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                # convert a list of strings to a list o ints
                counts_sfs = [eval(i) for i in line.strip().split()]
                counts_sfs.insert(0,0)
                sfss.append(counts_sfs)       
            except:
                print(sfss)
                pass
    
    return sfss

def averagesfsbin(sfss):
    arr = np.array(sfss) 
    avrs = sum(arr)/len(sfss)
    avrs = avrs.tolist()
    #avrs = [int(i) for i in avrs]
    
    return avrs

# Calculate the ratio of two SFSs
def sfsratios(basename, path):
    sfsfs = loadSFSs(filename = (basename + "_csfs_selected"), path=path)
    nfsfs = loadSFSs(filename = (basename + "_csfs_neutral"), path=path)

    if len(sfsfs) != len(nfsfs):
        raise Exception("Sorry, I can't handle it! The number of folded SFSs differ between sets!!!")
    
    fsfs_ratios = []
    el = 0
    for i in range(0,len(sfsfs)):
        if len(sfsfs[i]) != len(nfsfs[i]):
            raise Exception("Sorry, I can't handle it! The number of bins in folded SFSs # " + el +  "differ between sets!!!")
        el += 1
        ratio = np.divide(sfsfs[i],nfsfs[i]).tolist()
        ratio = [0 if math.isnan(x) else x for x in ratio]
        fsfs_ratios.append(ratio)

    n_avrgs = averagesfsbin(nfsfs)
    s_avrgs = averagesfsbin(sfsfs)
    r_avrgs = averagesfsbin(fsfs_ratios)
    
    return nfsfs, sfsfs, fsfs_ratios, n_avrgs, s_avrgs, r_avrgs


# Define the command line arguments
def parseargs():
    parser = argparse.ArgumentParser("python run_slim.py",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", help="number of simulations",
                        dest="nsimulations", default = 20, type=int)
    parser.add_argument("-f", help="genome-wide theta",
                        dest="theta", default=2000, type=int)
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
    
    # CMD arguments
    # ntrialsperg = 20
    # theta = 2000
    # sampleSize = 40
    # model = "popstructureN2"
    # parent_dir = "SLiM/results/prfratio"
    # out_dir = "results/SLiM"
      
    ntrialsperg = args.nsimulations
    theta = args.theta
    sampleSize = args.sampleSize
    model = args.model
    parent_dir = args.parent_dir
    out_dir = args.out_dir

    # Print arguments to the screen:
    print("Number of trials: " + str(ntrialsperg))
    print("Expected genome-wide theta: " + str(theta))
    print("Model: " + model)
    
    # Define file path
    parent_dir = os.path.join(parent_dir, model)
    print("Running at: " + parent_dir)
    
    # Create out dir if it doesn't exists
    out_dir_path = os.path.join(out_dir, model)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    print("Saving  at: " + out_dir_path + "\n")

    # Expected thetas - hard-coded
    thetaNeutral = 370.40 #1481.6
    thetaSelected = 129.6 #518.4
    #np.random.seed(1)
    dobasicPRF = True
    doratioPRF = True
    n=2*sampleSize #n is the number of chromosome not the number of individuals
    # gvals = [5,2.5,2,1.5,1,0.5,0.2,0.1,0.05,0.0,-0.05,-0.1,-0.2,-0.5,-1,-1.5,-2,-2.5,-5]
    gvals = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    # gvals.reverse()
    # = ntrialsperg

    # PRF for selected folded-SFSs
    if dobasicPRF:
        print("Running basic PRF model...")
        plotfilename = (out_dir_path + '/' + model + '_basicPRF_estimator_bias_var_theta{}.pdf'.format(theta))
        plotfilename_thetas = (out_dir_path + '/' + model + '_basicPRF_theta_bias_var_theta{}.pdf'.format(theta))
        plotfilename_nbins = (out_dir_path + '/' + model + '_mean_neutral_sfs_bins{}.pdf'.format(theta))
        plotfilename_sbins = (out_dir_path + '/' + model + '_mean_selected_sfs_bins{}.pdf'.format(theta))
        plotfilename_rbins = (out_dir_path + '/' + model + '_mean_ratio_sfs_bins{}.pdf'.format(theta))
        bins = list(range(0,sampleSize+1))
        l_bins = []
        results = []
        res_thetas = []
        res_sbins = []
        res_nbins = []
        res_rbins = []
        for gi,g in enumerate(gvals):
            print("Running Ns=" + str(g))
            results.append([])
            res_thetas.append([])
            # update path
            path = os.path.join(parent_dir, str(float(g)))
            nsfs, ssfs, ratios, n_avrgs, s_avrgs, r_avrgs = sfsratios(basename=(model + "_" + str(float(g))), path=path) #ideally should simulate from here
            l_bins.append(bins)
            res_sbins.append(s_avrgs)
            res_nbins.append(n_avrgs)
            res_rbins.append(r_avrgs)
            for i in range(ntrialsperg):
                print(i)
                thetastart = 100.0
                gstart = -1.0
                maxi = n//2 - 1 # not used at the moment
                thetagresult = minimize(SFS_functions.negLfw,np.array([thetastart,gstart]),args=(maxi,n,ssfs[i]),method="Powell",bounds=[(thetastart/5,thetastart*5),(15*gstart,-15*gstart)])
                # store estimated g values
                results[gi].append(thetagresult.x[1])
                res_thetas[gi].append(thetagresult.x[0])           

        # Make a barplot for estimated gvals 
        #gvals_scaled = [-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1] # for ibottleneckfixeds
        #gvals_scaled = [-2.0, -1.0, -0.4, -0.2,  0.0,  0.2,  0.4,  1.0,  2.0] # for popstructureN2 when s is not re-scaled
        print("Making some plots..." + "\n")
        fig, ax = plt.subplots()
        ax.boxplot(results,showmeans=True,sym='',positions=gvals)
        #ax.boxplot(results,showmeans=True,sym='',positions=gvals_scaled) # for ibottleneckfixeds
        plt.xlabel("g (selection)")
        plt.ylabel("Estimates")
        plt.plot(gvals,gvals)
        #plt.plot(gvals_scaled,gvals_scaled)
        plt.savefig(plotfilename)
        #plt.show()
        plt.clf

        # Make a barplot for thetas ~ gvals
        fig, ax = plt.subplots()
        ax.boxplot(res_thetas, showmeans=True,sym='',positions=gvals)
        #ax.boxplot(res_thetas, showmeans=True,sym='',positions=gvals_scaled) # for ibottleneckfixeds
        plt.axhline(y = thetaSelected, color = 'r', linestyle = '--')
        plt.xlabel("g (selection)")
        plt.ylabel("Theta selected")
        plt.savefig(plotfilename_thetas)
        # plt.show()
        plt.clf
        
        # Make a scatterplot of [N]SFSs ~ [S]SFSs
        # Flatten the nested lists
        x = [val for sublist in l_bins for val in sublist]
        y1 = [val for sublist in res_nbins for val in sublist]
        y2 = [val for sublist in res_sbins for val in sublist]
        y3 = [val for sublist in res_rbins for val in sublist]
        colors = [i for i in range(len(gvals)) for _ in range(len(res_nbins[i]))]
        classes = [str(i) for i in gvals]

        # New list ranges
        x = x[:-164]
        y1 = y1[:-164]
        y2 = y2[:-164]
        y3 = y3[:-164]
        colors = colors[:-164]
        
        # Mean [N]SFS ~ bin
        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y1, c=colors, cmap="Spectral")
        plt.xlabel('bins')
        plt.ylabel('[N]SFSs')
        legend = ax.legend(handles=scatter.legend_elements()[0], loc="upper right", title="Ns", labels=classes)
        ax.add_artist(legend)
        plt.savefig(plotfilename_nbins)
        plt.clf
        #plt.show()

        # Mean [S]SFS ~ bin
        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y2, c=colors, cmap="Spectral")
        plt.xlabel('bins')
        plt.ylabel('[S]SFSs')
        legend = ax.legend(handles=scatter.legend_elements()[0], loc="upper right", title="Ns", labels=classes)
        ax.add_artist(legend)
        plt.savefig(plotfilename_sbins)
        plt.clf
        #plt.show()

        # Mean ratio ~ bin
        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y3, c=colors, cmap="Spectral")
        plt.xlabel('bins')
        plt.ylabel('[Ratio]SFSs')
        legend = ax.legend(handles=scatter.legend_elements()[0], loc="upper left", title="Ns", labels=classes)
        ax.add_artist(legend)
        plt.savefig(plotfilename_rbins)
        plt.clf
        #plt.show()


    # PRF ratio
    if doratioPRF:
        print("Running PRF ratio model...")
        plotfilename = (out_dir_path + '/' + model + '_ratioPRF_estimator_bias_var_theta{}.pdf'.format(theta))
        plotfilename_thetas = (out_dir_path + '/' + model + '_ratioPRF_theta_bias_var_theta{}.pdf'.format(theta))
        results = []
        res_thetan = []
        res_thetas = []
        for gi,g in enumerate(gvals):
            print("Running Ns=" + str(g))
            results.append([])
            res_thetan.append([])
            res_thetas.append([])
            # update path
            path = os.path.join(parent_dir, str(float(g)))
            nsfs, ssfs, ratios, n_avrgs, s_avrgs, r_avrgs = sfsratios(basename=(model + "_" + str(float(g))), path=path) #ideally should simulate from here
            for i in range(ntrialsperg):
                print(i)
                thetastart = 100.0
                gstart = -1.0
                maxi = n//2 - 1 # not used at the moment
                ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,thetastart,gstart]),args=(maxi,n,ratios[i],False),method="Powell",bounds=[(thetastart/5,thetastart*5),(thetastart/5,thetastart*5),(30*gstart,-30*gstart)])
                #ratiothetagresult =  minimize(SFS_functions.negL_ratio,np.array([thetastart,thetastart,gstart]),args=(maxi,n,ratios[i],False),method="Nelder-Mead",bounds=[(thetastart/5,thetastart*5),(thetastart/5,thetastart*5),(15*gstart,-15*gstart)])
                # store estimated g values
                results[gi].append(ratiothetagresult.x[2]) 
                # store theta values
                res_thetan[gi].append(ratiothetagresult.x[0]) 
                res_thetas[gi].append(ratiothetagresult.x[1])            
        
        # Make a barplot
        print("Making some plots..." + "\n")
        fig, ax = plt.subplots()
        ax.boxplot(results,showmeans=True,sym='',positions=gvals)
        #ax.boxplot(results,showmeans=True,sym='',positions=gvals_scaled) # for ibottleneckfixeds
        plt.xlabel("g (selection)")
        plt.ylabel("Estimates")
        plt.plot(gvals,gvals)
        #plt.plot(gvals_scaled,gvals_scaled)
        plt.savefig(plotfilename)
        # plt.show()
        plt.clf

        # Make a barplot for thetas ~ gvals
        fig, ax = plt.subplots()
        ax.boxplot(res_thetan, showmeans=True,sym='', positions=gvals)
        ax.boxplot(res_thetas, showmeans=True,sym='', positions=gvals)
        #ax.boxplot(res_thetan, showmeans=True,sym='', positions=gvals_scaled)
        #ax.boxplot(res_thetas, showmeans=True,sym='', positions=gvals_scaled)
        plt.axhline(y = thetaSelected, color = 'r', linestyle = '--')
        plt.axhline(y = thetaNeutral, color = 'b', linestyle = '--')
        plt.xlabel("g (selection)")
        plt.ylabel("Thetas")
        plt.savefig(plotfilename_thetas)
        # plt.show()
        plt.clf

if __name__ == "__main__":

    if len(sys.argv) < 2:
        main(['-h'])
    else:
        main(sys.argv[1:])

