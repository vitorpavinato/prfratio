
"""
runs an analysis of estimator bias and variance .  
For a range of true values of the selection coefficient g generate boxplots of estimates 

User must set theta 

when using external data, use -a   
code must be updated to deal with this each each point where
    SFS_functions.simsfs_gdist or SFS_functions.simsfs  is called

usage: SFS_estimator_bias_variance.py [-h] [-d DENSITYOF2NS] [-f] -k NTRIALS [-l PLOTFILELABEL]
                                      [-m MAXI] -n N [-p] -q THETAS [-r] [-s SEED] [-t THETAN]
                                      [-w] [-x GDENSITYMAX] [-z]

options:
  -h, --help        show this help message and exit
  -d DENSITYOF2NS   gamma or lognormal, only if simulating a distribution of Ns, else single
                    values of Ns are used
  -f                use folded SFS distribution
  -k NTRIALS        number of trials per parameter set
  -l PLOTFILELABEL  optional string for labelling plot file names
  -m MAXI           optional setting for the maximum bin index to include in the calculations
  -n N              sample size
  -p                simulate Fisher-Wright SFS, basic PRF
  -q THETAS         theta for selected sites
  -r                simulate Fisher-Wright RATIO, RATIOPRF
  -s SEED           random number seed (positive integer)
  -t THETAN         set theta for neutral sites, optional when -r is used, if -t is not
                    specified then thetaN and thetaS are given by -q
  -w                use watterson estimator for thetaN
  -x GDENSITYMAX    maximum value of 2Ns density, default is 1.0, use with -d
  -z                estimate the maximum of the density of 2Ns


"""
import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import chi2
import SFS_functions
import math
import argparse
import simulate_SFS_withSLiM as slim
# import warnings
# warnings.filterwarnings("ignore")

# as of 8/23/2023 Kuethe method does not work as well
# if SFS_functions.SSFconstant_dokuethe:
#     optimizemethod="Powell"
# else:
#     optimizemethod="Nelder-Mead"
optimizemethod="Powell"

np.seterr(divide='ignore', invalid='ignore')

def makeSFScomparisonstring(headers,sfslist):
    slist = []
    slist.append("\t".join(headers) + "\n")
    n = len(sfslist[0])
    k = len(sfslist)
    for i in range(n):
        if k ==6:
            temp = ["{}".format(sfslist[0][i]),"{}".format(sfslist[1][i]),"{:.3f}".format(sfslist[2][i]),"{}".format(sfslist[3][i]),"{}".format(sfslist[4][i]),"{:.3f}".format(sfslist[5][i])]
        else:
            temp = ["{}".format(sfslist[j][i]) for j in range(k)]
        temp.insert(0,str(i))
        slist.append("\t\t" + "\t".join(temp)+"\n")
    slist.append("\n")   
    return ''.join(slist) 

def make_outfile_name_base(args):
    if args.plotfilelabel != "" and args.plotfilelabel[-1] != "_": # add a spacer if needed
        args.plotfilelabel += "_"  
    a = []
    if args.dobasicPRF:
        a.append("{}basicPRF_k{}_n{}_Qs{:.0f}_".format(args.plotfilelabel,args.ntrials,args.n,args.thetaS))
        a.append("")
    else:
        a.append("{}ratioPRF_k{}_n{}_Qs{:.0f}_Qn{:.0f}_".format(args.plotfilelabel,args.ntrials,args.n,args.thetaS,args.thetaN))
    if args.dofolded:
        a.append("folded_")
    if args.densityof2Ns:
        a.append("{}_".format(args.densityof2Ns))
    if args.gdensitymax != 1.0:
        a.append("gmax{}_".format(args.gdensitymax))
    if args.use_watterson_thetaN:
        a.append("WQn_".format(args.densityof2Ns))        
    if args.maxi:
        a.append("maxi{}_".format(args.maxi))
    basename = ''.join(a)
    if basename[-1] =='_':
        basename = basename[:-1]
    # print (args)
    # print(basename)
    return basename


def run(args):

    SFS_functions.SSFconstant_dokuethe = False #  True

    np.random.seed(args.seed)
    ntrialsperg = args.ntrials
    dobasicPRF = args.dobasicPRF
    doratioPRF = args.doratioPRF
    densityof2Ns = args.densityof2Ns 
    dofolded = args.dofolded
    gdm = args.gdensitymax
    if gdm != 1.0:
        SFS_functions.reset_g_xvals(gdm)

    n = args.n

    # Additional parameters for SLiM
    simuslim = args.simuslim
    model = args.model
    mu = args.mu
    rec = args.rec
    popSize = args.popSize
    seqLen = args.seqLen
    nSeqs = args.nSeqs
    parent_dir = args.parent_dir
    savefile = args.savefile
    
    if args.thetaN:
        thetaN = args.thetaN
        thetaNresults = []
    else:
        thetaN = args.thetaS
        args.thetaN = args.thetaS
    thetaS = args.thetaS
    foldstring = "folded" if dofolded else "unfolded"
    thetaSresults = []
    if densityof2Ns == 'lognormal':  
        gvals = [[0.3, 0.5], [1, 0.7], [2.0, 1.0], [2.2, 1.4], [3, 1.2]]
        ln1results = []
        ln2results = []
    elif densityof2Ns == 'gamma':
        gvals = [(2,0.5),(1.6,1.3),(2.5,3.0),(4,1.8),(4.5,2.2)]
        ln1results = []
        ln2results = []
    else: #densityof2Ns==None
        gvals = [-10,-5,-2,-1,0,1,2,5,10]
        gresults = []
    thetastart = 200 
    gdensitystart = [2.0,1.0]
    SFScomparesultsstrings = []
    basename = make_outfile_name_base(args)
    if densityof2Ns:
        plotfile1name = '{}_term1_plot.png'.format(basename)
        plotfile2name = '{}_term2_plot.png'.format(basename)
        
    else:
        plotfilename = '{}_2Ns_plot.png'.format(basename)
    estimates_filename = '{}_results.txt'.format(basename)    
    
    if dobasicPRF:
        savedSFSS = [[] for i in range(len(gvals))]
        results = []
        for gi,g in enumerate(gvals):
            thetaSresults.append([])
            if densityof2Ns:
                ln1results.append([])
                ln2results.append([])
            else:
                gresults.append([])  
                
            for i in range(ntrialsperg):
                if densityof2Ns:
                    if simuslim:
                        sims_csfs_neutral, sfsfolded, sims_csfs_ratio, sims_nss, sims_seeds = slim.simulateSFSslim(nsimulations = 1, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                                                                                   nsdist = densityof2Ns, nsdistargs = g, sampleSize = int(n/2), model = model, 
                                                                                                                   nSeqs =nSeqs, parent_dir = parent_dir, savefile = savefile)

                    else:    
                        sfs,sfsfolded =  SFS_functions.simsfs_continuous_gdist(thetaS,gdm,n,args.maxi,densityof2Ns,(g[0],g[1]),False)  
                    
                    startarray = [thetastart,gdensitystart[0],gdensitystart[1]]
                    boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]  
                    
                    result = minimize(SFS_functions.NegL_SFS_ThetaS_Ns_density,np.array(startarray),args=(gdm,n,args.dofolded,densityof2Ns,sfsfolded if dofolded else sfs),method=optimizemethod,bounds=boundsarray)
                    thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                    SFScompareheaders = ["Params:{} {} Trial#{}:".format(g[0],g[1],i+1),"sim","fit"]
                    sfsfit,sfsfoldedfit =  SFS_functions.simsfs_continuous_gdist(thetaSresults[gi][-1],gdm,n,args.maxi,densityof2Ns,(ln1results[gi][-1],ln2results[gi][-1]),True)  
                else:
                    if simuslim:
                        sims_csfs_neutral, sfsfolded, sims_csfs_ratio, sims_nss, sims_seeds = slim.simulateSFSslim(nsimulations = 1, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                                                                                   ns = g, nsdist = "fixed", sampleSize = int(n/2), model = model, 
                                                                                                                   nSeqs = nSeqs, parent_dir = parent_dir, savefile = savefile)

                    else:
                        sfs,sfsfolded = SFS_functions.simsfs(thetaS,g,n,args.maxi,False)
                    
                    thetastart = 100.0
                    gstart = -1.0
                    boundsarray = [(thetastart/10,thetastart*10),(15*gstart,-15*gstart)]  
                    result = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,sfsfolded if dofolded else sfs),method=optimizemethod,bounds=boundsarray)
                    thetaSresults[gi],gresults[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],gresults[gi]], result.x)]   
                    SFScompareheaders = ["Param:{} Trial#{}:".format(g,i+1),"sim","fit"]                 
                    sfsfit,sfsfoldedfit = SFS_functions.simsfs(thetaSresults[gi][-1],gresults[gi][-1],n,args.maxi,True)
                SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[sfs,sfsfit]))
                savedSFSS[gi].append(sfsfolded if dofolded else sfs)

        if densityof2Ns: 
            term1vals = []
            term2vals = []
            for gi,g in enumerate(gvals):
                term1vals.append(g[0])
                term2vals.append(g[1])
            fig, ax = plt.subplots()
            ax.boxplot(ln1results,showmeans=True,sym='',positions=term1vals)
            plt.xlabel("{} term 1".format(densityof2Ns,n,foldstring))
            plt.ylabel("Estimates")
            plt.plot(gvals,gvals)
            plt.savefig(plotfile1name)
            plt.clf
            fig, ax = plt.subplots()
            ax.boxplot(ln2results,showmeans=True,sym='',positions=term2vals)
            plt.xlabel("{} term 2".format(densityof2Ns,n,foldstring))
            plt.ylabel("Estimates")
            plt.plot(gvals,gvals)
            plt.savefig(plotfile2name)
            plt.clf

        else:
            fig, ax = plt.subplots()
            ax.boxplot(gresults,showmeans=True,sym='',positions=gvals)
            plt.xlabel("g (selection)")
            plt.ylabel("Estimates")
            plt.plot(gvals,gvals)
            plt.savefig(plotfilename)
            plt.clf
        f = open(estimates_filename,"w")
        f.write("Program SFS_estimator_bias_variance.py results:\n\nCommand line arguments:\n=======================\n")
        for key, value in vars(args).items():
            f.write("\t{}: {}\n".format(key,value))
        f.write("\nCompare simulated and fitted SFS:\n=================================\n")
        f.write(''.join(SFScomparesultsstrings))
        f.write("Parameter Estimates:\n===================\n")
        if densityof2Ns:
            for gi,g in enumerate(gvals):
                f.write("\tSet {} Values:\t\tThetaS {}\t\tg1 {}\t\tg2 {}\n".format(gi+1,args.thetaS,g[0],g[1]))
                for k in range(ntrialsperg):
                    f.write("\t\t{}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(k+1,thetaSresults[gi][k],ln1results[gi][k],ln2results[gi][k]))
                f.write("\tMean:\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi]))))
                f.write("\tStDev:\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi]))))
            
        else:     
            for gi,g in enumerate(gvals):
                f.write("\tSet {} Values:\t\tThetaS {}\t\tg {}\n".format(gi+1,args.thetaS,g))
                for k in range(ntrialsperg):
                    f.write("\t\t{}\t\t{:.1f}\t\t{:.2f}\n".format(k+1,thetaSresults[gi][k],gresults[gi][k]))
                f.write("\tMean:\t\t{:.1f}\t\t{:.2f}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(gresults[gi]))))
                f.write("\tStDev:\t\t{:.1f}\t\t{:.2f}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(gresults[gi]))))
        f.write("\n\n")   

        f.write("Mean SFS Counts For Each Parameter Set:\n=================================================\n")     
        f.write("Selection parameter sets: {}\n".format(" ".join(list(map(str,gvals)))))
        for i in range(len(savedSFSS[0][0])):
            f.write("{}".format(i))
            for gi in range(len(gvals)):
                stemp = sum([savedSFSS[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                f.write("\t{:.1f}".format(stemp))
            f.write("\n")
        f.close()

    if doratioPRF:
        thetaNresults = []
        savedSFSS = [[] for i in range(len(gvals))]  
        savedSFSN = [[] for i in range(len(gvals))]      
        savedRATIOs = [[] for i in range(len(gvals))]
        results = []
        for gi,g in enumerate(gvals):
            
            thetaSresults.append([])
            thetaNresults.append([])
            if densityof2Ns:
                ln1results.append([])
                ln2results.append([])
            else:
                gresults.append([])
            for i in range(ntrialsperg):
                thetastart = 200
                if densityof2Ns:
                    SFScompareheaders = ["Params:{} {} Trial#{}:".format(g[0],g[1],i+1),"Nsim","Ssim","Ratiosim","Nfit","Sfit","Ratiofit"]
                    if simuslim:
                        nsfs, ssfs, ratios, sims_nss, sims_seeds = slim.simulateSFSslim(nsimulations = 1, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                                                        nsdist = densityof2Ns, nsdistargs = g, sampleSize = int(n/2), model = model, 
                                                                                        nSeqs = nSeqs, parent_dir = parent_dir, savefile = savefile)

                    else:
                        nsfs,ssfs,ratios =  SFS_functions.simsfsratio(thetaN,thetaS,gdm,n,args.maxi,dofolded,densityof2Ns,g,False) 
                    if args.use_watterson_thetaN:
                        thetaNest = sum(nsfs)/sum([1/i for i in range(1,n)]) # this should work whether or not the sfs is folded 
                        boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]          
                        startarray = [thetastart,gdensitystart[0],gdensitystart[1]]  
                        result = minimize(SFS_functions.NegL_SFSRATIO_Theta_Nsdensity_given_thetaN,np.array(startarray),args=(gdm,n,thetaN,dofolded,densityof2Ns,ratios),method=optimizemethod,bounds=boundsarray) 
                        thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                        thetaNresults[gi].append(thetaNest)                         
                    else:
                        boundsarray = [(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]     
                        startarray = [thetastart,thetastart,gdensitystart[0],gdensitystart[1]]   
                        result = minimize(SFS_functions.NegL_SFSRATIO_Theta_Nsdensity,np.array(startarray),args=(gdm,n,dofolded,densityof2Ns,ratios),method=optimizemethod,bounds=boundsarray)  
                        thetaNresults[gi],thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaNresults[gi],thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                    fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],gdm,n,args.maxi,dofolded,densityof2Ns,[ln1results[gi][-1],ln2results[gi][-1]],True)
                    SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[nsfs,ssfs,ratios,fitnsfs,fitssfs,fitratios]))
                else:
                    SFScompareheaders = ["Param:{} Trial#{}:".format(g,i+1),"Nsim","Ssim","Ratiosim","Nfit","Sfit","Ratiofit"]
                    if simuslim:
                        nsfs, ssfs, ratios, sims_nss, sims_seeds = slim.simulateSFSslim(nsimulations = 1, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                                                        ns = g, nsdist = "fixed", sampleSize = int(n/2), model = model, 
                                                                                        nSeqs = nSeqs, parent_dir = parent_dir, savefile = savefile)
                        
                    else:
                        nsfs,ssfs,ratios =  SFS_functions.simsfsratio(thetaN,thetaS,gdm,n,args.maxi,dofolded,None,g,False)
                    thetastart = 100.0
                    gstart = -1.0
                    if args.use_watterson_thetaN:
                        thetaNest = sum(nsfs)/sum([1/i for i in range(1,n)]) # this should work whether or not the sfs is folded 
                        boundsarray = [(thetastart/10,thetastart*10),(15*gstart,-15*gstart)]     
                        startarray = [thetastart,gstart]
                        nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Ns_given_thetaN 
                        nlargs = (n,thetaNest,dofolded,ratios,False)
                    else:
                        nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Ns
                        nlargs = (n,dofolded,ratios,False)
                        boundsarray = [(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(15*gstart,-15*gstart)]     
                        startarray = [thetastart,thetastart,gstart]
                    result =  minimize(nlfunc,np.array(startarray),args=nlargs,method=optimizemethod,bounds=boundsarray)
                    if args.use_watterson_thetaN:
                        thetaSresults[gi],gresults[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],gresults[gi]], result.x)]
                        thetaNresults[gi].append(thetaNest)                        
                    else:
                        thetaNresults[gi],thetaSresults[gi],gresults[gi] = [lst + [val] for lst, val in zip([thetaNresults[gi],thetaSresults[gi],gresults[gi]], result.x)]
                    fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],gdm,n,args.maxi,dofolded,None,gresults[gi][-1],True)
                    SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[nsfs,ssfs,ratios,fitnsfs,fitssfs,fitratios]))
                savedSFSS[gi].append(ssfs)
                savedSFSN[gi].append(nsfs)
                savedRATIOs[gi].append(ratios)
        if densityof2Ns:
            term1vals = []
            term2vals = []
            for gi,g in enumerate(gvals):
                term1vals.append(g[0])
                term2vals.append(g[1])
            fig, ax = plt.subplots()
            ax.boxplot(ln1results,showmeans=True,sym='',positions=term1vals)
            plt.xlabel("{} term 1".format(densityof2Ns,n,foldstring))
            plt.ylabel("Estimates")
            plt.plot(gvals,gvals)
            plt.savefig(plotfile1name)
            plt.clf
            fig, ax = plt.subplots()
            ax.boxplot(ln2results,showmeans=True,sym='',positions=term2vals)
            plt.xlabel("{} term 2".format(densityof2Ns,n,foldstring))
            plt.ylabel("Estimates")
            plt.plot(gvals,gvals)
            plt.savefig(plotfile2name)
            plt.clf
        else:
            fig, ax = plt.subplots()
            ax.boxplot(gresults,showmeans=True,sym='',positions=gvals)
            plt.xlabel("g (selection)")
            plt.ylabel("Estimates")
            plt.plot(gvals,gvals)
            plt.savefig(plotfilename)
            plt.clf            
        f = open(estimates_filename,"w")
        f.write("Program SFS_estimator_bias_variance.py results:\n\nCommand line arguments:\n=======================\n")
        for key, value in vars(args).items():
            f.write("\t{}: {}\n".format(key,value))
        f.write("\nCompare simulated and fitted SFS:\n=================================\n")
        f.write(''.join(SFScomparesultsstrings))

        # f = open(estimates_filename,"a")
        f.write("Parameter Estimates:\n===================\n")
        for gi,g in enumerate(gvals):
            if densityof2Ns:
                f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg1 {}\t\tg2 {}\n".format(gi+1,args.thetaS,args.thetaN,g[0],g[1]))
                for k in range(ntrialsperg):
                    f.write("\t\t{}\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],ln1results[gi][k],ln2results[gi][k]))
                f.write("\tMean:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi]))))
                f.write("\tStDev:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi]))))
            else:
                f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg {}\n".format(gi+1,args.thetaS,args.thetaN,g))
                for k in range(ntrialsperg):
                    f.write("\t\t{}\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],gresults[gi][k]))
                f.write("\tMean:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(gresults[gi]))))
                f.write("\tStDev:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(gresults[gi]))))
        f.write("\n\n")
        f.write("Mean SFS Counts For Each Parameter Set (Neutral, Selected, Ratio):\n==================================================================\n")     
        f.write("Selection parameter sets: {}\n".format(" ".join(list(map(str,gvals)))))
        for gi in range(len(gvals)):   
            f.write("\tNeutral\tSelect\tRatio")
        f.write("\n")
        for i in range(len(savedSFSS[0][0])):
            f.write("{}".format(i))
            for gi in range(len(gvals)):
                ntemp = sum([savedSFSN[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                stemp = sum([savedSFSS[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                rtemp = sum([savedRATIOs[gi][j][i] for j in range (ntrialsperg)])/ntrialsperg
                f.write("\t{:.1f}\t{:.1f}\t{:.3f}".format(ntemp,stemp,rtemp))
            f.write("\n")
        f.close()            


def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",dest="densityof2Ns",default = None,type=str,help="gamma or lognormal, only if simulating a distribution of Ns, else single values of Ns are used")
    parser.add_argument("-f",dest="dofolded",action="store_true",default=False,help="use folded SFS distribution")    
    parser.add_argument("-k",dest="ntrials",type = int, help="number of trials per parameter set", required=True)    
    parser.add_argument("-l",dest="plotfilelabel",default = "", type=str, help="optional string for labelling plot file names ")    
    parser.add_argument("-m",dest="maxi",default=None,type=int,help="optional setting for the maximum bin index to include in the calculations")
    parser.add_argument("-n",dest="n",type = int, help="sample size", required=True)
    parser.add_argument("-p",dest="dobasicPRF",action="store_true",default=False,help="simulate Fisher-Wright SFS, basic PRF")
    parser.add_argument("-q",dest="thetaS",type=float,help = "theta for selected sites",required=True)    
    parser.add_argument("-r",dest="doratioPRF",action="store_true",default=False,help="simulate Fisher-Wright RATIO, RATIOPRF")
    parser.add_argument("-s",dest="seed",type = int,help = " random number seed (positive integer)",default=1)
    parser.add_argument("-t",dest="thetaN",type=float,help = "set theta for neutral sites, optional when -r is used, if -t is not specified then thetaN and thetaS are given by -q")
    parser.add_argument("-w",dest="use_watterson_thetaN",action="store_true",default=False,help="use watterson estimator for thetaN")   
    parser.add_argument("-x",dest="gdensitymax",type=float,default=1.0,help = "maximum value of 2Ns density,  default is 1.0,  use with -d")
    parser.add_argument("-z",dest="estimategmax",action="store_true",default=False,help="estimate the maximum of the density of 2Ns")
    parser.add_argument("-slim",dest="simuslim",action="store_true",default=False, help="simulate SFSs with SLiM")
    parser.add_argument("-model",dest="model", type=str, default= "constant", help="The demographic model to simulate SFSs")
    parser.add_argument("-U", dest="mu", type=float, default=1e-6/4, help="Per site mutation rate per generation")
    parser.add_argument("-R", dest="rec", type=float, default=1e-6/4, help="Per site recombination rate per generation")
    parser.add_argument("-N", dest="popSize", default=1000, type=int, help="Population census size")
    parser.add_argument("-L", dest="seqLen", default=10000, type=int, help="Sequence length")
    parser.add_argument("-G", dest="nSeqs", type=int, help="Number of sequences")
    parser.add_argument("-W", dest="parent_dir",default = "results/slim",type = str, help="Path for working directory")
    parser.add_argument("-S", dest="savefile",action="store_true", default=False,help="Save simulated SFS to a file")
    
    args  =  parser.parse_args(sys.argv[1:])  
    if not (args.dobasicPRF or args.doratioPRF):
        parser.error('No action requested, add -p or -r')
    if (args.dobasicPRF and args.doratioPRF):
        parser.error('Multiple actions requested, use either -p or -r,  not both')        
    if (args.dobasicPRF):
        if args.maxi:
            parser.error('-m cannot be used with -p (basic PRF model)')
        if args.thetaN:
            parser.error('-t (neutral theta) cannot be used with -p (basic PRF model)')
        if args.use_watterson_thetaN:
            parser.error('-w cannot be used with -p (basic PRF model)')

    if args.densityof2Ns != None and not (args.densityof2Ns in ['lognormal','gamma']):
        parser.error("-d term {} is not either 'lognormal' or 'gamma'".format(args.densityof2Ns))
    if args.densityof2Ns== None and args.estimategmax:
        parser.error("-z requires that a density function be specified (i.e. -d )")
    if args.densityof2Ns== None and args.gdensitymax != 1.0:
        parser.error("-x requires that a density function be specified (i.e. -d )")        
    args.commandstring = " ".join(sys.argv[1:])
    return args

    # return parser

if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    run(args)