
"""
runs an analysis of estimator bias and variance .  
For a range of true values of the selection coefficient g generate boxplots of estimates 

User must set theta 

when using external data, use -a   
code must be updated to deal with this each each point where
    SFS_functions.simsfs_gdist or SFS_functions.simsfs  is called


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
# import warnings
# warnings.filterwarnings("ignore")

if SFS_functions.SSFconstant_dokuethe:
    optimizemethod="Powell"
else:
    optimizemethod="Nelder-Mead"
optimizemethod="Powell"

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

def run(args):

    SFS_functions.SSFconstant_dokuethe = False #  True

    np.random.seed(args.seed)
    ntrialsperg = args.ntrials
    dobasicPRF = args.dobasicPRF
    doratioPRF = args.doratioPRF
    densityof2Ns = args.densityof2Ns 
    dofolded = args.dofolded
    if args.plotfilelabel != "" and args.plotfilelabel[-1] != "_": # add a spacer if needed
        args.plotfilelabel += "_"  
    n = args.n
    
    if args.thetaN:
        thetaN = args.thetaN
        thetaNresults = []
        estimatetwothetas = True
    else:
        thetaN = args.thetaS
        estimatetwothetas = False
    thetaS = args.thetaS
    foldstring = "folded" if dofolded else "unfolded"
    thetaSresults = []
    if densityof2Ns == 'lognormal':  
        # gvals = [(0.3, 0.5),(0.7, 0.8),(1.0, 1.0),(2.0, 1.4),(5.0, 2.2)]
        gvals = [[0.3, 0.5], [1, 0.7], [2.0, 1.0], [2.2, 1.4], [3, 1.2]]
        # gvals = [gvals[-1]]
        ln1results = []
        ln2results = []
    elif densityof2Ns == 'gamma':
        gvals = [(2,0.5),(1.6,1.3),(2.5,3.0),(4,1.8),(4.5,2.2)]
        ln1results = []
        ln2results = []
    else: #densityof2Ns==None
        gvals = [-10,-5,-2,-1,0,1,2,5,10]
        gresults = []

    SFScomparesultsstrings = []

    if dobasicPRF:
        if estimatetwothetas:
            exit() #not in use as of 8/20/2023
            # if densityof2Ns:
            #     plotfile1name = '{}basicPRF_{}_term_1_k{}_n{}_{}_QS{}_QN{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))
            #     plotfile2name = '{}basicPRF_{}_term_2_k{}_n{}_{}_QS{}_QN{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))
            # else:
            #     plotfilename = '{}basicPRF_g_k{}_n{}_{}_QS{}_QN{}.png'.format(args.plotfilelabel,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))            
            # estimates_filename = '{}basicPRF_{}_ThetaS_ThetaN_k{}_n{}_{}_QS{}_QN{}_results.txt'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))
            

            # savedSFSN = [[]*len(gvals)]
            # savedSFSS = [[]*len(gvals)]
            #     SFScompareheaders = ["{}_{}:".format(g[0],g[1]),"Nsim","Ssim","Nfit","Sfit"]
        else:
            if densityof2Ns:
                plotfile1name = '{}basicPRF_{}_term_1_k{}_n{}_{}_QS{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS))
                plotfile2name = '{}basicPRF_{}_term_2_k{}_n{}_{}_QS{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS))
                
            else:
                plotfilename = '{}basicPRF_g_k{}_n{}_{}_QS{}.png'.format(args.plotfilelabel,ntrialsperg,n,foldstring,round(thetaS))
            estimates_filename = '{}basicPRF_{}_ThetaS_k{}_n{}_{}_QS{}_results.txt'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS))
            
            savedSFSS = [[] for i in range(len(gvals))]

        # estimatesF = open(estimates_filename,"w")
        # estimatesF.write("Program SFS_estimator_bias_variance.py results:\n\nCommand line arguments:\n=======================\n")
        # for key, value in vars(args).items():
        #     estimatesF.write("\t{}: {}\n".format(key,value))
        # estimatesF.write("\nCompare simulated and fitted SFS:\n=================================\n")
        # estimatesF.close()                       

        results = []
        for gi,g in enumerate(gvals):
            thetaSresults.append([])
            if densityof2Ns:
                ln1results.append([])
                ln2results.append([])

                if estimatetwothetas:      
                    SFScompareheaders = ["Params:{} {} Trial#{}:".format(g[0],g[1,gi+1]),"Nsim","Ssim","Nfit","Sfit"]
                else:
                    SFScompareheaders = ["Params:{} {} Trial#{}:".format(g[0],g[1],gi+1),"sim","fit"]
            else:
                gresults.append([])  

                if estimatetwothetas:      
                    SFScompareheaders = ["Param:{} Trial#{}:".format(g,gi+1),"Nsim","Ssim","Nfit","Sfit"]
                else:
                    SFScompareheaders = ["Param:{} Trial#{}:".format(g,gi+1),"sim","fit"]
            for i in range(ntrialsperg):
                if estimatetwothetas:
                    exit()  # not done
                        # use NegL_2SFS_ThetaN_ThetaS_Ns
                else:
                    if densityof2Ns:
                        thetastart = 200
                        if densityof2Ns=='lognormal':
                            sfs,sfsfolded =  SFS_functions.simsfs_gdist(thetaS,n,args.maxi,lognormal = g)  
                            term1start = 1.0
                            term2start = 1.0
                            boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]
                        else :
                            sfs,sfsfolded =  SFS_functions.simsfs_gdist(thetaS,n,args.maxi,gamma = g) 
                            term1start = 2.0
                            term2start = 2.0
                            boundsarray = [(thetastart/10,thetastart*10),(1,10),(0,10)]
                        startarray = [thetastart,term1start,term2start]
                        if densityof2Ns=='lognormal':
                            nlfunc = SFS_functions.NegL_SFS_Theta_Ns_Lognormal
                        elif densityof2Ns == 'gamma':
                            nlfunc = SFS_functions.NegL_SFS_Theta_Ns_Gamma
                        else:
                            print("no distribution function")
                            exit()
                        
                        result = minimize(nlfunc,np.array(startarray),args=(n,dofolded,sfsfolded if dofolded else sfs),method=optimizemethod,bounds=boundsarray)
                        thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                    else:
                        sfs,sfsfolded = SFS_functions.simsfs(thetaS,g,n,args.maxi)
                        thetastart = 100.0
                        gstart = -1.0
                        result = minimize(SFS_functions.NegL_SFS_Theta_Ns,np.array([thetastart,gstart]),args=(n,dofolded,sfsfolded if dofolded else sfs),method=optimizemethod,bounds=[(thetastart/10,thetastart*10),(15*gstart,-15*gstart)])
                        thetaSresults[gi],gresults[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],gresults[gi]], result.x)]                    
                savedSFSS[gi].append(sfsfolded if dofolded else sfs)

            if densityof2Ns:
                if estimatetwothetas:      
                    exit() # not done
                else:
                    if densityof2Ns=='lognormal':
                        sfsfit,sfsfoldedfit =  SFS_functions.simsfs_gdist(thetaSresults[gi][-1],n,args.maxi,lognormal = (ln1results[gi][-1],ln2results[gi][-1]))  
                        
                    if densityof2Ns=='gamma':
                        sfsfit,sfsfoldedfit =  SFS_functions.simsfs_gdist(thetaSresults[gi][-1],n,args.maxi,gamma = (ln1results[gi][-1],ln2results[gi][-1]))  
            else:
                if estimatetwothetas:  
                    exit() #not done    
                else:
                    sfsfit,sfsfoldedfit = SFS_functions.simsfs(thetaSresults[gi][-1],gresults[gi][-1],n,args.maxi)
            SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[sfs,sfsfit]))
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
        for gi,g in enumerate(gvals):
            f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg1 {}\t\tg2 {}\n".format(gi+1,args.thetaS,args.thetaN,g[0],g[1]))
            for k in range(ntrialsperg):
                f.write("\t\t{}\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],ln1results[gi][k],ln2results[gi][k]))
            f.write("\tMean:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi]))))
            f.write("\tStDev:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi]))))
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

    if doratioPRF:
        thetaSresults = []
        savedSFSS = [[] for i in range(len(gvals))]  
        savedSFSN = [[] for i in range(len(gvals))]      
        savedRATIOs = [[] for i in range(len(gvals))]
        if estimatetwothetas:
            if densityof2Ns:
                plotfile1name = '{}ratioPRF_{}_term_1_k{}_n{}_{}_QS{}_QN{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))
                plotfile2name = '{}ratioPRF_{}_term_2_k{}_n{}_{}_QS{}_QN{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))
                estimates_filename = '{}ratioPRF_{}_ThetaS_ThetaN_k{}_n{}_{}_QS{}_QN{}_results.txt'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))

            else:
                plotfilename = '{}ratioPRF_g_k{}_n{}_{}_QS{}_QN{}.png'.format(args.plotfilelabel,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))            
                estimates_filename = '{}ratioPRF_ThetaS_ThetaN_k{}_n{}_{}_QS{}_QN{}_results.txt'.format(args.plotfilelabel,ntrialsperg,n,foldstring,round(thetaS),round(thetaN))
        else:
            if densityof2Ns:
                plotfile1name = '{}ratioPRF_{}_term_1_k{}_n{}_{}_QS{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS))
                plotfile2name = '{}ratioPRF_{}_term_2_k{}_n{}_{}_QS{}.png'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS))
                estimates_filename = '{}ratioPRF_{}_ThetaS_k{}_n{}_{}_QS{}_results.txt'.format(args.plotfilelabel,densityof2Ns,ntrialsperg,n,foldstring,round(thetaS))    
            else:
                plotfilename = '{}ratioPRF_g_k{}_n{}_{}_QS{}.png'.format(args.plotfilelabel,ntrialsperg,n,foldstring,round(thetaS))
                estimates_filename = '{}ratioPRF_ThetaS_k{}_n{}_{}_QS{}_results.txt'.format(args.plotfilelabel,ntrialsperg,n,foldstring,round(thetaS))
        # estimatesF = open(estimates_filename,"w")
        # estimatesF.write("Program SFS_estimator_bias_variance.py results:\n\nCommand line arguments:\n=======================\n")
        # for key, value in vars(args).items():
        #     estimatesF.write("\t{}: {}\n".format(key,value))
        # estimatesF.write("\nCompare simulated and fitted SFS:\n=================================\n")
        # estimatesF.close()               
        results = []
        for gi,g in enumerate(gvals):
            
            thetaSresults.append([])
            if args.thetaN:
                thetaNresults.append([])
            if densityof2Ns:
                ln1results.append([])
                ln2results.append([])
                SFScompareheaders = ["Params:{} {} Trial#{}:".format(g[0],g[1],gi+1),"Nsim","Ssim","Ratiosim","Nfit","Sfit","Ratiofit"]
            else:
                gresults.append([])
                SFScompareheaders = ["Param:{} Trial#{}:".format(g,gi+1),"Nsim","Ssim","Ratiosim","Nfit","Sfit","Ratiofit"]
            for i in range(ntrialsperg):
                thetastart = 200
                if densityof2Ns:
                    if densityof2Ns=='lognormal':
                        nsfs,ssfs,ratios =  SFS_functions.simsfsratio(thetaN,thetaS,n,args.maxi,dofolded,g = g, distribution="lognormal")
                        term1start = 1.0
                        term2start = 1.0
                        if args.use_watterson_thetaN:
                            thetaNest = sum(nsfs)/sum([1/i for i in range(1,n)]) # this should work whether or not the sfs is folded 
                            nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Lognormal_given_thetaN
                            # countratio = sum(nsfs)/sum(ssfs)
                            # nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Gamma_EX

                            boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]          
                            startarray = [thetastart,term1start,term2start]                                       
                        else:                        
                            if estimatetwothetas:
                                boundsarray = [(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]
                            else:
                                boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]
                            nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Lognormal
                    elif densityof2Ns=="gamma":
   
                        nsfs,ssfs,ratios =  SFS_functions.simsfsratio(thetaN,thetaS,n,args.maxi,dofolded,g = g,distribution="gamma") 
                        term1start = 2.0
                        term2start = 2.0
                        if args.use_watterson_thetaN:
                            thetaNest = sum(nsfs)/sum([1/i for i in range(1,n)])  # this should work whether or not the sfs is folded 
                            nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Gamma_given_thetaN
                            # countratio = sum(nsfs)/sum(ssfs)
                            # nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Gamma_EX

                            boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]          
                            startarray = [thetastart,term1start,term2start]                                       
                        else:
                            if estimatetwothetas:
                                boundsarray = [(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]     
                            else:
                                boundsarray = [(thetastart/10,thetastart*10),(0.01,10),(0.01,10)]     
                            nlfunc = SFS_functions.NegL_SFSRATIO_Theta_Gamma
                    if args.use_watterson_thetaN:
                        result = minimize(nlfunc,np.array(startarray),args=(n,thetaN,dofolded,ratios),method=optimizemethod,bounds=boundsarray) 
                        thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                        thetaNresults[gi].append(thetaNest)
                        # result = minimize(nlfunc,np.array(startarray),args=(n,countratio,dofolded,ratios),method=optimizemethod,bounds=boundsarray)       
                        # thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                        # thetaNresults[gi].append(SFS_functions.estimate_thetaN(result.x,n,countratio,dofolded))                        
                    else:
                        if estimatetwothetas:
                            startarray = [thetastart,thetastart,term1start,term2start]           
                        else:
                            startarray = [thetastart,term1start,term2start] 
                        result = minimize(nlfunc,np.array(startarray),args=(n,dofolded,ratios),method=optimizemethod,bounds=boundsarray)  
                        
                        if estimatetwothetas:
                            thetaNresults[gi],thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaNresults[gi],thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]
                        else:
                            thetaSresults[gi],ln1results[gi],ln2results[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],ln1results[gi],ln2results[gi]], result.x)]

                    if estimatetwothetas:
                        fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],n,args.maxi,dofolded,g = [ln1results[gi][-1],ln2results[gi][-1]], distribution=densityof2Ns)
                    else:
                        fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaSresults[gi][-1],thetaSresults[gi][-1],n,args.maxi,dofolded,g = [ln1results[gi][-1],ln2results[gi][-1]], distribution=densityof2Ns)
                    SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[nsfs,ssfs,ratios,fitnsfs,fitssfs,fitratios]))
                else:
                    nsfs,ssfs,ratios =  SFS_functions.simsfsratio(thetaN,thetaS,n,args.maxi,dofolded,g = g)
                    thetastart = 100.0
                    gstart = -1.0
                    if estimatetwothetas:
                        boundsarray = [(thetastart/10,thetastart*10),(thetastart/10,thetastart*10),(15*gstart,-15*gstart)]     
                        startarray = [thetastart,thetastart,gstart]
                    else:
                        boundsarray = [(thetastart/10,thetastart*10),(15*gstart,-15*gstart)]     
                        startarray = [thetastart,gstart]
                    result =  minimize(SFS_functions.NegL_SFSRATIO_Theta_Ns,np.array(startarray),args=(n,dofolded,ratios,False),method=optimizemethod,bounds=boundsarray)
                    if estimatetwothetas:
                        thetaNresults[gi],thetaSresults[gi],gresults[gi] = [lst + [val] for lst, val in zip([thetaNresults[gi],thetaSresults[gi],gresults[gi]], result.x)]
                        fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaNresults[gi][-1],thetaSresults[gi][-1],n,args.maxi,dofolded,g = gresults[gi][-1],distribution=densityof2Ns)
                        SFScomparesultsstrings.append(makeSFScomparisonstring(SFScompareheaders,[nsfs,ssfs,ratios,fitnsfs,fitssfs,fitratios]))
                    else:
                        thetaSresults[gi],gresults[gi] = [lst + [val] for lst, val in zip([thetaSresults[gi],gresults[gi]], result.x)]     
                        fitnsfs,fitssfs,fitratios =  SFS_functions.simsfsratio(thetaSresults[gi][-1],thetaSresults[gi][-1],n,args.maxi,dofolded,g = gresults[gi][-1],distribution=densityof2Ns)
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
            f.write("\tSet {} Values:\t\tThetaS {}\t\tThetaN {}\t\tg1 {}\t\tg2 {}\n".format(gi+1,args.thetaS,args.thetaN,g[0],g[1]))
            for k in range(ntrialsperg):
                f.write("\t\t{}\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(k+1,thetaSresults[gi][k],thetaNresults[gi][k],ln1results[gi][k],ln2results[gi][k]))
            f.write("\tMean:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.mean(np.array(thetaSresults[gi])),np.mean(np.array(thetaNresults[gi])),np.mean(np.array(ln1results[gi])),np.mean(np.array(ln2results[gi]))))
            f.write("\tStDev:\t\t{:.1f}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\n".format(np.std(np.array(thetaSresults[gi])),np.std(np.array(thetaNresults[gi])),np.std(np.array(ln1results[gi])),np.std(np.array(ln2results[gi]))))
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
        # f.write("\n")
        # for gi,g in enumerate(gvals):
        #     f.write("Theta estimates for Ns (g) value {}:\n".format(g))
        #     if estimatetwothetas:
        #         f.write("\tthetaS mean: {:.4f}  stdev {:.4f} \n".format(np.mean(np.array(thetaSresults[gi])),np.std(np.array(thetaSresults[gi]))) )
        #         f.write("\tthetaN mean: {:.4f}  stdev {:.4f} \n".format(np.mean(np.array(thetaNresults[gi])),np.std(np.array(thetaNresults[gi]))) )
        #     else:
        #         f.write("\ttheta mean: {:.4f}  stdev {:.4f}\n".format(np.mean(np.array(thetaSresults[gi])),np.std(np.array(thetaSresults[gi]))) )
        f.close()            


def createparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a",dest="datafilename",default=None,type=str,help="optional data file with simulated SFSs")    
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
    parser.add_argument("-t",dest="thetaN",type=float,help = "theta for neutral sites, optional, if -t is not specified then thetaN and thetaS are given by -q")
    parser.add_argument("-w",dest="use_watterson_thetaN",action="store_true",default=False,help="use watterson estimator for thetaN")   

    return parser

if __name__ == '__main__':
    """

    """
    parser = createparser()
    args  =  parser.parse_args(sys.argv[1:])      
    run(args)