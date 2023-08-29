import os
import sys
import argparse
import csv
import time
import shlex, subprocess
import numpy as np


#---FUNCTION DEFINITION---#

# Function to run SLiM as a external process
#def runSlim (simulation, mu, rec, popSize, seqLen, ns, sampleSize, ngenes1, ngenes2, model, path): # REMOVE?
def runSlim (simulation, mu, rec, popSize, seqLen, nsdist, distpars, ns, sampleSize, model, path):
    
    # # Model dictionary
    # avail_models = {"constant": "Constant size model",
    #                 "iexpansion": "Instantaeous expansion model",
    #                 "ibottlneck": "Instantaneous bottleneck model",
    #                 "popstructure": "Constant size, population structure model",
    #                 "popstructure2N": "N/2 population size after split, population structure model",
    #                 "Gravel2011OOA": "Gravel et al. 2011 Out-of-Africa Human demography model"}
    
    # if model in avail_models:
    #         print("Ok, " + avail_models.get(model) + " is available!")
    # else:
    #     print("Sorry, model " + model + " does not exists!")
    #     sys.exit()

    # Path to SLiM
    slim = "/usr/local/bin/slim"

    # Sample a seed every function call
    seed = str(int(np.random.uniform(low=100000000, high=900000000)))
    #seed = str(123456) # use for debugging only
    
    # Run SLiM as a python subprocess
    run = subprocess.run([slim, "-s", seed, "-d", ("simu="+str(simulation)), "-d", ("MU="+str(mu)), "-d", ("R="+str(rec)),
                          "-d", ("N="+str(popSize)), "-d", ("L="+str(seqLen)), "-d", ("Ns="+str(ns)), 
                          "-d", ("dist="+"'"+nsdist+"'"), "-d", ("distpars="+"'"+distpars+"'"), "-d", ("n="+str(sampleSize)),
                          "-d", ("model="+"'"+model+"'"), "-d", ("outDir="+"'"+path+"'"), 
                          ("models/" + model + ".slim")], capture_output=True)
    
    return seed, run

# Function to combine individuals SFSs in a file (each one in a row)
def combineSFSs(sampleSize, filename, path):
    
    csfs = [0]*(sampleSize)
    file = (path + "/" + filename + ".txt")
    for line in open(file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                sfs = line.strip().split()
                for i in range(len(csfs)): 
                    csfs[i] += int(sfs[i])
            except:
                print(sfs)
                pass
    return csfs

# Define the command line arguments
def parseargs():
    parser = argparse.ArgumentParser("python run_slim.py",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", help="number of simulations",
                        dest="nsimulations", default = 3, type=int)
    parser.add_argument("-U", help="Per site mutation rate per generation",
                        dest="mu", default=1e-6/4, type=float)
    parser.add_argument("-R", help="Per site recombination rate per generation",
                        dest="rec", default=1e-6/4, type=float)
    parser.add_argument("-N", help="Population census size",
                        dest="popSize", default=1000, type=int)
    parser.add_argument("-L", help="Sequence length",
                        dest="seqLen", default=10000, type=int)
    parser.add_argument("-f", help="Number of sequences",
                        dest="nSeqs", default=5, type=int)
    parser.add_argument("-d", help="Set a distribution for Ns",
                        dest="nsdist", default="fixed", type=str)
    parser.add_argument("-g", help="Non-synonymous population selection coefficient, 2Ns (Slim uses 1-(2Ns/2N))",
                        dest="ns",default=0.0, type=float)
    parser.add_argument("-a", help="Set the parameters of the chosen distribution",
                        dest="nsdistargs", nargs= "+", default = [0.0, 0.0], type=float)
    parser.add_argument("-n", help="Sample size",
                        dest="sampleSize", default=40, type=int)
    parser.add_argument("-m", help="Model",
                        dest="model", default="constant", #required = True,
                        type = str)
    parser.add_argument("-p", help="Parent directory path for results files",
                        dest="parent_dir",default = "results/prfratio",type = str)
    return parser


#argv = "-r 3 -U 1e-6/4 -R 1e-6/4 -N 1000 -L 10000 -f 5 -d lognormal -a 10.0 2.0 -n 40 -m constant -d results/prfration"
# sys.argv = argv.split()
# print(sys.argv)
parser = parseargs()
args = parser.parse_args('-d gamma -a 10.0 1.0'.split())


# python src/run_slim.py -r 3 -U 2.5E-07 -R 2.5E-07 -N 1000 -L 10000 -f 5 -d fixed -g 2 -n 40 -m constant -p results/prfratio

#---PROGRAM DEFINITION---#
# Pipeline to run multiple sequences (or genes, chrms) for each of the multiples simulations (or replicates)
# It is a wrapper for SLiM code and it allows run different models with different parameters, especially Ns
# This generates 1 file with 200 SFS per run (100 runs x 200 simulations of SFS)
def main(argv):
    starttime = time.time()
    parser = parseargs()
    if argv[-1] =='':
        argv = argv[0:-1]
    args = parser.parse_args(argv)
    
    # CMD arguments 
    # nsimulations = 3
    # mu = 1e-6/4
    # rec = 1e-6/4
    # popSize = 1000
    # seqLen = 10000
    # ns = -0.02
    # nsdist = "fixed"
    # nsdistargs = 0.0
    # sampleSize = 40
    # model = "popstructureN2"
    # nSeqs = 5
    # parent_dir = "results/prfratio"
    
    nsimulations = args.nsimulations
    mu = args.mu
    rec = args.rec
    popSize = args.popSize
    seqLen = args.seqLen
    ns = args.ns
    nsdist = args.nsdist
    nsdistargs = args.nsdistargs
    sampleSize = args.sampleSize
    model = args.model
    nSeqs = args.nSeqs
    parent_dir = args.parent_dir

    # Constant value parameters:
    # Intron length and total intron size
    intronL = 810
    intron_totalL = (8*intronL) + 928 # Hard-coded
    
    # Exon length and total exon size
    exonL = 324
    exon_totalL = 8*exonL

    # SFS format
    sfs_format = "folded"
    
    # First, check if specified SLiM model exists!
    # Model dictionary
    avail_models = {"constant": "Constant size model",
                    "iexpansion": "Instantaeous expansion model",
                    "ibottlneck": "Instantaneous bottleneck model",
                    "popstructure": "Constant size, population structure model",
                    "popstructure2N": "N/2 population size after split, population structure model",
                    "Gravel2011OOA": "Gravel et al. 2011 Out-of-Africa Human demography model"}
    
    if model in avail_models:
            print("Ok, " + avail_models.get(model) + " is available!")
    else:
        print("Sorry, model " + model + " does not exists!")
        sys.exit()

    # Define thetas given the chosen models
    # Neutral theta
    thetaNeutral = (4*popSize*mu)*intron_totalL*nSeqs
    if model == "ibottleneck":
        thetaNeutral = thetaNeutral/10
    if model == "iexpansion":
        thetaNeutral = thetaNeutral * 10
    
    # Selected theta
    thetaSelected = (4*popSize*mu)*exon_totalL*nSeqs
    if model == "ibottleneck":
        thetaSelected = thetaSelected /10
    if model == "iexpansion":
        thetaSelected  = thetaSelected * 10

    # Second, check if the distribution exists
    # and if parameters are correct!
    avail_nsdist = {"fixed": "Fixed Ns",
                    "lognormal": "Ns sample from a lognormal distribution",
                    "gamma": "Ns sample from a gamma distribution"}
    
    if nsdist in avail_nsdist:
         print("Ok, " + avail_nsdist.get(nsdist) + " exists")
         if nsdist != "fixed":
              if sum(nsdistargs) == 0.0:
                   print("You selected a different distribution, but did not changed the default values. Are you sure you want to proceed with no selection?")
                   sys.exit()

              else:
                   # Combine output directory
                   path = os.path.join(parent_dir, model, nsdist, (str(nsdistargs[0]) + "-" + str(nsdistargs[1]))) 

                   # Define base name given the model and Ns distribution
                   basename = (model + "_" + nsdist + "_" + (str(nsdistargs[0]) + "-" + str(nsdistargs[1])))

                   # Prapare the headers for each model/Ns simulation
                   header_neutral = "# 4Nmu(intron total length)={t} distribution={nsdist} dist_pars={nsdistargs} n={n} Neutral {sfs} SFS".format(t=thetaNeutral, nsdist=nsdist, nsdistargs=nsdistargs, n=sampleSize, sfs=sfs_format)
                   header_selected = "# 4Nmu(exon total length)={t} distribution={nsdist} dist_pars={nsdistargs} n={n} Selected {sfs} SFS".format(t=thetaSelected,nsdist=nsdist, nsdistargs=nsdistargs, n=sampleSize, sfs=sfs_format)

         else:
              # Combine output directory
              path = os.path.join(parent_dir, model, nsdist, str(ns)) 

              # Define base name given the model and Ns distribution
              basename = (model + "_" + nsdist + "_" + str(ns))

              # Prapare the headers for each model/Ns simulation
              header_neutral = "# 4Nmu(intron total length)={t} distribution={nsdist} Ns={ns} n={n} Neutral {sfs} SFS".format(t=thetaNeutral, nsdist=nsdist, ns=ns, n=sampleSize, sfs=sfs_format)
              header_selected = "# 4Nmu(exon total length)={t} distribution={nsdist} Ns={ns}  n={n} Selected {sfs} SFS".format(t=thetaSelected, nsdist=nsdist, ns=ns, n=sampleSize, sfs=sfs_format)

    else:
        print("Sorry, Ns distribution " + nsdist + " does not exists!")
        sys.exit()

    # Check if output already exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Prepare the output for the neutral mutations CFSFS
    fout_neutral = (path + "/" + basename + "_csfs_neutral.txt")
    with open(fout_neutral, 'a') as f:
            f.write(header_neutral + "\n")
            f.write("\n")
            f.close

    # Prepare the output for the selected mutations CFSFS
    fout_selected = (path + "/" + basename + "_csfs_selected.txt")
    with open(fout_selected, 'a') as f:
            f.write(header_selected + "\n")
            f.write("\n")
            f.close

    # Run many replicates
    for i in range(0,nsimulations):
        csfs_neutral = []
        csfs_selected = []

        simulation = i
        # Run many SLiM sequences for one simulation
        fg = open((path + "/" + basename + "_" + str(simulation) + "_gs.txt"), 'w')
        fs = open((path + "/" + basename + "_" + str(simulation) + "_seeds.txt"), 'w')
        j = 0
        while j < nSeqs:
            # Select Ns (or s) from a distribution
            if nsdist == "lognormal":
                    ns = -1 * np.random.lognormal(nsdistargs[0], nsdistargs[1])
                    print(ns)
            if nsdist == "gamma":
                    ## CODE HERE
                    ns = -1 * np.random.gamma(nsdistargs[0], nsdistargs[1])
                    print(ns)
            fg.write(str(ns) + "\n")  
            seed, run = runSlim(simulation = simulation, 
                                mu = mu, rec = rec, popSize = popSize, seqLen = seqLen,
                                ns=ns, nsdist=nsdist, distpars=(str(nsdistargs[0]) + "-" + str(nsdistargs[1])), 
                                sampleSize = sampleSize, model = model, path = path)
            fs.write(seed + "\n")
            #run.stdout
            #run.stderr
            j += 1
        fg.close()
        fs.close()

        # Save the combined FSFS in a list to use csv.writer.writerows function (but reset the list each time)
        # Neutral mutations
        csfs_neutral.append(combineSFSs(sampleSize = sampleSize, filename = (basename + "_" + str(simulation) + "_sfs_neutral"), path = path))
    
        with open(fout_neutral, 'a') as f:
            wr = csv.writer(f, delimiter=" ")
            wr.writerows(csfs_neutral)
            f.write("\n")
            f.close
        
        # Selected mutations
        csfs_selected.append(combineSFSs(sampleSize = sampleSize, filename = (basename + "_" + str(simulation) + "_sfs_selected"), path = path))

        with open(fout_selected, 'a') as f:
            wr = csv.writer(f, delimiter=" ")
            wr.writerows(csfs_selected)
            f.write("\n")
            f.close

if __name__ == "__main__":

    if len(sys.argv) < 2:
        main(['-h'])
    else:
        main(sys.argv[1:])

    








