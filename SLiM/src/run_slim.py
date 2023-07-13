import os
import shlex, subprocess
import numpy as np
import argparse
import csv
import time
import sys
import argparse


#---FUNCTION DEFINITION---#

# Function to run SLiM as a external process
def runSlim (simulation, mu, rec, popSize, seqLen, ns, sampleSize, model, path):
    
    # Path to SLiM
    slim = "/usr/local/bin/slim"

    # Sample a seed every function call
    seed = str(int(np.random.uniform(low=100000000, high=900000000)))
    #seed = str(123456) # use for debugging only
    
    # Run SLiM as a python subprocess
    run = subprocess.run([slim, "-s", seed, "-d", ("simu="+str(simulation)), "-d", ("MU="+str(mu)), "-d", ("R="+str(rec)),
                          "-d", ("N="+str(popSize)), "-d", ("L="+str(seqLen)), "-d", ("Ns="+str(ns)), "-d", ("n="+str(sampleSize)),
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
    parser.add_argument("-g", help="Non-synonymous population selection coefficient, 2Ns (Slim uses 1-(2Ns/2N))",
                        dest="ns",default=-0.02, type=float)
    parser.add_argument("-n", help="Sample size",
                        dest="sampleSize", default=40, type=int)
    parser.add_argument("-m", help="Model",
                        dest="model",required = True,type = str)
    parser.add_argument("-d", help="Parnt directory path for results files",
                        dest="parent_dir",default = "results/prfratio",type = str)
    return parser


#argv = "-r 3 -U 1e-6/4 -R 1e-6/4 -N 1000 -L 10000 -f 5 -g -0.02 -n 40 -m constant -d results/prfration"
#sys.argv = argv.split()
# print(sys.argv)


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
    
    #CMD arguments 
    nsimulations = args.nsimulations
    mu = args.mu
    rec = args.rec
    popSize = args.popSize
    seqLen = args.seqLen
    ns = args.ns
    sampleSize = args.sampleSize
    model = args.model
    nSeqs = args.nSeqs
    parent_dir = args.parent_dir
    
    # Other arguments
    thetaL = 4*popSize*mu*seqLen 
    sfs_format = "folded"
    
    # Combine output directory
    path = os.path.join(parent_dir, model, str(ns)) 

    # Check if output already exists
    if not os.path.exists(path):
        os.makedirs(path)

    basename = (model + "_" + str(ns))

    # Prapare the headers for each model/Ns simulation
    header_neutral = "# 4NmuL={t} Ns={ns} n={n} Neutral {sfs} SFS".format(t=thetaL, ns=ns, n=sampleSize, sfs=sfs_format)
    header_selected = "# 4NmuL={t} Ns={ns} n={n} Selected {sfs} SFS".format(t=thetaL, ns=ns, n=sampleSize, sfs=sfs_format)

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
        fs = open((path + "/" + basename + "_" + str(simulation) + "_seeds.txt"), 'w')
        j = 0
        while j < nSeqs:
            seed, run = runSlim(simulation = simulation, 
                                mu = mu, rec = rec, popSize = popSize, seqLen = seqLen,
                                ns=ns, sampleSize = sampleSize, model = model, path = path)
            fs.write(seed + "\n")
            j += 1
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
    
    
    
    





    








