#!/bin/bash

# Activate conda environment
# that has a version of parallel install 
#conda activate mss

runJob () {
echo python simulate_SFS_withSLiM.py -r 5 -U 2.5E-07 -R 2.5E-07 -N 1000 -L 10000 -f 10 -d lognormal -a ${1} -n 40 -m ${2} -p results/slim

}
export -f runJob

parallel -j 8 runJob ::: "0.3 0.5" "1.0 0.7" "2.0 1.0" "2.2 1.4" "3.0 1.2" ::: constant iexpansion ibottleneck popstructure popstructureN2

echo "Job done!"
#conda deactivate


#lognormal:  "0.3 0.5" "1.0 0.7" "2.0 1.0" "2.2 1.4" "3.0 1.2"
#gamma: "2. 0.5" "1.6 1.3" "2.5 3.0" "4.0 1.8" "4.5 2.2"

#OOAgravel2011 N=7310