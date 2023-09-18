#!/bin/bash

# Activate conda environment
# that has a version of parallel install 
#conda activate mss

#runJob () {
#	python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 740 -r -f -d ${1} -l ${2} -slim -model ${2} -G 100 -W results/slim
#
#}
#export -f runJob
#
#parallel -j 8 runJob ::: lognormal gamma ::: constant iexpansion ibottleneck popstructure popstructureN2

#echo "Job done!"
#conda deactivate



# Run manually:
# LOGNORMAL with no Watterson Theta
#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d lognormal -l constant -slim -model constant -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 26 -t 74 -r -f -d lognormal -l ibottleneck -slim -model ibottleneck -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 2582 -t 7408 -r -f -d lognormal -l iexpansion -slim -model iexpansion -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d lognormal -l popstructure -slim -model popstructure -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d lognormal -l popstructureN2 -slim -model popstructureN2 -G 100 -W results/slim

python SFS_estimator_bias_variance.py -k 20 -n 80 -q 25678 -t 73388 -r -f -d lognormal -l OOAgravel2011 -slim -model OOAgravel2011 -G 100 -W results/slim

# LOGNORMAL WITH Watterson Theta
#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d lognormal -l constant -slim -model constant -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 26 -t 74 -r -f -d lognormal -l ibottleneck -slim -model ibottleneck -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 2582 -t 7408 -r -f -d lognormal -l iexpansion -slim -model iexpansion -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d lognormal -l popstructure -slim -model popstructure -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d lognormal -l popstructureN2 -slim -model popstructureN2 -G 100 -W results/slim -w

##[SLOW, needs rescaling]python SFS_estimator_bias_variance.py -k 20 -n 80 -q 25678 -t 73388 -r -f -d lognormal -l OOAgravel2011 -slim -model OOAgravel2011 -G 100 -W results/slim -w




# GAMMA with no Watterson Theta
#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d gamma -l constant -slim -model constant -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 26 -t 74 -r -f -d gamma -l ibottleneck -slim -model ibottleneck -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 2582 -t 7408 -r -f -d gamma -l iexpansion -slim -model iexpansion -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d gamma -l popstructure -slim -model popstructure -G 100 -W results/slim

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d gamma -l popstructureN2 -slim -model popstructureN2 -G 100 -W results/slim

##[SLOW, needs rescaling]python SFS_estimator_bias_variance.py -k 20 -n 80 -q 25678 -t 73388 -r -f -d gamma -l OOAgravel2011 -slim -model OOAgravel2011 -G 100 -W results/slim

# GAMMA WITH Watterson Theta
#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d gamma -l constant -slim -model constant -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 26 -t 74 -r -f -d gamma -l ibottleneck -slim -model ibottleneck -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 2582 -t 7408 -r -f -d gamma -l iexpansion -slim -model iexpansion -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d gamma -l popstructure -slim -model popstructure -G 100 -W results/slim -w

#python SFS_estimator_bias_variance.py -k 20 -n 80 -q 260 -t 741 -r -f -d gamma -l popstructureN2 -slim -model popstructureN2 -G 100 -W results/slim -w

##[SLOW, needs rescaling]python SFS_estimator_bias_variance.py -k 20 -n 80 -q 25678 -t 73388 -r -f -d gamma -l OOAgravel2011 -slim -model OOAgravel2011 -G 100 -W results/slim -w




