"""
    poisson random field SFS work 
    a module of various functions
    called by check_SFS_LLR_1.py
    see notebook SFS_ratio_modelling.nb
"""
import numpy as np
import  mpmath 
#mpmath.hyp1f1() corresponds to mathematica Hypergeometric1F1 with variables in the same order
#mpmath.coth() corresponds to mathematica Coth(),  hyperbolic cotangent 
import math
from scipy.optimize import minimize 


def Lfw_i(p,i,n,count):
    """
        returns the log likelihood for count of a Fisher-Wright population sample from bin i. 
        uses basic poisson random field math
        if p is a float it is theta,  else an array with theta and g = Ns
    """
    
    if isinstance(p,float): # p is simply a theta value,  no g  
        theta = p
        if theta <= 0:
            return -math.inf
        un = theta*n/(i*(n-i))
        temp = -un + math.log(un)*count - math.lgamma(count+1)
    else:
        theta = p[0]
        if theta <= 0:
            return -math.inf
        g = p[1]
        temph = float(mpmath.hyp1f1(i,n,2*g)+mpmath.hyp1f1(n-i,n,2*g))
        tempc = float(mpmath.coth(g))
        us = theta*(n/(2*i*(n-i)))*(tempc-1)*(2*math.exp(2*g)-temph)
        temp = -us + math.log(us)*count - math.lgamma(count+1)
    return temp 

def negLfw(p,maxi,n,counts):
    """
        counts begins with a 0
        returns the negative of the log of the likelihood for a Fisher Wright sample 
        calls Lfw_i()
    """
    assert(counts[0]==0)
    sum = 0
    for i in range(1,len(counts)):
        sum += Lfw_i(p,i,n,counts[i])
    return -sum 

def Lfw2_i(p,i,n,nval,sval,nog):
    """
        returns the negative of the log likelihood for a pair of counts from bin i. 
        one count is from the selected sfs and the other from the neutral sfs
        the log likelihood assumes fisher wright and is just the sum of the poisson probabilities for the two terms.
        if nog then two regular FW PRFs without selection
    """
    if nog: # then p should be a scalar
        thetaN = p
        thetaS = p
    else:
        thetaN = thetaS = p[0]
        g = p[1]
    if nog or g==0.0:
        un = thetaN*n/(i*(n-i))
        temp = -un + math.log(un)*nval - math.lgamma(nval+1)
        us = thetaS*n/(i*(n-i))
        temp += -un + math.log(un)*sval - math.lgamma(sval+1)        
    else:
        un = thetaN*n/(i*(n-i))
        temph = float(mpmath.hyp1f1(i,n,2*g)+mpmath.hyp1f1(n-i,n,2*g))
        tempc = float(mpmath.coth(g))
        us = thetaS*(n/(2*i*(n-i)))*(tempc-1)*(2*math.exp(2*g)-temph)
        temp = -un + math.log(un)*nval - math.lgamma(nval+1)
        temp += -us + math.log(us)*sval - math.lgamma(sval+1)
    return temp # negative of the log

def negLfw2(p,maxi,n,nvals,svals,nog):
    """
        for two fisher wright folded sfs,  the first can have selection, the second is neutral
        returns the negative of the log of the likelihood for both sfs's sampled from a Fisher Wright population
        p has 2 values, theta and g,  the single theta applies to both sfs's
        counts begins with a 0
        returns the negative of the log of the likelihood for a sample from a Fisher Wright population
        calls Lfw2_i()
        if nog then both sfs's are without selection
    """
    assert(svals[0]==nvals[0]==0)
    sum = 0
    for i in range(1,len(svals)):
        sum += Lfw2_i(p,i,n,svals[i],nvals[i],nog)
    return -sum 

def Li_ratio(p,i,n,z,nog):
    """
        returns the negative of the log of the liklihood for the ratio of term i of the folded distributions (selected over neutral)
        if nog then p contains only theta
        there can be one or two theta values, if two, the first is thetaN and the second is thetaS
    """
    try:
        if z==math.inf or z==0.0:
            return 0.0
        if nog:
            if isinstance(p,float):
                if p <= 0.0:
                    return -math.inf
                thetaN = thetaS = p
            else:
                thetaN = p[0]
                thetaS = p[1]
                if thetaN <= 0.0 or thetaS <= 0.0:
                    return -math.inf
            ux = thetaS*n/(i*(n-i))
        else:
            if len(p) == 2:
                thetaN = p[0]
                thetaS = p[0]
                g = p[1]
            else:
                thetaN = p[0]
                thetaS = p[1]
                g = p[2]
            temph = float(mpmath.hyp1f1(i,n,2*g)+mpmath.hyp1f1(n-i,n,2*g))
            tempc = float(mpmath.coth(g))
            ux = thetaS*(n/(2*i*(n-i)))*(tempc-1)*(2*math.exp(2*g)-temph)
        uy = thetaN*n/(i*(n-i))            
        sigmax = pow(ux,1/2)
        sigmay = pow(uy,1/2)
        deltax = 1/sigmax
        deltay = 1/sigmay
        beta = ux/uy
        rho = sigmay/sigmax
        rho2 = rho*rho
        z2 = z*z
        q = (1+beta*rho2*z)/(deltay*pow(1+rho2*z2,1/2))
        erfq = math.erf(q/pow(2,1/2))
        temp1 = math.log(rho/(math.pi * (1+rho2*z2))) -(rho2*beta*beta + 1)/(2*deltay*deltay)
        try:
            temp2a = pow(math.pi/2,1/2)*q*erfq*math.exp(q*q/2)
            temp2 = math.log(1+temp2a)
            if temp2 == math.inf: # use an approximation for log of (1+x) when x is very large
                temp2 = math.log(1) + 0.5*math.log(math.pi/2) + math.log(q) + math.log(erfq) + q*q/2 
        except: # use an approximation for log of (1+x) when  math.log(1+temp2a) fails
            temp2 = math.log(1) + 0.5*math.log(math.pi/2) + math.log(q) + math.log(erfq) + q*q/2 
        return (temp1+temp2)
    except:
        return -math.inf
    
def negL_ratio(p,maxi,n,zvals,nog):
    """
        returns the negative of the log of the likelihood for a list of ratios of selected over neutral counts 
        calls Li_ratio()
    """
    assert zvals[0]==0 or zvals[0] == math.inf
    sum = 0
    tempa = []
    for i in range(1,len(zvals)):
        temp =  Li_ratio(p,i,n,zvals[i],nog)
        tempa.append([i,temp])
        sum += temp
        # sum += Li_ratio(p,i,n,zvals[i],nog)
    return -sum 


def simsfs(theta,g,n):
    """
        simulates sfs,  folded and unfolded for Fisher Wright under Poisson Random Field
    """
    if g==0:
        sfsexp = [0]+[theta/i for i in range(1,n)]
    else:
        sfsexp = [0]
        for i in range(1,n):
            temph = float(mpmath.hyp1f1(i,n,2*g))
            tempc = float(mpmath.coth(g))
            u = theta*(n/(2*i*(i-n)))*(-1 - tempc + (-1 + tempc)*temph)
            sfsexp.append(u)        
    sfs = [np.random.poisson(expected) for expected in sfsexp]
    sfsfolded = [0]
    for i in range(1,1+n//2):
        # print(i,n-i,sfs[i],sfs[n-i])
        sfsfolded.append(sfs[i]+sfs[n-i] if (n-i != i) else sfs[i])
    return sfs,sfsfolded


def simsfsratio(thetaN,thetaS,g,n):
    """
    simulate the ratio of folded selected SFS to folded neutral SFS
    if a bin of the neutral SFS ends up 0,  the program stops
    """
    
    nsfs,nsfsfolded = simsfs(thetaN,0,n)
    sfs,ssfsfolded = simsfs(thetaS,g,n)
    assert nsfsfolded[0]==ssfsfolded[0]==0
    ratios = [0]
    for i in range(1,len(nsfsfolded)):
        # assert nsfsfolded[i], "nsfs {} is zero".format(i)
        try:
            ratios.append(ssfsfolded[i]/nsfsfolded[i])
        except:
            ratios.append(math.inf)
    return nsfsfolded,ssfsfolded,ratios

def watterson_L(n,counts,folded=True):
    """
        uses waterson's estimator of theta 
    """
    #counts begins with a 0
    # if folded the length is 1+n//2
    wterm = 0.0
    for i in range(1,n):
        wterm += 1/i
    thetaest = sum(counts)/wterm
    if folded==False:
        sumL = 0.0
        for i in range(1,n):
            un = thetaest/i
            sumL += -un + math.log(un)*counts[i] - math.lgamma(counts[i]+1)
        return thetaest,sumL
    else:
        sumL = 0.0
        for i in range(1,len(counts)):# len(counts) should be 1+n//2
            un = thetaest*n/(i*(n-i))
            sumL += -un + math.log(un)*counts[i] - math.lgamma(counts[i]+1)
    return thetaest,sumL

