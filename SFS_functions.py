"""
    poisson random field SFS work 
    a module of various functions
    see notebook SFS_ratio_modelling.nb
"""
import numpy as np
import  mpmath 
import math
from scipy.optimize import minimize 
import scipy
from scipy.integrate import quad as quad 

# as of 8/23/2023 Kuethe method does not work as well as that of  Díaz-Francés, E. and F. J. Rubio
# if False then use the ratio probability function of Díaz-Francés, E. and F. J. Rubio, which works better 
SSFconstant_dokuethe = False 

import warnings
# warnings.simplefilter('error')
# Configure warnings to treat RuntimeWarning as an exception
warnings.simplefilter("error", RuntimeWarning)
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    print("XXX",message,"\n", category,"\n",filename,"\n",lineno)
    raise category(message)
# Register the custom warning filter
warnings.showwarning = custom_warning_handler

#function naming: primary elements if main likelihood function names 
#   NegL if negative of likelihood
#   L    meaning Likelihood 
#   SFS  or SFSRATIO 
#   Theta  Ns  gamma lognormal   : things being estimated
#   bin_i  if for just a particular bin 

#constants
sqrt2 =pow(2,1/2)
pi2r = np.sqrt(2 * np.pi)
pidiv2r = np.sqrt(np.pi/2)

## an array of 2Ns values spanning a useful range,  used for numerically integrating over the density of 2Ns, assumes max is 1.0 
g_xvals = np.concatenate([np.array([-1000,-500,-200,-100,-50,-40,-30,-20]),np.linspace(-19,-1.1,50),np.linspace(-1,0.99999,40)])

def reset_g_xvals(gdensitymax):
    """
        reset g_xvals for a new max value
    """
    global g_xvals
    g_xvals = np.concatenate([np.array([-1000,-500,-200,-100,-50,-40,-30,-20]),np.linspace(-19,-1.1,50),np.linspace(-1,gdensitymax - 1e-5,40)])


def coth(x):
    if abs(x) > 15: # save a bit of time 
        return -1.0 if x < 0 else 1.0
    else:
        return np.cosh(x)/np.sinh(x) 

def logprobratio(alpha,beta,z):
    """
        returns the log of the probability of a ratio z of two normal densities when for each normal density the variance equals the mean 
        is called from other functions,  where alpha, beta and the ratio z have been calculated for a particular frequency bin

        two versions 
        Kuethe DO, Caprihan A, Gach HM, Lowe IJ, Fukushima E. 2000. Imaging obstructed ventilation with NMR using inert fluorinated gases. Journal of applied physiology 88:2279-2286.
        the gamma in the paper goes away because it is equal to 1/alpha^(1/2) when we assume the normal distributions have mean equal to variance 

        Díaz-Francés, E. and F. J. Rubio paper expression (1)

        The functions are similar, but the Díaz-Francés, E. and F. J. Rubio  works much better overall,  e.g. LLRtest check and ROC curves are much better 
        However the Díaz-Francés and Rubio function gives -inf often enough to cause problems for the optimizer,  so we set final probability p = max(p,1e-50)

    """
    minp = 1e-50
    if SSFconstant_dokuethe:
        try:
            alpha2 = alpha*alpha
            alphaR = math.sqrt(alpha)
            z2 = z*z 
            beta2 = beta*beta
            z2b1term = 1+z2/alpha
            xtemp = -(alpha+1)/(2*beta2)
            if xtemp < -709:
                temp1 = 0.0
            else:
                temp1num = math.exp(xtemp)
                temp1denom = math.pi*alphaR*z2b1term
                temp1 = temp1num/temp1denom
            xtemp = -pow(z-alpha,2)/(2*alpha2*beta2*z2b1term)
            if xtemp < -709: # exp of this is zero and so the entire 2nd term of the probability will be zero 
                p = temp1
            else:
                temp2num1 = (1+z)*math.exp(xtemp)
                temp2num2 = math.erf((z+1)/(sqrt2*beta*math.sqrt(z2b1term)))
                temp2denom = pi2r * alphaR*beta*pow(z2b1term,1.5)
                p = temp1 + (temp2num1*temp2num2)/temp2denom
        except RuntimeWarning as rw:
            print(f"Caught a RuntimeWarning: {rw}")
        except Exception as e:
            print(f"Caught an exception: {e}")        
        if p > 0:
            return math.log(p)
        else:
            return -math.inf
    else: # Díaz-Francés, E. and F. J. Rubio 
        try:
            # rename beta and alpha to match variables in the paper by  Díaz-Francés, E. and F. J. Rubio 
            delta = beta
            beta = alpha
            z2 = z*z
            delta2 = delta*delta
            z1 = 1+z
            z2b1 = 1+z2/beta
            z2b1r = math.sqrt(z2b1)
            z2boverb = (z2+beta)/beta
            betar = math.sqrt(beta)
            xtemp1 = -(1+beta)/(2*delta2)
            if xtemp1 < -709:
                temp1 = 0.0
            else:
                temp1denom = (math.pi*z2b1*betar)
                temp1 = math.exp(xtemp1)/temp1denom
            xtemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            if xtemp2 < -709:
                p=temp1
            else:
                temp2num = math.exp(xtemp2)* z1 * math.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))) 
                temp2denom = pi2r *  betar * delta*pow(z2boverb,1.5)
                temp2 = temp2num/temp2denom 
                p = temp1 + temp2 
        except RuntimeWarning as rw:
            print(f"Caught a RuntimeWarning: {rw}")
        except Exception as e:
            print(f"Caught an exception: {e}")        
        if p < minp:
            p=minp
        return math.log(p)
        # if p > 0:
        #     return math.log(p)
        # else:
        #     return -math.inf        

def NegL_SFS_Theta_Ns(p,n,dofolded,counts): 
    """
        for fisher wright poisson random field model,  with with selection or without
        if p is a float,  then the only parameter is theta and there is no selection
        else p is a list (2 elements) with theta and Ns values 
        counts begins with a 0
        returns the negative of the log of the likelihood for a Fisher Wright sample 
    """
    def L_SFS_Theta_Ns_bin_i(p,i,n,dofolded,count): 
        if isinstance(p,float): # p is simply a theta value,  no g  
            theta = p
            if theta <= 0:
                return -math.inf
            un = theta*n/(i*(n-i)) if dofolded else theta/i
            temp = -un + math.log(un)*count - math.lgamma(count+1)
        else:
            theta = p[0]
            if theta <= 0:
                return -math.inf
            g = p[1]
            us = theta * prf_selection_weight(n,i,g,dofolded)
            temp = -us + math.log(us)*count - math.lgamma(count+1)
        return temp     
    assert(counts[0]==0)
    sum = 0
    for i in range(1,len(counts)):
        sum += L_SFS_Theta_Ns_bin_i(p,i,n,dofolded,counts[i])
    return -sum 

def prf_selection_weight(n,i,g,dofolded):
    """
        Poisson random field selection weight for g=2Ns for bin i  (folded or unfolded)
        this is the function you get when you integrate the product of two terms:
             (1) WF term for selection    (1 - E^(-2 2 N s(1 - q)))/((1 - E^(-2 2 N s)) q(1 - q))  
             (2) bionomial sampling formula for i copies,  given allele frequency q 
        over the range of allele frequencies 
    """
    tempc = coth(g)
    if tempc==1:
        if dofolded:
            us = 2*(n/(i*(n-i)))
        else:
            us = (n/(i*(n-i)))
    if dofolded:
        temph = scipy.special.hyp1f1(i,n,2*g)+scipy.special.hyp1f1(n-i,n,2*g)
        us = (n/(2*i*(n-i)))*(2 +2*tempc - (tempc-1)*temph)
    else:
        temph = scipy.special.hyp1f1(i,n,2*g)
        us = (n/(2*i*(n-i)))*(1 + tempc - (tempc-1)* temph)       
    return us

def prfdensityfunction(g,n,i,arg1,arg2,gdm,densityof2Ns,dofolded):
    """
    returns the product of poisson random field weight for a given level of selection (g) and a probability density for g 
    used for integrating over g 
    
    """
    us = prf_selection_weight(n,i,g,dofolded)
    if densityof2Ns=="lognormal":   
        mean = arg1
        std_dev = arg2
        x = float(gdm-g)
        p = (1 / (x * std_dev * pi2r)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))
    if densityof2Ns=="gamma":
        alpha = arg1
        beta = arg2
        x = float(gdm-g)
        p = ((x**(alpha-1))*np.exp(-(x)/beta))/(math.gamma(alpha)*(beta**alpha))
    if p*us < 0.0:
        return 0.0
        # print("prf density problem")
        # exit()
    if np.isnan(p):
        return(0.0)
    return p*us
           
def NegL_SFS_ThetaS_Ns_density(p,gdm,n,dofolded,densityof2Ns,counts):
    """
        likelihood for fisher wright poisson random field model 
        uses a density for 2Ns
        p is an array with 2 elements, alpha and beta if gamma density, and mean and stdev if lognormal,  gdm is  the shift term 
    """
    sum = 0
    thetaS = p[0]
    term1 = p[1]
    term2 = p[2]
    if thetaS <= 0: #theta can't be negative,  shape parameter can't be <= 1 else mode is at 0 
        return -math.inf
    k = 100
    for i in range(1,len(counts)):
        density_values = np.array([prfdensityfunction(x,n,i,term1,term2,gdm,densityof2Ns,dofolded) for x in g_xvals])
        us=float(thetaS*np.trapz(density_values,g_xvals))
        sum += -us + math.log(us)*counts[i] - math.lgamma(counts[i]+1)        
    return -sum    
 


def NegL_SFSRATIO_Theta_Ns(p,n,dofolded,zvals,nog):
    """
        returns the negative of the log of the likelihood for a list of ratios of selected over neutral counts 
        the selected counts are generated using a single Ns value, not a distribution
    """
    def L_SFSRATIO_Theta_Ns_bin_i(p,i,n,dofolded,z,nog):
        """
            returns the negative of the log of the likelihood for the ratio of term i of the folded distributions (selected over neutral)
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
                ux = thetaS*n/(i*(n-i)) if dofolded else thetaS/i
            else:
                if len(p) == 2:
                    thetaN = p[0]
                    thetaS = p[0]
                    g = p[1]
                else:
                    thetaN = p[0]
                    thetaS = p[1]
                    g = p[2]
                ux = thetaS*prf_selection_weight(n,i,g,dofolded)
            
            uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i     
            alpha = ux/uy
            sigmay = math.sqrt(uy)
            beta = 1/sigmay
            return logprobratio(alpha,beta,z)

        except:
            return -math.inf    
    assert zvals[0]==0 or zvals[0] == math.inf
    sum = 0
    for i in range(1,len(zvals)):
        temp =  L_SFSRATIO_Theta_Ns_bin_i(p,i,n,dofolded,zvals[i],nog)
        sum += temp
        if sum==-math.inf:
            break        
    return -sum 

def NegL_SFSRATIO_Theta_Ns_given_thetaN(p,n,thetaN,dofolded,zvals,nog):
    """
        uses thetaN estimated directly from snp count using watterson's estimator
        returns the negative of the log of the likelihood for a list of ratios of selected over neutral counts 
        the selected counts are generated using a single Ns value, not a distribution
        calls L_SFSRATIO_Theta_Ns_bin_i()
    """
    def L_SFSRATIO_Theta_Ns_bin_i(p,i,n,dofolded,z,nog):
        """
            returns the negative of the log of the likelihood for the ratio of term i of the folded distributions (selected over neutral)
            if nog then p contains only theta
            there can be one or two theta values, if two, the first is thetaN and the second is thetaS
        """
        try:
            if z==math.inf or z==0.0:
                return 0.0
            if nog:
                thetaN = p[0]
                thetaS = p[1]
                if thetaN <= 0.0 or thetaS <= 0.0:
                    return -math.inf
                ux = thetaS*n/(i*(n-i)) if dofolded else thetaS/i
            else:
                thetaN = p[0]
                thetaS = p[1]
                g = p[2]
                ux = thetaS*prf_selection_weight(n,i,g,dofolded)
         
            uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i     
            alpha = ux/uy
            sigmay = math.sqrt(uy)
            beta = 1/sigmay
            return logprobratio(alpha,beta,z)

        except:
            return -math.inf    
    assert zvals[0]==0 or zvals[0] == math.inf
    sum = 0
    p = list(p)
    p.insert(0,thetaN)
    for i in range(1,len(zvals)):
        temp =  L_SFSRATIO_Theta_Ns_bin_i(p,i,n,dofolded,zvals[i],nog)
        sum += temp
        if sum==-math.inf:
            break        
    return -sum 

def NegL_SFSRATIO_Theta_Nsdensity(p,gdm,n,dofolded,densityof2Ns,zvals): 
    """
        returns the negative of the likelihood for the SFS ratio when a lognormal or gamma distribution is used
        
    """    
    def L_SFSRATIO_Theta_Nsdensity_bin_i(p,i,gdm,n,dofolded,densityof2Ns,z):

        try:
            if z==math.inf or z==0.0:
                return 0.0
            if len(p) == 3: # just a single theta for for selected and neutral 
                thetaN = p[0]
                thetaS = p[0]
                g = (p[1],p[2])
            else:
                thetaN = p[0]
                thetaS = p[1]
                g = (p[2],p[3])
            density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],gdm,densityof2Ns,dofolded) for x in g_xvals])
            ux=float(thetaS*np.trapz(density_values,g_xvals))
            if ux<= 0.0:
                return -math.inf
               
            uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i    
            alpha = ux/uy
            sigmay = math.sqrt(uy)
            beta = 1/sigmay
            return logprobratio(alpha,beta,z)             
        except Exception as mistake:
            print(mistake)
            return -math.inf    
    assert zvals[0]==0 or zvals[0] == math.inf
    sum = 0
    for i in range(1,len(zvals)):
        temp =  L_SFSRATIO_Theta_Nsdensity_bin_i(p,i,gdm,n,dofolded,densityof2Ns,zvals[i])
        sum += temp
        if sum==-math.inf:
            break        
    return -sum         


    
# def NegL_SFSRATIO_Theta_Lognormal(p,gdm,n,dofolded,zvals): 
#     """
#         like L_SFSRATIO_Theta_Ns() but assumes numerator is the sum of a bunch of terms from a discretized lognormal 
#         returns the negative of the log of the likelihood for the ratio of term i of the folded distributions (selected over neutral)
#         there can be one or two theta values, if two, the first is thetaN and the second is thetaS
#     """    
#     def L_SFSRATIO_Theta_Lognormal_bin_i(p,i,gdm,n,dofolded,z):

#         try:
#             if z==math.inf or z==0.0:
#                 return 0.0
#             if len(p) == 3: # just a single theta for for selected and neutral 
#                 thetaN = p[0]
#                 thetaS = p[0]
#                 g = (p[1],p[2])
#             else:
#                 thetaN = p[0]
#                 thetaS = p[1]
#                 g = (p[2],p[3])
#             density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],gdm,"lognormal",dofolded) for x in g_xvals])
#             ux=float(thetaS*np.trapz(density_values,g_xvals))
#             if ux<= 0.0:
#                 return -math.inf
               
#             uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i    
#             alpha = ux/uy
#             sigmay = math.sqrt(uy)
#             beta = 1/sigmay
#             return logprobratio(alpha,beta,z)             
#         except Exception as mistake:
#             print(mistake)
#             return -math.inf    
#     assert zvals[0]==0 or zvals[0] == math.inf
#     sum = 0
#     for i in range(1,len(zvals)):
#         temp =  L_SFSRATIO_Theta_Lognormal_bin_i(p,i,gdm,n,dofolded,zvals[i])
#         sum += temp
#         if sum==-math.inf:
#             break        
#     return -sum         


    
# def NegL_SFSRATIO_Theta_Gamma(p,gdm,n,dofolded,zvals): 
#     """
#         like L_SFSRATIO_Theta_Ns but assumes numerator is the sum of a bunch of terms from a discretized lognormal 
#         returns the negative of the log of the likelihood for the ratio of term i of the folded distributions (selected over neutral)
#         there can be one or two theta values, if two, the first is thetaN and the second is thetaS
#     """    
#     def L_SFSRATIO_Theta_Gamma_bin_i(p,i,gdm,n,dofolded,z): 

#         try:
#             if z==math.inf or z==0.0:
#                 return 0.0
#             if len(p) == 3: # just a single theta for for selected and neutral 
#                 thetaN = p[0]
#                 thetaS = p[0]
#                 g = (p[1],p[2])
#             else:
#                 thetaN = p[0]
#                 thetaS = p[1]
#                 g = (p[2],p[3])
#             density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],gdm,"gamma",dofolded) for x in g_xvals])
#             ux=float(thetaS*np.trapz(density_values,g_xvals))
 
#             if ux<= 0.0:
#                 return -math.inf    

#             uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i   
#             alpha = ux/uy
#             sigmay = math.sqrt(uy)
#             beta = 1/sigmay
#             return logprobratio(alpha,beta,z)             
#         except:
#             return -math.inf    
#     assert zvals[0]==0 or zvals[0] == math.inf
#     sum = 0
#     for i in range(1,len(zvals)):
#         temp =  L_SFSRATIO_Theta_Gamma_bin_i(p,i,gdm,n,dofolded,zvals[i])
#         sum += temp
#         if sum==-math.inf:
#             break
#     return -sum   

def NegL_SFSRATIO_Theta_Nsdensity_given_thetaN(p,gdm,n,thetaN,dofolded,densityof2Ns,zvals): 
    """
    uses thetaN estimated directly from snp count using watterson estimator
    """
    
    def L_SFSRATIO_Theta_Nsdensity_bin_i(p,i,gdm,n,dofolded,densityof2Ns,z): 
        try:
            if z==math.inf or z==0.0:
                return 0.0
            thetaN = p[0]
            thetaS = p[1]
            g = (p[2],p[3])
            density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],gdm,densityof2Ns,dofolded) for x in g_xvals])
            sint = float(np.trapz(density_values,g_xvals))
            ux=thetaS*sint                               
            uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i     
            alpha = ux/uy
            sigmay = math.sqrt(uy)
            beta = 1/sigmay
            return logprobratio(alpha,beta,z)             
        except:
            return -math.inf 
    assert zvals[0]==0 or zvals[0] == math.inf
    p = list(p)
    p.insert(0,thetaN)
    sum = 0
    for i in range(1,len(zvals)):
        temp =  L_SFSRATIO_Theta_Nsdensity_bin_i(p,i,gdm,n,dofolded,densityof2Ns,zvals[i])
        sum += temp
    return -sum   



# def NegL_SFSRATIO_Theta_Lognormal_given_thetaN(p,gdm,n,thetaN,dofolded,zvals): 
#     """
#     uses thetaN estimated directly from snp count 
#     """
#     m = 1.0
#     def L_SFSRATIO_Theta_Lognormal_bin_i(p,i,gdm,n,dofolded,z): 
#         try:
#             if z==math.inf or z==0.0:
#                 return 0.0
#             thetaN = p[0]
#             thetaS = p[1]
#             g = (p[2],p[3])
#             density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],gdm,"lognormal",dofolded) for x in g_xvals])
#             sint = float(np.trapz(density_values,g_xvals))
#             ux=thetaS*sint                               
#             uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i     
#             alpha = ux/uy
#             sigmay = math.sqrt(uy)
#             beta = 1/sigmay
#             return logprobratio(alpha,beta,z)             
#         except:
#             return -math.inf 
#     assert zvals[0]==0 or zvals[0] == math.inf
#     p = list(p)
#     p.insert(0,thetaN)
#     sum = 0
#     for i in range(1,len(zvals)):
#         temp =  L_SFSRATIO_Theta_Lognormal_bin_i(p,i,gdm,n,dofolded,zvals[i])
#         sum += temp
#     return -sum   


# def NegL_SFSRATIO_Theta_Gamma_given_thetaN(p,gdm,n,thetaN,dofolded,zvals): 
#     """
#         uses thetaN estiamted directly from snp count 
#     """

#     def L_SFSRATIO_Theta_Gamma_bin_i(p,i,gdm,n,dofolded,z): 
#         try:
#             if z==math.inf or z==0.0:
#                 return 0.0
#             thetaN = p[0]
#             thetaS = p[1]
#             g = (p[2],p[3])
#             density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],gdm,"gamma",dofolded) for x in g_xvals])
#             sint = float(np.trapz(density_values,g_xvals))
#             ux=thetaS*sint                               
#             uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i     
#             alpha = ux/uy
#             sigmay = math.sqrt(uy)
#             beta = 1/sigmay
#             return logprobratio(alpha,beta,z)             
#         except:
#             return -math.inf 
#     assert zvals[0]==0 or zvals[0] == math.inf
#     p = list(p)
#     p.insert(0,thetaN)
#     sum = 0
#     for i in range(1,len(zvals)):
#         temp =  L_SFSRATIO_Theta_Gamma_bin_i(p,i,gdm,n,dofolded,zvals[i])
#         sum += temp
#     return -sum   



def simsfs_continuous_gdist(theta,gdm,n,maxi,gdist, params, returnexpected):
    """
    simulate the SFS under selection, assuming a PRF Wright-Fisher model 
    uses a distribution of g (2Ns) values 
    gdist is "lognormal" or "gamma" ,params is two values
       
    """
    sfs = [0]*n
    
    for i in range(1,n):
        density_values = np.array([prfdensityfunction(x,n,i,params[0],params[1],gdm,gdist,False) for x in g_xvals])
        z = np.trapz(density_values,g_xvals)
        sfsexp = theta*z
        assert sfsexp>= 0
        if returnexpected:
            sfs[i] = sfsexp
        else:
            sfs[i] = np.random.poisson(sfsexp)
    sfsfolded = [0]
    for i in range(1,1+n//2):
        # print(i,n-i,sfs[i],sfs[n-i])
        sfsfolded.append(sfs[i]+sfs[n-i] if (n-i != i) else sfs[i])
    if maxi:
        assert maxi < n, "maxi setting is {} but n is {}".format(maxi,n)
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded

def simsfs(theta,g,n,maxi, returnexpected):
    """
        simulate the SFS under selection, assuming a PRF Wright-Fisher model 
        uses just a single value of g (2Ns), not a distribution
        if returnexpected,  use expected values, not simulated
        generates,  folded and unfolded for Fisher Wright under Poisson Random Field
    """
    if g==0:
        sfsexp = [0]+[theta/i for i in range(1,n)]
    else:
        sfsexp = [0]
        for i in range(1,n):
            u = prf_selection_weight(n,i,g,False)
            sfsexp.append(u*theta)    
    if returnexpected:
        sfs = sfsexp
    else:    
        sfs = [np.random.poisson(expected) for expected in sfsexp]
    sfsfolded = [0]
    for i in range(1,1+n//2):
        # print(i,n-i,sfs[i],sfs[n-i])
        sfsfolded.append(sfs[i]+sfs[n-i] if (n-i != i) else sfs[i])
    if maxi:
        assert maxi < n, "maxi setting is {} but n is {}".format(maxi,n)
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded


def simsfsratio(thetaN,thetaS,gdm,n,maxi,dofolded,gdist,params, returnexpected):
    """
    simulate the ratio of selected SFS to neutral SFS
    if returnexpected,  use expected values, not simulated
    if gdist is None,  params is just a g value,  else it is a list of distribution parameters
    if a bin of the neutral SFS ends up 0,  the program stops
    """
    
    nsfs,nsfsfolded = simsfs(thetaN,0,n,maxi,returnexpected)
    if gdist is None:
        ssfs,ssfsfolded = simsfs(thetaS,params,n,maxi,returnexpected)
    else:
        ssfs,ssfsfolded = simsfs_continuous_gdist(thetaS,gdm,n,maxi,gdist,params,returnexpected)
    ratios = [0]
    if dofolded:
        for i in range(1,len(nsfsfolded)):
            # assert nsfsfolded[i], "nsfs {} is zero".format(i)
            try:
                ratios.append(ssfsfolded[i]/nsfsfolded[i])
            except:
                ratios.append(math.inf)
    else:
        for i in range(1,len(nsfs)):
            try:
                ratios.append(ssfs[i]/nsfs[i])
            except:
                ratios.append(math.inf)        
    if dofolded:
        return nsfsfolded,ssfsfolded,ratios
    else:
        return nsfs,ssfs,ratios

#not in use as of 8/10/23
# def watterson_L(n,maxi,dofolded,counts):
#     """
#         uses waterson's estimator of theta 
#     """
#     #counts begins with a 0
#     # if folded the length is 1+n//2
#     wterm = 0.0
#     for i in range(1,n):
#         wterm += 1/i
#     thetaest = sum(counts)/wterm
#     if dofolded==False:
#         sumL = 0.0
#         for i in range(1,n):
#             un = thetaest/i
#             sumL += -un + math.log(un)*counts[i] - math.lgamma(counts[i]+1)
#         return thetaest,sumL
#     else:
#         sumL = 0.0
#         for i in range(1,len(counts)):# len(counts) should be 1+n//2
#             un = thetaest*n/(i*(n-i))
#             sumL += -un + math.log(un)*counts[i] - math.lgamma(counts[i]+1)
#     return thetaest,sumL

# not in use as of 8/10/23
# def NegL_2SFS_ThetaN_ThetaS_Ns(p,n,dofolded,nvals,svals,nog):
#     """
#         for two fisher wright sfs,  the first can have selection, the second is neutral
#         returns the negative of the log of the likelihood for both sfs's sampled from a Fisher Wright population
#         p has 2 or 3 values, one or two thetas and g,  if just one theta, it  applies to both sfs's
#         counts begins with a 0
#         returns the negative of the log of the likelihood for a sample from a Fisher Wright population
#         calls L_2SFS_ThetaN_ThetaS_Ns_bin_i()
#         if nog then both sfs's are without selection
#     """
#     def L_2SFS_ThetaN_ThetaS_Ns_bin_i(p,i,n,dofolded,nval,sval,nog): 
#         if nog: # then p should be a scalar
#             if len(p) == 2:
#                 thetaN = p[0]
#                 thetaS = p[1]
#             else:
#                 thetaN = p[0]
#                 thetaS = p[0]
#         else:
#             if len(p)==3:
#                 thetaN = p[0]
#                 thetaS = p[1]
#                 g = p[2]
#             else: #2 
#                 thetaN = thetaS = p[0]
#                 g = p[1]
#         if nog or g==0.0:
#             if dofolded:
#                 un = thetaN*n/(i*(n-i))
#                 temp = -un + math.log(un)*nval - math.lgamma(nval+1)
#                 us = thetaS*n/(i*(n-i))
#                 temp += -un + math.log(un)*sval - math.lgamma(sval+1)        
#             else:
#                 un = thetaN/i
#                 temp = -un + math.log(un)*nval - math.lgamma(nval+1)
#                 us = thetaS/i
#                 temp += -un + math.log(un)*sval - math.lgamma(sval+1)                    
#         else:
#             if dofolded:
#                 un = thetaN*n/(i*(n-i))
#                 temph = scipy.special.hyp1f1(i,n,2*g)+scipy.special.hyp1f1(n-i,n,2*g)
#                 tempc = coth(g)
#                 us = thetaS*(n/(2*i*(n-i)))*(tempc-1)*(2*math.exp(2*g)-temph)
#                 temp = -un + math.log(un)*nval - math.lgamma(nval+1)
#                 temp += -us + math.log(us)*sval - math.lgamma(sval+1)
#             else:
#                 un = thetaN/i
#                 temph = scipy.special.hyp1f1(i,n,2*g)
#                 tempc = coth(g)
#                 us = theta*(n/(2*i*(i-n)))*(-1 - tempc + (tempc-1)* temph)     
#                 temp = -un + math.log(un)*nval - math.lgamma(nval+1)
#                 temp += -us + math.log(us)*sval - math.lgamma(sval+1)
#         return temp     
#     assert(svals[0]==nvals[0]==0)
#     sum = 0
#     for i in range(1,len(svals)):
#         sum += L_2SFS_ThetaN_ThetaS_Ns_bin_i(p,i,n,dofolded,svals[i],nvals[i],nog)
        # if sum==-math.inf:
        #     break
#     return -sum 

#experimental,  calls estimate_thetaN(), has not been useful as of 8/10/23
# def NegL_SFSRATIO_Theta_Gamma_EX(p,n,countratio,dofolded,zvals): 
#     """
#         countratio is the ratio of total neutral counts divided by total selected counts 
#     """

#     def L_SFSRATIO_Theta_Gamma_bin_i(p,i,n,dofolded,z): 
#         try:
#             if z==math.inf or z==0.0:
#                 return 0.0
#             thetaN = p[0]
#             thetaS = p[1]
#             g = (p[2],p[3])
#             density_values = np.array([prfdensityfunction(x,n,i,g[0],g[1],"gamma",dofolded) for x in g_xvals])
#             sint = float(np.trapz(density_values,g_xvals))
#             ux=thetaS*sint                               
#             uy = thetaN*n/(i*(n-i)) if dofolded else thetaN/i       
#             alpha = ux/uy
#             sigmay = math.sqrt(uy)
#             beta = 1/sigmay
#             return logprobratio(alpha,beta,z)             
#         except:
#             return -math.inf 
#     assert zvals[0]==0 or zvals[0] == math.inf
#     thetaN = estimate_thetaN(p,n,countratio,dofolded)
#     p = list(p)
#     p.insert(0,thetaN)
#     sum = 0
#     for i in range(1,len(zvals)):
#         temp =  L_SFSRATIO_Theta_Gamma_bin_i(p,i,n,dofolded,zvals[i])
#         sum += temp
#     return -sum   
