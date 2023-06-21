import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
"""
simulate distribution of ratio Z=X/Y  where X and Y are poisson rvs
use expressions in Díaz-Francés, E. and F. J. Rubio (2013). "On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables." Statistical Papers 54: 309-323.
to approximate the density	
plot actual function for the density of the ratio of two normals (ex (1) of paper )
and plot the function for the normal approximation (ex (11) in paper) with mean uz=ux/uy  and sigma  uz * pow(deltax2 + deltay2,1/2)

to run:
    set the poisson rates,  ux for the numerator and uy for the denominator 
    if the normal approximation for the ratio is desired in the plot,  napprox = True

"""
def f(z,beta,rho,deltay):
    rho2 = rho*rho
    z2 = z*z
    q = (1+beta*rho2*z)/(deltay*pow(1+rho2*z2,1/2))
    erfq = math.erf(q/pow(2,1/2))
    temp1 = rho/(math.pi * (1+rho2*z2))
    temp2 = math.exp(-(rho2*beta*beta + 1)/(2*deltay*deltay))
    if temp2 > 0:
        temp3 = 1+ pow(math.pi/2,1/2)*q*erfq*math.exp(q*q/2)
        return temp1*temp2*temp3
    else:
        return 0
ux = 5
uy = 10
napprox = False

sigmax = pow(ux,1/2)
sigmay = pow(uy,1/2)
deltax = 1/sigmax
deltay = 1/sigmay
deltax2 = 1/ux 
deltay2 = 1/uy
uz = ux/uy
sigmaz = uz * pow(deltax2 + deltay2,1/2)
varz = sigmaz*sigmaz

beta = ux/uy
rho = sigmay/sigmax
sigmax = pow(ux,1/2)
n = 10000000
xvals = np.array([np.random.poisson(ux) for i in range(n)])
yvals = np.array([np.random.poisson(uy) for i in range(n)])
zvals = np.array([xvals[i]/yvals[i] for i in range(n) if yvals[i] > 0])
zspace = np.linspace(zvals.min(), zvals.max(), 200)
fzspace = np.array([f(z,beta,rho,deltay) for z in zspace])

print("mean",zvals.mean(),"var",zvals.var())
print("approx mean",uz,"approx var",varz)
# Fit a normal distribution to the data:
# mu, std = norm.fit(zvals)

# Plot the histogram.
plt.hist(zvals, bins=200, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
if napprox:
    approxfz = norm.pdf(x, uz, sigmaz)
    plt.plot(x, approxfz, linewidth=2)
plt.plot(zspace, fzspace, color='red')
title = "Fit results: mu = %.2f,  std = %.4f" % (uz,sigmaz)
plt.title(title)

plt.show()
