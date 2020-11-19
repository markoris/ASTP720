"""
Michael Lam
ASTP-720, Fall 2020

Below is code associated with HW#9. It is provided as a complement to the
Jupyter notebook and contains the same functionality. You must have both the
emcee and corner packages installed on your machine.
"""


import numpy as np
from matplotlib.pyplot import *
from matplotlib import rc
import emcee
import corner

# Make more readable plots
rc('font',**{'size':14})
rc('xtick',**{'labelsize':16})
rc('ytick',**{'labelsize':16})
rc('axes',**{'labelsize':18,'titlesize':18})


"""
## Define the (log-)priors

Here, the function should take a vector of parameters, `theta`, and return `0.0`
if the it is in the prior range and `-np.inf` if it is outside. This is equivalent
to a uniform prior over the parameters. You can, of course, define a different
set of priors if you so choose!
"""


def lnprior(theta):
    """
    Parameters
    ----------
    theta : np.ndarray
        Array of parameters.

    Returns
    -------
    Value of log-prior.
    """
#    if np.all(np.abs(theta) < 1): return 0
    if np.all((np.abs(theta[:3]) < 1) & (theta[3] > 0)): return 0
    return -np.inf
#    return np.where(np.abs(theta) <= 1, 0, -np.inf)



"""
## Define the (log)-likelihood
"""

def lnlike(theta, data):
    """
    Parameters
    ----------
    theta : np.ndarray
        Array of parameters.
    data : np.ndarray


    Returns
    -------
    Value of log-likelihood
    """
    mean = np.mean(data)
#    sigma_z = 1
    sigma_z = theta[3]
    residuals = np.zeros(len(data)-121)  # 3141 entries
    for idx in np.arange(121, len(data)):
    	residuals[idx-121] = (data[idx]-mean) - theta[0]*(data[idx-1]-mean) - theta[1]*(data[idx-12]-mean) - theta[2]*(data[idx-121]-mean) # entry 0 of residuals = entry 121 - entry 120 - entry 109 - entry 0
    residuals = -np.sum(residuals**2)/(2*sigma_z**2)
    residuals += -(len(data)-121)/2*np.log(2*np.pi*sigma_z**2)
    return residuals

"""
## Define total (log-)probability

No need to change this if the other two functions work as described.
"""

def lnprob(theta, data):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data)

"""
## Set up the MCMC sampler here
"""

# Number of walkers to search through parameter space
nwalkers = 10
# Number of iterations to run the sampler for
niter = 5000
# Initial guess of parameters. For example, if you had a model like
# s(t) = a + bt + ct^2
# and your initial guesses for a, b, and c were 5, 3, and 8, respectively, then you would write
# pinit = np.array([5, 3, 8])
# Make sure the guesses are allowed inside your lnprior range!
pinit = np.array([0.05, 0.05, 0.50, 1])
# Number of dimensions of parameter space
ndim = len(pinit)
# Perturbed set of initial guesses. Have your walkers all start out at
# *slightly* different starting values
p0 = [pinit + 1e-4*pinit*np.random.randn(ndim) for i in range(nwalkers)]


"""
## Load the data, plot to show
"""
# Data: decimal year, sunspot number
decyear, ssn = np.loadtxt("SN_m_tot_V2.0.txt", unpack=True, usecols=(2, 3))
plot(decyear, ssn, 'k.')
xlabel('Year')
ylabel('Sunspot Number')
show()


"""
## Run the sampler
"""
# Number of CPU threads to use. Reduce if you are running on your own machine
# and don't want to use too many cores
nthreads = 4
# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssn,), threads=nthreads)
# Run the sampler. May take a while! You might consider changing the
# number of iterations to a much smaller value when you're testing. Or use a
# larger value when you're trying to get your final results out!
sampler.run_mcmc(p0, niter, progress=True)


"""
## Get the samples in the appropriate format, with a burn value
"""

# Burn-in value = 1/4th the number of iterations. Feel free to change!
burn = int(0.25*niter)
# Reshape the chains for input to corner.corner()
samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

new_theta = np.mean(samples, axis=0)
print(new_theta)
model = np.zeros(len(ssn)-121)  # 3141 entries
for idx in np.arange(121, len(ssn)):
	model[idx-121] = (ssn[idx]) - new_theta[0]*(ssn[idx-1]) - new_theta[1]*(ssn[idx-12]) - new_theta[2]*(ssn[idx-121])

plot(decyear[121:], ssn[121:], 'k.')
plot(decyear[121:], model, 'r.')
xlabel('Year')
ylabel('Sunspot Number')
show()

"""
## Make a corner plot

You should feel free to adjust the parameters to the `corner` function.
You **should** also add labels, which should just be a list of the names
of the parameters. So, if you had two parameters, $\phi_1$ and $\phi_2$,
then you could write:

labels = [r"$\phi_1$", r"$\phi_2$"]

and that will make the appropriate label in LaTeX (if the distribution is
installed correctly) for the two 1D posteriors of the corner plot.
"""

fig = corner.corner(samples, bins=50, color='C0', smooth=0.5, plot_datapoints=False, plot_density=True, \
                    plot_contours=True, fill_contour=False, show_titles=True)#, labels=labels)
fig.savefig("corner.png")
show()
