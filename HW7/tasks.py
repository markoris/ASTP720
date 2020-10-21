import numpy as np
import matplotlib.pyplot as plt

def likelihood(times, fluxes, params):
	'''
	gaussian likelihood evaluating how well model represents data
	'''
	sigma = 1
	residual = fluxes - model(times, params)
	residual /= sigma
	residual = residual**2
	residual *= -0.5
	residual = np.exp(residual)
	residual *= 1/np.sqrt(2*np.pi*sigma**2) # Gaussian for array of times and fluxes
	residual = np.sum(np.log(residual))  # log of Gaussian allows for summing over likelihood
	residual = np.exp(residual)
	return residual

def model(times, params):
	'''
	transit model using simple inverted boxcar
	'''
	out = np.zeros_like(times)
	idxs_flat = np.where((times < params[1]) | (times > params[2]))
	idxs_dip = np.where((times > params[1]) & (times < params[2]))
	out[idxs_flat] = 1
	out[idxs_dip] = 1 - params[0]
	return out

# --- Creating folded light curve --- 

p = 3.5485 # days

time, flux = np.loadtxt('lightcurve_data.txt', unpack=True)
tmin = time[0]
tmax = time[-1]

time -= tmin

nbins = np.argmin(np.abs(time-p))

frac = (tmax - tmin)/nbins

lcbins = np.zeros(nbins)
counter = np.zeros(nbins)

for value in range(time.shape[0]):
	idx = int((time[value] % p) * frac*nbins)
	lcbins[idx] += flux[value]
	counter[idx] += 1

idxs = np.where(lcbins>0)

lcbins[idxs] /= counter[idxs]

foldtime = time[:nbins]
foldtime, lcbins = foldtime[idxs], lcbins[idxs]

# --- Using Metropolis-Hastings to find optimal parameters

dip = 0.007 # initial guess at dip of transit
tmin = 2    # initial guess at start of transit
tmax = 2.3  # initial guess at end of transit

params = np.array([dip, tmin, tmax])

print(likelihood(foldtime, lcbins, params), likelihood(foldtime, lcbins, np.array([0.5, 2, 2.3])))

sim = model(foldtime, params) # boxcar representation of model given initial guess of parameters

plt.plot(foldtime, lcbins)
plt.plot(foldtime, sim)
#plt.show() # pretty good! could be better...
plt.close()

cov_mat = np.identity(3)
cov_mat[0, 0] = dip/10.
cov_mat[1, 1] = tmin/10.
cov_mat[2, 2] = tmax/10.

dips = np.array([dip])
tmins = np.array([tmin])
tmaxs = np.array([tmax])

for run in range(1000):

	proposal = np.random.multivariate_normal(params, cov_mat, 1).flatten() # this is the Y vector, drawing from 3 different distributions, each with a mean at the current guess, with covariances = 1 on the diagonal

	while (np.any(proposal < 0)) | (proposal[2] < proposal[1]) | (proposal[2] > foldtime[-1]): # ensure that all values must be positive, tmax must be > tmin, and tmax should not be greater than the length of the relevant portion of the folded light curve
		proposal = np.random.multivariate_normal(params, cov_mat, 1).flatten()          # if any of above conditions are not met, re-draw samples until they are

	r = likelihood(foldtime, lcbins, proposal)/likelihood(foldtime, lcbins, params) # metropolis ratio, ratio of likelihood given new proposed parameters to likelihood of current proposed parameters

#	print(params, likelihood(foldtime, lcbins, params))
#	print(proposal, likelihood(foldtime, lcbins, proposal))
#	print(r)

	if r >= 1: params = proposal # if ratio > 1, proposed parameters are more likely to fit the data than current parameters, set current parameters to be the proposed ones
	dips = np.append(dips, params[0])
	tmins = np.append(tmins, params[1])
	tmaxs = np.append(tmaxs, params[2])
	if r < 1: 
		threshold = np.minimum(1, r) # otherwise, use the ratio as the threshold for acceptance
		check = np.random.uniform(0, 1, 1) # draw from uniform sample
		if check <= threshold: # if random draw is less than the threshold, accept the parameters, otherwise keep parameters same and draw another proposed set of parameters on the next iteration
			params = proposal
			dips = np.append(dips, params[0])
			tmins = np.append(tmins, params[1])
			tmaxs = np.append(tmaxs, params[2])

#plt.hist(dips, bins=50)
#plt.show()

#plt.plot(np.linspace(1, dips.shape[0], dips.shape[0]), dips)
#plt.show()

print(params)

plt.plot(foldtime, lcbins)
plt.plot(foldtime, model(foldtime, params))
plt.show()
