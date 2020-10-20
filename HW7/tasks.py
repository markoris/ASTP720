import numpy as np
import matplotlib.pyplot as plt

def likelihood(times, fluxes, params):
	sigma = 0.1
	residual = fluxes - model(times, params)
	residual /= sigma
	residual = -0.5*np.sum(residual**2)
	residual = np.exp(residual)
	residual *= 1/np.sqrt(2*np.pi*sigma**2)
	return residual
#	return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*np.sum(((fluxes - model(times, params))/sigma)**2))
#	return 1/np.sqrt(2*np.pi)*np.exp(-0.5*np.sum((fluxes-model(times, params))**2/fluxes))

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

dip = 0.007
tmin = 2
tmax = 2.3

params = np.array([dip, tmin, tmax])

sim = model(foldtime, params)

plt.plot(foldtime, lcbins)
plt.plot(foldtime, sim)
plt.show()

for run in range(10):
	proposal = np.random.multivariate_normal(params, np.identity(3), 1).flatten() # this is the Y vector

	while (np.any(proposal < 0)) | (proposal[2] < proposal[1]) | (proposal[2] > foldtime[-1]):
		proposal = np.random.multivariate_normal(params, np.identity(3), 1).flatten() # this is the Y vector

	r = likelihood(foldtime, lcbins, proposal)/likelihood(foldtime, lcbins, params)

	if r >= 1: params = proposal
	if r < 1:
		threshold = np.minimum(1, r)
		check = np.random.uniform(0, 1, 1)
		if check <= threshold:
			params = proposal

print(params)

plt.plot(foldtime, lcbins)
plt.plot(foldtime, model(foldtime, params))
plt.show()
