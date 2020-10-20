import numpy as np
import matplotlib.pyplot as plt

#def likelihood(times, fluxes):
#	return 1./np.sqrt(2*np.pi)*np.exp(-0.5*(fluxes-model(times))**2)

def gaussnorm(x):
	'''
	standard normal distribution
	'''
	return 1/np.sqrt(2*np.pi)*np.exp(-(x)**2/(2))

def model(times, dip, tmin, tmax):
	'''
	transit model using simple inverted boxcar
	'''
	out = np.zeros_like(times)
	idxs_flat = np.where((times < tmin) | (times > tmax))
	idxs_dip = np.where((times > tmin) & (times < tmax))
	out[idxs_flat] = 1
	out[idxs_dip] = 1 - dip
	return out

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
#foldtime -= time[np.argmin(lcbins)]

dip = 0.007
tmin = 2
tmax = 2.3

sim = model(foldtime, dip, tmin, tmax)

plt.plot(foldtime, lcbins)
plt.plot(foldtime, sim)
plt.show()

params = np.array([dip, tmin, tmax])

for run in range(1000):
	proposal = np.random.multivariate_normal(params, np.identity(3)*1e-6, 1).flatten() # this is the Y vector

	r = gaussnorm(proposal)/gaussnorm(params)

	for value in range(len(r)):
		if r[value] >= 1:
			params[value] = proposal[value]
		if r[value] < 1:
			threshold = np.minimum(1, r[value])
			check = np.random.uniform(0, 1, 1)
			if check <= threshold:
				params[value] = proposal[value]

print(params)

plt.plot(foldtime, lcbins)
plt.plot(foldtime, model(foldtime, params[0], params[1], params[2]))
plt.show()
