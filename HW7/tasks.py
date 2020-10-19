import numpy as np
import matplotlib.pyplot as plt

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

lcbins /= counter

idxs = np.where(lcbins>0)

foldtime = time[:nbins] # 174 time steps
foldtime, lcbins = foldtime[idxs], lcbins[idxs]
foldtime -= time[np.argmin(lcbins)]

plt.plot(foldtime, lcbins)
plt.show()