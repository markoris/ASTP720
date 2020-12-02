import numpy as np
import matplotlib.pyplot as plt
# finite difference method
# dI/ds = (I(s + h) - I(s))/h
# I(s+h) = h*dI/ds + I(s)
# I1 = 0.1*dI/ds + I0

'''
Assumptions:
- blackbody emission
- photon moving out at steady time steps (not Monte Carlo with wind, just assumed outward trajectory rather than actual random walk)
- simple alpha/r behavior corresponding to thinning opacity as photon travels further out of the atmosphere
- 
'''

def gaussian(x, mu, sigma):
	return 1./np.sqrt(2*np.pi*sigma**2)*np.exp(-(x - mu)**2/(2*sigma**2))

def planck(wav, T):
	h = 6.626e-34 # J s
	c = 2.998e8 # m / s
	kb = 1.380e-23 # J / K
	return np.array([2*h*np.power(c, 2)/np.power(wav, 5)/(np.exp(h*c/(wav*kb*T))-1)]).reshape(-1, len(wav))

def I_next(I_now, alpha, ds, wav, T):
	# does not make sense to use epsilon (photon destruction probability) because that's used in non-LTE radiative transfer, we are assuming LTE for simplicity :)
	return ds*(alpha*planck(wav, T) - I_now) + I_now

wavelengths = 'ugrizyJHK'
colors = {"K":"darkred", "H":"red", "J":"orange", "y":"gold", "z":"greenyellow", "i":"green", "r":"lime", "g":"cyan", "u":"blue"}
classifications = 'OBAFGKMLTY'
temps = np.array([40000, 20000, 8750, 6750, 5600, 4450, 3050, 1850, 1000, 600]) # effective temperature at stellar surface for OBAFGKMLTY star
wav = np.array([365, 476, 621, 754, 900, 1020, 1220, 1630, 2190])*1e-9 # ugrizyJHK
ds = 0.1
rel_diff = 1e-2
u = np.loadtxt('filters/bess-u.pass', unpack=True).T # taken from Michael Richmond's ASTP-613 notes on optical detectors (week 7B)
r = np.loadtxt('filters/bess-r.pass', unpack=True).T
i = np.loadtxt('filters/bess-i.pass', unpack=True).T
filters = np.array([u, r, i])
verbose = True
filter_to_wav = {0: 0, 1: 2, 2:3} # first filter is u, translate to first wavelength in wav, second filter is r, translate to third wavelength in wav, etc...
qe1, qe2 = np.random.uniform(0.5, 1), np.random.uniform(0, 0.5)

for T in temps:

	intensity = planck(wav, T)*np.random.uniform(10, 100)
	dist = np.array([ds])
	
	for idx in range(10000000):
		alpha = 1/dist[-1]
		alpha_coeffs = wav/np.sum(wav)/(wav[-1]/np.sum(wav)) # make alpha dependent on wavelength, bluer wavelengths should have larger scattering/opacity than redder
		alpha *= alpha_coeffs # scale alpha values accordingly
		intensity = np.append(intensity, I_next(intensity[-1], alpha, ds, wav, T), axis=0)
		dist = np.append(dist, (idx+2)*ds)
		if idx > 0:
			diff = (intensity[-2, :]-intensity[-1, :])/intensity[-2, :]
			if np.average(diff) <= rel_diff:
				if verbose: print('relative difference below threshold of %e, terminating early at %d steps' % (rel_diff, idx))
				break

	plt.figure(figsize=(19.2, 10.8))
	for idx in range(intensity.shape[1]):
		plt.plot(dist, intensity[:, idx], colors[wavelengths[idx]], label=wavelengths[idx])
	plt.legend()
	plt.yscale('log')
	plt.savefig('figures/%s.png' % classifications[int(np.where(temps==T)[0])])
	plt.close()

	for idx in range(filters.shape[0]):
		filt = filters[idx]
		if T == temps[0]:
			filt[:, 0] /= 10 # convert from Angstroms to actual nanometers
			filt[:, 0] *= 1e-9
		cutoff_wav = 0.6e-6 # above 600 nanometers, your CCD suddenly drops in quantum efficiency for some reason (probably a grad student made a mistake)
		quantum_efficiency = np.piecewise(filt[:, 0], [filt[:, 0] <= cutoff_wav, filt[:, 0] > cutoff_wav], [qe1, qe2])
		transmissivity = np.zeros_like(filt[:, 0])
		transmissivity[np.where(filt[:, 0] < cutoff_wav)[0]] = filt[np.where(filt[:, 0] < cutoff_wav)[0], 1]*quantum_efficiency[0]
		transmissivity[np.where(filt[:, 0] >= cutoff_wav)[0]] = filt[np.where(filt[:, 0] >= cutoff_wav)[0], 1]*quantum_efficiency[-1]
		plt.yscale('log')
		plt.plot(filt[:, 0], transmissivity*intensity[-1, filter_to_wav[idx]])
		plt.show()
	
