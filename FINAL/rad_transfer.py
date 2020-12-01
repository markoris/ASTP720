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
verbose = True

for T in temps:
	wav = np.array([365, 476, 621, 754, 900, 1020, 1220, 1630, 2190])*1e-9 # grizyJHK
	ds = 0.1
	rel_diff = 1e-4
	
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
	plt.savefig('%s.png' % classifications[int(np.where(temps==T)[0])])
	plt.close()