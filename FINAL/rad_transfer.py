import numpy as np
import matplotlib.pyplot as plt

'''
Assumptions:
- blackbody emission
- photon moving out at steady time steps (not Monte Carlo with wind, just assumed outward trajectory rather than actual random walk)
- simple alpha/r behavior corresponding to thinning opacity as photon travels further out of the atmosphere
- very simple piece-wise quantum efficiency for camera
- only three viewing filters (u, r, and i)
'''

def planck(wav, T):
	'''
	Planck function which accepts a float or array of wavelengths (nm) and temperatures (Kelvin) and evaluates the function at those values

	wav : float or array
		float or array of wavelength values in nanometers

	T : float or array
		float or array of temperature values in Kelvin
	'''
	h = 6.626e-34 # J s
	c = 2.998e8 # m / s
	kb = 1.380e-23 # J / K
	return np.array([2*h*np.power(c, 2)/np.power(wav, 5)/(np.exp(h*c/(wav*kb*T))-1)]).reshape(-1, len(wav))

def I_next(I_now, alpha, ds, wav, T):
	'''	
	finite difference method
	dI/ds = (I(s + h) - I(s))/h  definition of finite difference derivative
	I(s+h) = h*dI/ds + I(s)      moving things around to isolate the value of the function at the next step
	I1 = 0.1*dI/ds + I0          first iteration example utilizing the initial condition as a starter

	Note: does not make sense to use photon destruction probability because that's used in non-LTE radiative transfer, we are assuming LTE for simplicity

	I_now : float
		value of the intensity at the current distance

	alpha : float
		value of the absorption coefficient, with 0 representing no absorption (vacuum)

	ds : float
		the step size which decides how rapidly the distance from the source is incremented

	wav : float or array
		float or array of wavelength values in nanometers

	T : float or array
		float or array of temperature values in Kelvin
	'''
	return ds*(alpha*planck(wav, T) - I_now) + I_now

### Setting up values to be used in the finite difference loop

plt.rc('font', size=30)
plt.rc('lines', lw=3)
wavelengths = 'ugrizyJHK' # letter representations of all the filters 
colors = {"K":"darkred", "H":"red", "J":"orange", "y":"gold", "z":"greenyellow", "i":"green", "r":"lime", "g":"cyan", "u":"blue"} # plotting colors chosen logically with shorter wavelengths being bluer and longer wavelengths being redder
classifications = 'OBAFGKMLTY' # stellar classifications 
temps = np.array([40000, 20000, 8750, 6750, 5600, 4450, 3050, 1850, 1000, 600]) # effective temperature at stellar surface for OBAFGKMLTY star
wav = np.array([365, 476, 621, 754, 900, 1020, 1220, 1630, 2190])*1e-9 # ugrizyJHK filter wavelengths
ds = 0.1 # step size in meters
rel_diff = 1e-2 # relative difference requirement between subsquent steps; if the value of (s_{i} - {s_i+1})/s_{i} is less than this, the evaluation is terminated
u = np.loadtxt('filters/bess-u.pass', unpack=True).T # taken from Michael Richmond's ASTP-613 notes on optical detectors (week 7B)
r = np.loadtxt('filters/bess-r.pass', unpack=True).T
i = np.loadtxt('filters/bess-i.pass', unpack=True).T
filters = np.array([u, r, i], dtype=object) # the three filter considered for the camera portion
verbose = True # whether or not to print early termination warning
filter_to_wav = {0: 0, 1: 2, 2:3} # first filter is u, translate to first wavelength in wav, second filter is r, translate to third wavelength in wav, etc...
qe1, qe2 = np.random.uniform(0.5, 1), np.random.uniform(0, 0.5) # quantum efficiences, with the first one (shorter wavelengths) arbitrarily chosen to be higher and the second one (longer wavelengths) arbitrarily chosen to be lower

### Looping over the temperature values in the temps array to consider all the stellar classes specified in classifications

for T in temps: # looping over temperatures

	intensity = planck(wav, T)*np.random.uniform(10, 100) # the initial value of the intensity I_0 is somewhere between 10 to 100 times the Planck function
	dist = np.array([ds]) # initialize distances array
	
	for idx in range(10000000): # assume many iterations since the rel_diff will likely stop before we hit max iterations
		alpha = 1/dist[-1] # take the absorption to drop off as 1/r which roughly corresponds to how the density would change as a function of distance
		alpha_coeffs = wav/np.sum(wav)/(wav[-1]/np.sum(wav)) # make alpha dependent on wavelength, bluer wavelengths should have larger scattering/opacity than redder
		alpha *= alpha_coeffs # scale alpha values accordingly
		intensity = np.append(intensity, I_next(intensity[-1], alpha, ds, wav, T), axis=0) # append intensity value at next step to intensity array
		dist = np.append(dist, (idx+2)*ds) # append next step to distance array
		if idx > 0: # comparison cannot be made at first time step, need at least 2 values to compare relative difference
			diff = (intensity[-2, :]-intensity[-1, :])/intensity[-2, :] # (s_{i} - {s_i+1})/s_{i} for each of the wavelengths in the wav array
			if np.average(diff) <= rel_diff: # if the average relative difference is below the threshold, terminate
				if verbose: print('relative difference below threshold of %e, terminating early at %d steps' % (rel_diff, idx))
				break

### Plotting intensity as a function of distance for all wavelengths considered in the wav array

	plt.figure(figsize=(19.2, 10.8))
	for idx in range(intensity.shape[1]):
		plt.plot(dist, intensity[:, idx], colors[wavelengths[idx]], label=wavelengths[idx]) # plot the intensity as a function of distance for each of the wavelengths in the wav array
	plt.legend()
	plt.yscale('log') # log-scale y-axis represents the change in intensity in a more meaningful way
	plt.xlabel(r'$s$')
	plt.ylabel(r'$I(\lambda, T)$')
	plt.title('%s Star Intensities at Given Wavelengths' % classifications[int(np.where(temps==T)[0])])
	plt.savefig('figures/%s.png' % classifications[int(np.where(temps==T)[0])])
	plt.close()

### Examining what the intensity throughput would be when using a camera with some simple piece-wise quantum efficiency (above and below a threshold wavelength) viewed through Johnson u, r, and i filters

	for idx in range(filters.shape[0]): # loop through the u, r, and i filters
		filt = filters[idx]
		if T == temps[0]: # if the first of the temperatures considered, scale the values in the filter data from Angstroms to nanometers
			filt[:, 0] /= 10 # convert from Angstroms
			filt[:, 0] *= 1e-9 # and then to actual nanometers
		cutoff_wav = 0.6e-6 # above 600 nanometers, your CCD suddenly drops in quantum efficiency for some reason (probably a grad student made a mistake)
		quantum_efficiency = np.piecewise(filt[:, 0], [filt[:, 0] <= cutoff_wav, filt[:, 0] > cutoff_wav], [qe1, qe2]) # if wav < 600 nm, quantum efficiency is the first value, otherwise it is the second value
		transmissivity = np.zeros_like(filt[:, 0]) # initialize the transmissivity array
		transmissivity[np.where(filt[:, 0] < cutoff_wav)[0]] = filt[np.where(filt[:, 0] < cutoff_wav)[0], 1]*quantum_efficiency[0] # transmissivity is the product of the quantum efficiency times the filter's throughput
		transmissivity[np.where(filt[:, 0] >= cutoff_wav)[0]] = filt[np.where(filt[:, 0] >= cutoff_wav)[0], 1]*quantum_efficiency[-1]
		plt.figure(figsize=(19.2, 10.8))
		plt.xlabel(r'$\lambda$')
		plt.ylabel(r'$I(\lambda, T)$')
		plt.title('%s Star Intensities at Given Wavelengths Viewed by Camera' % classifications[int(np.where(temps==T)[0])])
		plt.yscale('log')
		plt.plot(filt[:, 0], transmissivity*intensity[-1, filter_to_wav[idx]]) # plot the last value of the intensity (where the finite differencing terminated) times the throughput of the camera + filters
#		plt.show()
		plt.close()	
