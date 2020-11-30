import numpy as np
import matplotlib.pyplot as plt

# dI/ds = (I(s + h) - I(s))/h
# I(s+h) = h*dI/ds + I(s)
# I1 = 0.1*dI/ds + I0

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

wav = np.array([365, 476, 621, 754, 900, 1020, 1220, 1630, 2190])*1e-9 # grizyJHK
T = np.array([5700])
ds = 0.1
rel_diff = 0.001

intensity = planck(wav, T)*np.random.uniform(10, 100)
dist = np.array([ds])

for idx in range(1000):
	alpha = 1/dist[-1]
	intensity = np.append(intensity, I_next(intensity[-1], alpha, ds, wav, T), axis=0)
	dist = np.append(dist, (idx+2)*ds)
	if idx > 0:
		diff = (intensity[-2, :]-intensity[-1, :])/intensity[-2, :]
		if np.average(diff) <= rel_diff:
			print('relative difference below threshold, terminating early at %d steps' % idx)
			break

for idx in range(intensity.shape[1]):
	plt.plot(dist, intensity[:, idx], colors[wavelengths[idx]], label=wavelengths[idx])
plt.legend()
plt.yscale('log')
plt.show()