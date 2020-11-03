import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt

def time2freq(time):
	samp_time = np.mean(np.diff(time))
	samp_freq = 1/samp_time
	samp_freq /= 2 # Nyquist limit
	freq = np.linspace(0, 1, np.floor(time.shape[0]/2.).astype(int)-1)*samp_freq
	return freq

def dft(inputs):
	'''
	unoptimized DFFT in condensed form
	'''
	return np.real(np.array([np.sum(inputs[n]*np.exp(-2j*np.pi*k*n/inputs.shape[0]) for n in range(inputs.shape[0])) for k in range(inputs.shape[0])]))

def fft(inputs):
	'''
	optimized FFT using only 2 DFTs and a calculation of w
	'''
	N = inputs.shape[0]
	w = np.exp(-2j*np.pi/N)
	idxs = np.arange(0, np.floor(N/2).astype(int)-1)
	fft_even, fft_odd = np.zeros(idxs.shape[0]), np.zeros(idxs.shape[0])
	for i in idxs:
		fft_even[i] = np.sum([inputs[2*idx]*np.power(w, 2*i*idx) for idx in idxs])
		fft_odd[i] = np.sum([inputs[2*idx+1]*np.power(w, 2*i*idx)*np.power(w, i) for idx in idxs])
	return np.real(fft_even+fft_odd)

## --- Start of Consistency Test --- #
#
#time = np.linspace(0, 15, 2**10)
#sine = np.sin(2*np.pi*6*time)
#
#freq = time2freq(time)
#
#sine_fft = dft(sine)[:freq.shape[0]]
#
#good_fft  = np.real(np.fft.fft(sine)[:freq.shape[0]])
#
#transform = fft(sine)
#
#nested = nest_fft(sine)
#
#plt.plot(freq, sine_fft, 'k')
#plt.plot(freq, good_fft, 'r')
#plt.plot(freq, transform, 'b')
#plt.plot(freq, nested[:511], 'g')
#plt.show() # all 3 methods should produce roughly the same output
#
## --- End of Consistency Test --- #

# --- Start of Task 1 --- #

strain = np.load('strain.npy') # strain measurements, one minute apart
time = np.arange(strain.shape[0])*60 # seconds!

freq = time2freq(time)
#transform = fft(strain)
transform = np.fft.fft(strain) # using numpy's fft because mine takes forever, not nested yet...
transform = np.real(transform[:freq.shape[0]])

plt.yscale('log')
plt.xscale('log')
plt.plot(freq, transform)
#plt.show()

offset = np.where(freq > 0.001)[0][0]

peak_freq = freq[np.argmax(transform[np.where((freq > 0.001) & (freq < 0.004))])+offset] # limit between these frequencies, otherwise it'll catch the larger amplitude stuff at higher frequencies
peak = transform[np.argmax(transform[np.where((freq > 0.001) & (freq < 0.004))])+offset]
peak /= strain.shape[0] # divide by N to normalize amplitude
peak *= 2 # using one-sided spectrum, so multiply by 2 to conserve information

print(peak)

peak_freq /= 1e-4 # this value is equal to sqrt(M/M_sun) * (R/R_sun)^(-3/2) or M^1/2 = freq * R^3/2
peak /= 2.6e-21 # this value is equal to (M/M_sun)^2 * (D/pc)^(-1) * (R/R_sun)^(-1)
peak *= 12 # this  value is (M/M_sun)^2 (R/R_sun)^(-1), or alternatively, M^2 = peak * R
peak = np.power(peak, 1./4) # now, M^1/2 = peak * R^1/4, therefore peak * R^1/4 = freq * R^3/2 -> peak * R^1/4 = freq * R^6/4 -> peak/freq = R^5/4, R = np.power(peak/freq, 4/5)
R = np.power(peak/peak_freq, 4./5)
M = np.sqrt(peak*R)
print(M, R)

# --- End of Task 1 --- #