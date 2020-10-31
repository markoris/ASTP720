import numpy as np
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
	# not working yet... not sure why. need to add recursion?
	fft = np.zeros(inputs.shape)
	N = inputs.shape[0]
	w = np.exp(-2j*np.pi/N)
	idxs = np.arange(0, np.floor(N/2).astype(int)-1)
	fft_even =  dft(inputs[2*idxs])
	fft_odd = dft(inputs[(2*idxs)+1])
	return np.array([fft_even[k] - fft_odd[k]*np.power(w, k) for k in range(fft_even.shape[0])])

# --- Start of Consistency Test --- #

time = np.linspace(0, 15, 2**10)
sine = np.sin(2*np.pi*6*time)

freq = time2freq(time)

sine_fft = dft(sine)[:freq.shape[0]]

good_fft  = np.real(np.fft.fft(sine)[:freq.shape[0]])

fft = fft(time)

print(fft.shape)

plt.plot(freq, sine_fft, 'k')
plt.plot(freq, good_fft, 'r')
plt.plot(freq, fft, 'b')
plt.show()

# --- End of Consistency Test --- #