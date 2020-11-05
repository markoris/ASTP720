import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt

plt.rc('font', size=40)

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
	optimized FFT using recursive function calls and a pre-calculated w factor
	'''
	N = inputs.shape[0]
	if N <= 1: return dft(inputs) # if array length below some threshold, just do a regular DFT. Using 1 since this matched the DFT best for the sine wave consistency check

	idxs = np.arange(N//2) # linearly spaced N/2 index numbers for odd and even number identification

	fft_even = fft(inputs[idxs*2]) # using even indices of the fft
	fft_odd = fft(inputs[(idxs*2)+1]) # using odd indices of the fft

	w = [np.exp(-2j*np.pi*k/N) for k in range(N)] # calculate w for each of the indices

	return np.append(fft_even + w[:N//2]*fft_odd, fft_even + w[N//2:]*fft_odd) # appending second half to ensure that array size is maintained throughout recursion, but we only need the frequencies relevant to the first half of the array


# --- Start of Consistency Test --- #

time = np.linspace(0, 15, 2**10)
sine = np.sin(2*np.pi*6*time)

freq = time2freq(time)

sine_dft = dft(sine)[:freq.shape[0]]

good_fft  = np.real(np.fft.fft(sine)[:freq.shape[0]])

sine_copy = np.copy(sine)

transform = fft(sine_copy)

plt.plot(freq, sine_dft, 'k', label='DFT')
plt.plot(freq, transform[:511], 'b', label='Cooley-Tukey FFT')
plt.plot(freq, good_fft, 'r', label='Numpy FFT')
plt.title('Comparison of Fourier Transform Methods for Sin(2*pi*6*t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.show() # all 3 methods should produce roughly the same output

# --- End of Consistency Test --- #

# --- Start of Task 1 --- #

strain = np.load('strain.npy') # strain measurements, one minute apart
time = np.arange(strain.shape[0])*60 # seconds!

freq = time2freq(time) # converting time to frequency domain
transform = fft(strain) # calculating fft of strain data
transform = np.real(transform[:freq.shape[0]])*2 # using only relevant part of FFT data and conserving variance by multiplying one-sided spectrum by 2

plt.yscale('log')
plt.xscale('log')
plt.plot(freq, transform)
plt.title('Cooley-Tukey FFT of strain data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

offset = np.where(freq > 0.001)[0][0] # index offset for finding the location of the peak in the entire array, not just the subset searched

peak_freq = freq[np.argmax(transform[np.where((freq > 0.001) & (freq < 0.004))])+offset] # limit between these frequencies, otherwise it'll catch the larger amplitude stuff at higher frequencies
peak = transform[np.argmax(transform[np.where((freq > 0.001) & (freq < 0.004))])+offset]
peak /= strain.shape[0] # divide by N to normalize amplitude

print("frequency = ", peak_freq, "hz, amplitude = ", peak)

peak_freq /= 1e-4 # this value is equal to sqrt(M/M_sun) * (R/R_sun)^(-3/2) or M^1/2 = freq * R^3/2
peak /= 2.6e-21 # this value is equal to (M/M_sun)^2 * (D/pc)^(-1) * (R/R_sun)^(-1)
peak *= 12 # this  value is (M/M_sun)^2 (R/R_sun)^(-1), or alternatively, M^2 = peak * R
peak = np.power(peak, 1./4) # now, M^1/2 = peak * R^1/4, therefore peak * R^1/4 = freq * R^3/2 -> peak * R^1/4 = freq * R^6/4 -> peak/freq = R^5/4, R = np.power(peak/freq, 4/5)
R = np.power(peak/peak_freq, 4./5) # R = np.power(peak/freq, 4/5)
M = np.sqrt(np.power(peak, 4)*R) # Since M^2 = peak * R, M = sqrt(peak*R), however this is the peak value before any modifications for finding separation R

print("mass =", M, "M_sun, separation =", R, "R_sun.")

# --- End of Task 1 --- #