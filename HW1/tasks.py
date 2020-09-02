# --- Start of Task 1 ---
#
# See root_finding_algos.py
#
# --- End of Task 1 ---

# --- Start of Task 2 ---

import root_finding_algos as rfa
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', size=30)

def pseudo_iso_sphere_unitless(x):
	'''
	Full-width half-maximum of pseudo-isothermal sphere with x in units of r_c.
	'''
	return (1. + x**2)**(-0.5) - 0.5

def pseudo_iso_sphere_unitless_deriv(x):
	'''
	Derivative of full-width half-maximum of pseudo-isothermal sphere with x in units of r_c.
	'''
	return -x/(1 + x**2)**(1.5)

thresholds = np.logspace(-1, -10, 10) # log-spaced thresholds ranging from 0.1 to 1e-10

for threshold in thresholds:
	root_bisect, iters_bisect = rfa.bisection(pseudo_iso_sphere_unitless, 0, 2, eps=threshold, verbose=True) # get bisection method root and number iterations
	root_newton, iters_newton = rfa.newton(pseudo_iso_sphere_unitless, pseudo_iso_sphere_unitless_deriv, 0.5, eps=threshold, verbose=True) # get newton's method root and number iterations
	root_secant, iters_secant = rfa.secant(pseudo_iso_sphere_unitless, 0, 2, eps=threshold, verbose=True) # get secant method root and number iterations
	try:
		iters_b = np.append(iters_b, iters_bisect) # append to array for plotting
		iters_n = np.append(iters_n, iters_newton)
		iters_s = np.append(iters_s, iters_secant)
	except NameError:
		iters_b = iters_bisect # if no array, create something to append to if on the first iteration
		iters_n = iters_newton
		iters_s = iters_secant
plt.scatter(np.flip(thresholds), iters_b, color='r', label='Bisection')
plt.scatter(np.flip(thresholds), iters_n, color='b', label='Newton')
plt.scatter(np.flip(thresholds), iters_s, color='g', label='Secant')
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Iterations until sub-threshold')
plt.xscale('log')
plt.show()

# --- End of Task 2 ---

# --- Start of Task 3 ---

def lens_equation(x):
	m = 11 # "month" which just corresponds to where in the orbit the observer is located, repeats as mod 12
	D = 3.086e21 # 1 kpc in centimeters
	a = 1.496e13 # 1 AU in centimeters
	wavelength = 21 # centimeters
	N_o = 0.01*3.086e18 # 0.01 pc cm^-3 as cm^-2
	r_e = (4.803e-10)**2/(9.1094e-28)/(2.998e10)**2 # classical radius e^2/m_e*c*2 in cgs units
	return x*(1 + (wavelength**2*r_e*N_o*D)/(np.pi*a**2)*np.exp(-(x/a)**2)) - (1 + np.cos(m*np.pi/6))

#x = rfa.secant(lens_equation, -1, 1) # find the x corresponding to the given month used in lens_equation function
#file = open('lens_equation.dat', 'a')  # Could not come up with a way to loop over the "months" of the orbit
#file.write(str(x)+'\n')			    # since return breaks the loop, so manually changed month value (m in lens_equation)
#file.close()						    # and recorded the output of each by appending to a file... very primitive and ugly :(

months = np.arange(0, 12, 1) # 12 months in a year, spanning from 0 to 11 numerically

xprimes = 1 + np.cos(months*np.pi/6) # locations of all the x primes for each "month" in the orbit
xs = np.loadtxt('lens_equation.dat') # loading x values saved using ugly method described a few lines prior
for idx in range(len(xs)):
	x1, y1 = [0, xs[idx]], [2, 1] # plot a line from the source to the location where ray hits on lens plane 
	x2, y2 = [xs[idx], xprimes[idx]], [1, 0] # plot line from where ray hits on lens plane to where observer views same ray
	plt.plot(x1, y1, x2, y2)
plt.ylabel('Arbitrary distance offset')
plt.xlabel('Observer location (AU)')
plt.title('Gaussian lens ray-tracing diagram')
plt.show()

# --- End of Task 3 ---

# --- Start of Task 4 ---

def lens_equation_pseudo_isothermal(x):
	m = 11 # "month" which just corresponds to where in the orbit the observer is located, repeats as mod 12
	D = 3.086e21 # 1 kpc in centimeters
	a = 1.496e13 # 1 AU in centimeters
	wavelength = 21 # centimeters
	N_o = 0.01*3.086e18 # 0.01 pc cm^-3 as cm^-2
	r_e = (4.803e-10)**2/(9.1094e-28)/(2.998e10)**2 # classical radius e^2/m_e*c*2 in cgs units
	r_c = 1.496e13
	return x*(1 + (wavelength**2*r_e*N_o*D)/(2*np.pi*r_c**2*(1 + (x/r_c)**2)**1.5)) - (1 + np.cos(m*np.pi/6))

#x = rfa.secant(lens_equation_pseudo_isothermal, -1, 1) # find the x corresponding to the given month used in lens_equation function
#file = open('lens_equation_pseudo_isothermal.dat', 'a')  # Could not come up with a way to loop over the "months" of the orbit
#file.write(str(x)+'\n')			    # since return breaks the loop, so manually changed month value (m in lens_equation)
#file.close()						    # and recorded the output of each by appending to a file... very primitive and ugly :(

months = np.arange(0, 12, 1) # 12 months in a year, spanning from 0 to 11 numerically

xprimes = 1 + np.cos(months*np.pi/6) # locations of all the x primes for each "month" in the orbit
xs = np.loadtxt('lens_equation_pseudo_isothermal.dat') # loading x values saved using ugly method described a few lines prior
for idx in range(len(xs)):
	x1, y1 = [0, xs[idx]], [2, 1] # plot a line from the source to the location where ray hits on lens plane 
	x2, y2 = [xs[idx], xprimes[idx]], [1, 0] # plot line from where ray hits on lens plane to where observer views same ray
	plt.plot(x1, y1, x2, y2)
plt.ylabel('Arbitrary distance offset')
plt.xlabel('Observer location (AU)')
plt.title('Pseudo-isothermal sphere ray-tracing diagram')
plt.show()

# --- End of Task 4

# --- Start of Task 5 ---
#
# See lin_interp.py
#
# --- End of Task 5 ---

# --- Start of Task 6 ---

import lin_interp as linterp
import numpy as np

xs, ys = np.loadtxt('lens_density.txt', unpack=True, delimiter=',') # load training data into x and y arrays

plt.scatter(xs, ys, color='k', label='Training data') # plot training data, color black to differentiate from interpolated data

for idx in range(len(xs)-1): # length of data points - 1 since there are n-1 consecutive pairs of n data points
	interp = linterp.linear_interpolator(xs[idx], xs[idx+1], ys[idx], ys[idx+1]) # train interpolator using (x0, y0), (x1, y1)
	xp = (xs[idx]+xs[idx+1])/2. # take the mean of x0 and x1 points to get the halfway value
	try:
		xps = np.append(xps, xp) # append halfway values to an array for plotting later
		interps = np.append(interps, interp(xp)) # append interpolated values to an array for plotting later
	except NameError:
		xps = xp # if array not yet created, use first value as the starting point
		interps = interp(xp) 

plt.scatter(xps, interps, color='r', label='Interpolated data') # plot interpolated data at interpolated points
plt.legend()
plt.title('Interpolation of column density as a function of position in lens plane')
plt.xlabel('Position in lens plane')
plt.ylabel('Column Density N_e')
plt.show()