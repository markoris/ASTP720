# --- Start of Task 2 ---
import root_finding_algos as rfa
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', size=30)

#def pseudo_iso_sphere_unitless(x):
#	'''
#	Full-width half-maximum of pseudo-isothermal sphere with x in units of r_c.
#	'''
#	return (1. + x**2)**(-0.5) - 0.5
#
#def pseudo_iso_sphere_unitless_deriv(x):
#	'''
#	Derivative of full-width half-maximum of pseudo-isothermal sphere with x in units of r_c.
#	'''
#	return -x/(1 + x**2)**(1.5)
#
##root_bisect, _ = rfa.bisection(pseudo_iso_sphere_unitless, 0, 2, verbose=True)
##root_newton, _ = rfa.newton(pseudo_iso_sphere_unitless, pseudo_iso_sphere_unitless_deriv, 0.5, verbose=True)
##root_secant, _ = rfa.secant(pseudo_iso_sphere_unitless, 0, 2, verbose=True)
#
#thresholds = np.logspace(-1, -10, 10)
#
#for threshold in thresholds:
#	root_bisect, iters_bisect = rfa.bisection(pseudo_iso_sphere_unitless, 0, 2, eps=threshold, verbose=True)
#	root_newton, iters_newton = rfa.newton(pseudo_iso_sphere_unitless, pseudo_iso_sphere_unitless_deriv, 0.5, eps=threshold, verbose=True)
#	root_secant, iters_secant = rfa.secant(pseudo_iso_sphere_unitless, 0, 2, eps=threshold, verbose=True)
#	try:
#		iters_b = np.append(iters_b, iters_bisect)
#		iters_n = np.append(iters_n, iters_newton)
#		iters_s = np.append(iters_s, iters_secant)
#	except NameError:
#		iters_b = iters_bisect
#		iters_n = iters_newton
#		iters_s = iters_secant
#plt.scatter(np.flip(thresholds), iters_b, color='r', label='Bisection')
#plt.scatter(np.flip(thresholds), iters_n, color='b', label='Newton')
#plt.scatter(np.flip(thresholds), iters_s, color='g', label='Secant')
#plt.legend()
#plt.xlabel('Threshold')
#plt.ylabel('Iterations until sub-threshold')
#plt.xscale('log')
#plt.show()

# --- End of Task 2 ---

# --- Start of Task 3 ---

#def lens_equation(x):
#	m = 11
#	D = 3.086e21 # 1 kpc in centimeters
#	a = 1.496e13 # 1 AU in centimeters
#	wavelength = 21 # centimeters
#	N_o = 0.01*3.086e18 # 0.01 pc cm^-3 as cm^-2
#	r_e = (4.803e-10)**2/(9.1094e-28)/(2.998e10)**2 # classical radius e^2/m_e*c*2 in cgs units
#	return x*(1 + (wavelength**2*r_e*N_o*D)/(np.pi*a**2)*np.exp(-(x/a)**2)) - (1 + np.cos(m*np.pi/6))
#
#x = rfa.secant(lens_equation, -1, 1)
#
##file = open('lens_equation.dat', 'a')
##file.write(str(x)+'\n')
##file.close()
#
#months = np.arange(0, 12, 1)
#
#xprimes = 1 + np.cos(months*np.pi/6)
#xs = np.loadtxt('lens_equation.dat')
#for idx in range(len(xs)):
#	x1, y1 = [0, xs[idx]], [2, 1]
#	x2, y2 = [xs[idx], xprimes[idx]], [1, 0]
#	plt.plot(x1, y1, x2, y2)
#plt.ylabel('Arbitrary distance offset')
#plt.xlabel('Observer location (AU)')
#plt.title('Gaussian Lens ray-tracing diagram')
#plt.show()

# --- End of Task 3 ---

# --- Start of Task 4 ---

# --- End of Task 4

# --- Start of Task 6 ---

#import lin_interp as linterp
#import numpy as np
#
#xs, ys = np.loadtxt('lens_density.txt', unpack=True, delimiter=',')
#
#plt.scatter(xs, ys, color='k', label='Training data')
#
#for idx in range(len(xs)-1):
#	interp = linterp.linear_interpolator(xs[idx], xs[idx+1], ys[idx], ys[idx+1])
#	xp = (xs[idx]+xs[idx+1])/2.
#	try:
#		xps = np.append(xps, xp)
#		interps = np.append(interps, interp(xp))
#	except NameError:
#		xps = xp
#		interps = interp(xp)
#
#plt.scatter(xps, interps, color='r', label='Interpolated data')
#plt.legend()
#plt.title('Interpolation of column density as a function of position in lens plane')
#plt.xlabel('Position in lens plane')
#plt.ylabel('Column Density N_e')
#plt.show()