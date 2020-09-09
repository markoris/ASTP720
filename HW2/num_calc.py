def num_deriv(func, xs):
	'''
	Symmetric numerical derivative of form (y_{i+1} - y{i-1})/(x_{i+1}-x_{i-1}), where the spacing
	between sequential x values need not be uniform. Returns a derivative array of n-2 values, where
	no derivative is calculated for the first and final points due to the three-point requirement of the method.
	'''
	idxs = range(len(xs))
	for idx in idxs:
		if idx==0 or idx==idxs[-1]: continue
		deriv = (func(xs[idx+1])-func(xs[idx-1]))/(xs[idx+1]-xs[idx-1])
		try:
			derivs.append(deriv)
		except NameError:
			derivs = [deriv]

	return derivs

def riemann_integral(func, xs, left=False, right=False, midpoint=True):
	'''
	words
	'''
	if left or right: midpoint=False
	idxs = range(len(xs)-1)
	integral = 0
	for idx in idxs:
		if left: integral += (xs[idx+1]-xs[idx])*func(xs[idx])
		if right: integral += (xs[idx+1]-xs[idx])*func(xs[idx+1])
		if midpoint: integral += (xs[idx+1]-xs[idx])*func((xs[idx+1]+xs[idx])/2.)
	return integral

def trap_integral(func, xs):
	'''
	words
	'''
	idxs = range(len(xs)-1)
	integral = 0
	for idx in idxs:
		integral += (func(xs[idx])+func(xs[idx+1]))/2.*(xs[idx+1]-xs[idx])
	return integral

def simp_integral(func, xs):
	import math # need ceiling function to get proper number of iterations
	idxs = range(math.ceil(len(xs)/2.-1)) # if len(xs) is odd, this undershoots num iterations unless ceil is used
	integral = 0
	for idx in idxs:
		integral += (xs[idx+2]-xs[idx])/6.*(func(xs[2*idx]) + 4.*func(xs[(2*idx)+1]) + func(xs[(2*idx)+2]))
		# xs[idx+2]-xs[idx] counts as two intervals, meaning we need to divide by 6, not 3
	if len(xs) % 2 == 0: integral += (xs[-1]-xs[-3])/24.*(-func(xs[-3]) + 8*func(xs[-2]) + 5*func(xs[-1])) # correction term for if even number of x-points
	# again, xs[-1]-xs[-3] is two intervals, so divide by 24, not 12, add correction term if number of x-points is even only
	return integral