def bisection(func, lower, upper, eps=1e-8, verbose=False, iters=0):
	'''
	Bisection method root-finding algorithm.

	func : function object
		The function whose root is being found.
	lower : float
		Lower limit for the initial root search.
	upper: float
		Upper limit for the initial root search.
	eps: float
		Threshold under which root-finding ceases if |func(root)| < eps.
	verbose: bool
		Flag which dictates whether number of iterations is printed to stdout and returned.
	iters: int
		Number of algorithm iterations required to find the root to within the threshold value.
	'''
	func_lower, func_upper = func(lower), func(upper)
	root = (lower+upper)/2.
	func_root = func(root)

	if abs(func_root) < eps: # if within the threshold, stop searching for new roots
		if verbose: 
			print("Root found at %.16f after %d iterations." % (root, iters))
			return root, iters
		return root

	iters += 1
	if func_lower*func_root < 0: return bisection(func, lower, root, eps, verbose, iters) # if f(a)f(c) < 0, then root must be between a and c
	else: return bisection(func, root, upper, eps, verbose, iters) # otherwise, root must be between c and b

def newton(func, func_deriv, root, eps=1e-8, verbose=False, iters=0):
	'''
	Newton's method root-finding algorithm.

	func : function object
		The function whose root is being found.
	func_deriv : function object
		The functional form of the derivative of func.
	root : float
		Initial guess for the root value.
	eps: float
		Threshold under which root-finding ceases if |func(root)| < eps.
	verbose: bool
		Flag which dictates whether number of iterations is printed to stdout and returned.
	iters: int
		Number of algorithm iterations required to find the root to within the threshold value.
	'''
	func_root = func(root)
	if abs(func_root) < eps: # if within the threshold, stop searching for new roots
		if verbose: 
			print("Root found at %.16f after %d iterations." % (root, iters))
			return root, iters
		return root
#	else:
	next_root = lambda x: x - float(func(x))/float(func_deriv(x)) # function finding the next guess of where the root might be according to Newton's method
	iters += 1
	return newton(func, func_deriv, next_root(root), eps, verbose, iters)

def secant(func, lower, upper, eps=1e-8, verbose=False, iters=0):
	'''
	Secant method root-finding algorithm.

	func : function object
		The function whose root is being found.
	lower : float
		Lower value for the initial secant.
	upper: float
		Upper value for the initial secant.
	eps: float
		Threshold under which root-finding ceases if |func(root)| < eps.
	verbose: bool
		Flag which dictates whether number of iterations is printed to stdout and returned.
	iters: int
		Number of algorithm iterations required to find the root to within the threshold value.
	'''
	func_lower, func_upper = func(lower), func(upper) # finding lower and upper bound function values
	next_root = lambda x_im1, x_i: x_i - func(x_i)*(x_i - x_im1)/float(func(x_i)-func(x_im1)) # function finding the next guess of where the root might be according to secant method
	root = next_root(lower, upper)
	func_root = func(root)
	if abs(func_root) < eps: # if within the threshold, stop searching for new roots
		if verbose: 
			print("Root found at %.16f after %d iterations." % (root, iters))
			return root, iters
		return root
#	else:
	iters += 1
	return secant(func, upper, root, eps, verbose, iters)