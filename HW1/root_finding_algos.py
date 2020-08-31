def bisection(func, lower, upper, eps=0.1, verbose=False, iters=0):
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

	if abs(func_root) < eps:
		if verbose: 
			print("Root found at %f after %d iterations." % (root, iters))
			return root, iters
		return root

	else:
		if func_lower*func_root < 0: upper = root
		else: lower = root
		iters += 1
		bisection(func, lower, upper, eps, verbose, iters)

	