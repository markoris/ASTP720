def linear_interpolator(x0, x1, y0, y1):
	return lambda xp: y0*(1-(xp-x0)/(x1-x0)) + y1*(xp-x0)/(x1-x0)
