import numpy as np
import num_calc as nc
import mat

def xsq(x):
	return x**2

xs = np.linspace(1, 10, 10)

print(nc.num_deriv(xsq, xs))
print(nc.riemann_integral(xsq, xs, left=True))
print(nc.riemann_integral(xsq, xs, right=True))
print(nc.riemann_integral(xsq, xs, midpoint=True))
print(nc.trap_integral(xsq, xs))
print(nc.simp_integral(xsq, xs))

a = mat.Matrix(2, 5)
a.transpose()
a.transpose()
a.populate(np.linspace(1, 10, 10))
b = mat.Matrix(2, 5)
a.add(b)
c = mat.Matrix(2, 5)
c.populate(np.linspace(1, 10, 10))
d = mat.Matrix(5, 2)
d.populate(np.linspace(1, 10, 10))
a.mult(d)
a.transpose()
print(a.trace())

a = mat.Matrix(3, 3)
a.populate([2, -1, -2, -4, 6, 3, -4, -2, 8])

l, u = a.ludecomp()

l.mult(u)

a = mat.Matrix(2,2)
a.populate([2, 3, 2, 2])

b = mat.Matrix(2, 2)
b.populate([2, 3, 2, 2])

b.invert()

b.mult(a)

print(b.values)

a = mat.Matrix(5,5)
a.populate(np.identity(5).flatten())

b = mat.Matrix(5, 5)
b.populate(np.identity(5).flatten())

a.invert()

b.mult(a)

print(b.values)
