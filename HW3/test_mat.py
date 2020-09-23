import unittest
import mat
import numpy as np

class TestMatrixMethods(unittest.TestCase):

	def test_init(self):
		mtx = mat.Matrix(2, 2)
		np.testing.assert_array_equal(mtx.values, np.zeros((2, 2)))

	def test_pop(self):
		mtx = mat.Matrix(3, 3)
		mtx.populate(np.linspace(1, 9, 9))
		np.testing.assert_array_equal(mtx.values, np.linspace(1, 9, 9).reshape(3, 3))

	def test_add(self):
		mtx1 = mat.Matrix(3, 3)
		mtx2 = mat.Matrix(3, 3)
		mtx1.populate(np.linspace(1, 9, 9))
		mtx2.populate(np.linspace(10, 19, 9))
		mtx1.add(mtx2)
		np.testing.assert_array_equal(mtx1.values, np.linspace(11, 28, 9).reshape(3, 3))

	def test_mult(self):
		mtx1 = mat.Matrix(3, 3)
		mtx2 = mat.Matrix(3, 3)
		mtx1.populate(np.linspace(1, 9, 9))
		mtx2.populate(np.linspace(10, 19, 9))
		prod = np.matmul(mtx1.values, mtx2.values)
		mtx1.mult(mtx2)
		np.testing.assert_array_equal(mtx1.values, prod)

	def test_transpose(self):
		mtx = mat.Matrix(3, 3)
		mtx.populate(np.linspace(1, 9, 9))
		mtx.transpose()
		np.testing.assert_array_equal(mtx.values, np.transpose(np.linspace(1, 9, 9).reshape(3, 3)))

	def test_trace(self):
		mtx = mat.Matrix(3, 3)
		mtx.populate(np.linspace(1, 9, 9))
		np.testing.assert_array_equal(mtx.trace(), np.trace(mtx.values))

	def test_LUdecomp(self):
		import scipy.linalg
		mtx = mat.Matrix(3, 3)
		mtx.populate([2, -1, -2, -4, 6, 3, -4, -2, 8])
		l, u = mtx.ludecomp()
		l.mult(u)
		np.testing.assert_array_equal(l.values, mtx.values)

	def test_inverse(self):
		mtx = mat.Matrix(2, 2)
		mtx.populate([2, 3, 2, 2])
		check = np.linalg.inv(mtx.values)
		mtx.invert()
		np.testing.assert_array_equal(mtx.values, check)

	def test_det(self):
		mtx = mat.Matrix(3, 3)
		mtx.populate(np.linspace(1, 9, 9))
		self.assertEqual(mtx.determinant(), np.linalg.det(mtx.values))

if __name__ == '__main__':
	unittest.main()