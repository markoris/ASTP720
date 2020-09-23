import numpy as np

class Matrix(object):

	def __init__(self, n_rows, n_cols):
		self.n_rows = int(n_rows)
		self.n_cols = int(n_cols)
		self.values = np.zeros((self.n_rows, self.n_cols)) # make matrix of just zeroes to start, populate later
		return

	def populate(self, values):

		if self.n_rows*self.n_cols != len(values): 
			print('Length of list of values does not match shape of matrix. Matrix not populated.')
			return
		for row in range(self.n_rows):
			for col in range(self.n_cols):
				self.values[row, col] = values[row*self.n_cols + col] # populates matrix row-wise by going through list of values
		return

	def add(self, other):
		if isinstance(other, int) or isinstance(other, float): # if integer or float, just add this value to every entry
			self.values += other
		elif type(other) == type(self):
			if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
				print('Invalid dimension of second array. Rows/cols of second array must match rows/cols of primary array.')
				return
			for row in range(self.n_rows):
				for col in range(self.n_cols):
					self.values[row, col] += other.values[row, col] # element-wise addition
			return

	def mult(self, other):
		if isinstance(other, int) or isinstance(other, float):
			self.values *= other
		elif type(other) == type(self):
			if self.n_cols != other.n_rows: # outer and inner dimensions need to match for multiplication
				print('Invalid dimension of second array. Rows/cols of second array must match cols/rows of primary array.')
				return
			temp = np.zeros((self.n_rows, other.n_cols)) # product will have shape of rows of A and columns of B
			for row in range(self.n_rows):
				for col in range(other.n_cols):
					for i in range(other.n_rows):
							temp[row, col] += self.values[row, i]*other.values[i, col] # populate product matrix 
			self.n_rows, self.n_cols = temp.shape # update number of rows and columns since it may not necessarily be the same post-multiplication
			self.values = temp


	def transpose(self):
		temp = np.zeros((self.n_cols, self.n_rows)) # temporary placeholder matrix used to avoid confusion
		for row in range(self.n_rows):
			for col in range(self.n_cols):
				temp[col, row] = self.values[row, col] # simply flip rows and columns
		self.n_rows, self.n_cols = temp.shape # update number of rows and columns since it may not necessarily be the same post-transpose
		self.values = temp

	def trace(self):
		if self.n_rows != self.n_cols:
			print('Matrix is not square and therefore the trace is undefined.')
			pass
		else:
			trace = 0
			for idx in range(self.n_rows):
				trace += self.values[idx, idx] # add diagonal values only, all other values unimportant
			return trace

	def ludecomp(self):
		L, U = Matrix(self.n_rows, self.n_cols), Matrix(self.n_rows, self.n_cols) # instantiate two new matrix classes

		for row in range(self.n_rows): 
			for col in range(row, self.n_cols):  

				temp = 0
				for idx in range(row): 
				    temp += L.values[row, idx] * U.values[idx, col] # sum over iteration index for given row, column
				
				U.values[row, col] = self.values[row, col] - temp # no need to divide by l_ii since diagonals of lower matrix are forced to be 1
  
			for col in range(row, self.n_cols): 

				if row == col: 
					L.values[row, col] = 1 # force that lower diagonal entries are 1

				else: 
					temp = 0
					for idx in range(row): 
					    temp += L.values[col, idx] * U.values[idx, row] # repeat same as above but flip column and row since U and L are shape-mirrored across diagonal
		
					L.values[col, row] = int((self.values[col, row] - temp) / U.values[row, row]) # divide by U values since they are not necessarily unitary

		return L, U

	def invert(self):

		if self.n_rows != self.n_cols:
			print('Matrix is not square and therefore inversion has been canceled.')
			pass
		identity = np.identity(self.n_rows)

		for row in range(self.n_rows):
			for col in range(row+1):
				if row != 0:
					identity[row:, col] -= self.values[row, col]*identity[row-1, col] # forward pass of getting matrix into row-echelon form
					self.values[row:, col] -= self.values[row, col]*self.values[row-1, col] # do the same for what started as the identity as well as the original matrix
				if row != col: 
					continue
				identity[row, :] /= self.values[row, col] # scale the first entry in the row to be 1
				self.values[row, :] /= self.values[row, col]

		for row in reversed(range(self.n_rows)): # starting from the bottom row
			for col in range(self.n_rows):
				if row >= col: continue
				if self.values[row, col] != 0: # if non-zero entries are present, clean them up so that we're left with just the identity matrix from the original input matrix
					identity[row, :] -= identity[col, :]*self.values[row, col] 
					self.values[row, :] -= self.values[col, :]*self.values[row, col]

		self.values = identity # what started as the identity matrix is now the inverse after Gaussian elimination
		pass


	def determinant(self):

		import numpy as np

		def minor(values, row_lim, col_lim):
		    return np.array([np.hstack((row[:col_lim], row[col_lim+1:])) for row in np.vstack((values[:row_lim], values[row_lim+1:]))])
		
		def recursive_det(values):

			if values.shape[0] == 2:
			    return values[0, 0]*values[1, 1] - values[0, 1]*values[1, 0]
			
			det = 0
			for col in range(values.shape[0]):
			    det += ((-1)**col)*values[0, col]*recursive_det(minor(values, 0, col))
			return det

		return recursive_det(self.values)
