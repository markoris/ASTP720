import numpy as np

class Matrix(object):

	def __init__(self, n_rows, n_cols):
		self.n_rows = int(n_rows)
		self.n_cols = int(n_cols)
		self.values = np.zeros((self.n_rows, self.n_cols))
		return

	def populate(self, values):
		'''
		Populates matrix one row at a time using values from a list
		'''
		if self.n_rows*self.n_cols != len(values): 
			print('Length of list of values does not match shape of matrix. Matrix not populated.')
			return
		for row in range(self.n_rows):
			for col in range(self.n_cols):
				self.values[row, col] = values[row*self.n_cols + col]
		return

	def add(self, other):
		if isinstance(other, int) or isinstance(other, float):
			self.values += other
		elif type(other) == type(self):
			if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
				print('Invalid dimension of second array. Rows/cols of second array must match rows/cols of primary array.')
				return
			for row in range(self.n_rows):
				for col in range(self.n_cols):
					self.values[row, col] += other.values[row, col]
			return

	def mult(self, other):
		if isinstance(other, int) or isinstance(other, float):
			self.values *= other
		elif type(other) == type(self):
			if self.n_rows != other.n_cols or self.n_cols != other.n_rows:
				print('Invalid dimension of second array. Rows/cols of second array must match cols/rows of primary array.')
				return
			temp = np.zeros((self.n_rows, other.n_cols))
			for row in range(self.n_rows):
				for col in range(other.n_cols):
					for i in range(other.n_rows):
							temp[row, col] += self.values[row, i]*other.values[i, col]
			self.n_rows, self.n_cols = temp.shape # update number of rows and columns since it may not necessarily be the same post-multiplication
			self.values = temp


	def transpose(self):
		temp = np.zeros((self.n_cols, self.n_rows))
		for row in range(self.n_rows):
			for col in range(self.n_cols):
				temp[col, row] = self.values[row, col]
		self.n_rows, self.n_cols = temp.shape # update number of rows and columns since it may not necessarily be the same post-transpose
		self.values = temp

	def trace(self):
		if self.n_rows != self.n_cols:
			print('Matrix is not square and therefore the trace is undefined.')
			return
		else:
			trace = 0
			for idx in range(self.n_rows):
				trace += self.values[idx, idx]
			return trace