import numpy as np
import matplotlib.pyplot as plt


class Node:
	def __init__(self, xmin, xmax, ymin, ymax, divs, particles):

		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.divs = divs
		self.particles = particles
		self.children = []
		self.leaf = False
		self.mass = 1e12
		self.accel = 0

		if self.particles.shape[0] == 0:
			x_com, y_com = 0, 0

		else:
			x_com = np.sum(self.particles[:, 0])/(self.particles.shape[0])
			y_com = np.sum(self.particles[:, 1])/(self.particles.shape[0])

		self.com = np.array([x_com, y_com])

	def subgrid(self):
		'''
		sub-divides the initial 10 x 10 Mpc box until there is only one particle per node
		'''

		if self.particles.shape[0] == 0:
			return

		if self.particles.shape[0] == 1:
			self.leaf = True
			return

		# northwest node

		nw_divs = self.divs+1
		nw_xmin = self.xmin
		nw_xmax = np.mean([self.xmin, self.xmax])
		nw_ymin = np.mean([self.ymin, self.ymax])
		nw_ymax = self.ymax
		nw_idxs_x = np.logical_and(self.particles[:, 0] < nw_xmax, self.particles[:, 0] > nw_xmin)
		nw_idxs_y = np.logical_and(self.particles[:, 1] < nw_ymax, self.particles[:, 1] > nw_ymin)
		nw_idxs = np.logical_and(nw_idxs_x, nw_idxs_y)
		nw_particles = self.particles[nw_idxs, :]
		nw = Node(nw_xmin, nw_xmax, nw_ymin, nw_ymax, nw_divs, nw_particles)
		self.children.append(nw)

		# northeast node

		ne_divs = self.divs+1
		ne_xmin = np.mean([self.xmin, self.xmax])
		ne_xmax = self.xmax
		ne_ymin = np.mean([self.ymin, self.ymax])
		ne_ymax = self.ymax
		ne_idxs_x = np.logical_and(self.particles[:, 0] < ne_xmax, self.particles[:, 0] > ne_xmin)
		ne_idxs_y = np.logical_and(self.particles[:, 1] < ne_ymax, self.particles[:, 1] > ne_ymin)
		ne_idxs = np.logical_and(ne_idxs_x, ne_idxs_y)
		ne_particles = self.particles[ne_idxs, :]
		ne = Node(ne_xmin, ne_xmax, ne_ymin, ne_ymax, ne_divs, ne_particles)
		self.children.append(ne)

		# southeast node

		se_divs = self.divs+1
		se_xmin = np.mean([self.xmin, self.xmax])
		se_xmax = self.xmax
		se_ymin = self.ymin
		se_ymax = np.mean([self.ymin, self.ymax])
		se_idxs_x = np.logical_and(self.particles[:, 0] < se_xmax, self.particles[:, 0] > se_xmin)
		se_idxs_y = np.logical_and(self.particles[:, 1] < se_ymax, self.particles[:, 1] > se_ymin)
		se_idxs = np.logical_and(se_idxs_x, se_idxs_y)
		se_particles = self.particles[se_idxs, :]
		se = Node(se_xmin, se_xmax, se_ymin, se_ymax, se_divs, se_particles)
		self.children.append(se)

		# southwest node

		sw_divs = self.divs+1
		sw_xmin = self.xmin
		sw_xmax = np.mean([self.xmin, self.xmax])
		sw_ymin = self.ymin
		sw_ymax = np.mean([self.ymin, self.ymax])
		sw_idxs_x = np.logical_and(self.particles[:, 0] < sw_xmax, self.particles[:, 0] > sw_xmin)
		sw_idxs_y = np.logical_and(self.particles[:, 1] < sw_ymax, self.particles[:, 1] > sw_ymin)
		sw_idxs = np.logical_and(sw_idxs_x, sw_idxs_y)
		sw_particles = self.particles[sw_idxs, :]
		sw = Node(sw_xmin, sw_xmax, sw_ymin, sw_ymax, sw_divs, sw_particles)
		self.children.append(sw)
		
		nw.subgrid()
		ne.subgrid()
		se.subgrid()
		sw.subgrid()

		return

	def find_leaves(self, n):
		'''
		returns the number of leaves, starting with n=0
		'''
		out = n
		if self.leaf == True:
			return n+1
		if len(self.children) == 0:
			pass
		for child in self.children:
			out += child.find_leaves(n)
		return out

	def return_leaves(self, leaves):
		'''
		returns an array of all the leaf nodes
		'''
		out = leaves
		if self.leaf == True:
			return np.append(leaves, self)
		if len(self.children) == 0:
			pass
		for child in self.children:
			out = np.append(out, child.return_leaves(leaves))
		return out

	def accel_calc(self):
		'''
		calculate acceleration in units of pc/yr^2
		'''
		leaves = self.return_leaves(np.array([]))
		G = 4.301e-3 * (3.241e-14*3.154e7)**2 # pc M_sun^-1 (pc/yr)^2
		eps = 1e-3 # "Plummer" radius for force softening
		for alpha in range(leaves.shape[0]): # particle being considered
			ax = 0
			ay = 0
			for beta in range(leaves.shape[0]): # all other particles
				if alpha == beta: continue
				r = (leaves[beta].particles[0] - leaves[alpha].particles[0]) * 1e6 #* 3.086e13 # Mpc to pc
				ax += G * leaves[beta].mass * (-1/np.abs(r[0]**2 + eps**2))*(r[0])/(np.abs(r[0])**2)
				ay += G * leaves[beta].mass * (-1/np.abs(r[1]**2 + eps**2))*(r[1])/(np.abs(r[1])**2)
				leaves[alpha].accel = np.array([ax, ay])
		return leaves
