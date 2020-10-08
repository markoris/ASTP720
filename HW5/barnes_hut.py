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

	def subgrid(self):

		if self.particles.shape[0] == 0:
			return

		if self.particles.shape[0] == 1:
			self.leaf = True
			return

		#if self.divs == 2: return

		nw_divs = self.divs+1
		nw_xmin = self.xmin
#		nw_xmax = self.xmax - self.xmax/(2.)
		nw_xmax = np.mean([self.xmin, self.xmax])
#		nw_ymin = self.ymax - self.ymax/(2.**nw_divs)
		nw_ymin = np.mean([self.ymin, self.ymax])
		nw_ymax = self.ymax
		nw_idxs_x = np.logical_and(self.particles[:, 0] < nw_xmax, self.particles[:, 0] > nw_xmin)
		nw_idxs_y = np.logical_and(self.particles[:, 1] < nw_ymax, self.particles[:, 1] > nw_ymin)
		nw_idxs = np.logical_and(nw_idxs_x, nw_idxs_y)
		nw_particles = self.particles[nw_idxs, :]
#		print(nw_particles.shape)
		nw = Node(nw_xmin, nw_xmax, nw_ymin, nw_ymax, nw_divs, nw_particles)
		self.children.append(nw)
#		print('nw', nw_xmin, nw_xmax, nw_ymin, nw_ymax)

		ne_divs = self.divs+1
#		ne_xmin = self.xmax - self.xmax/(2.)
		ne_xmin = np.mean([self.xmin, self.xmax])
		ne_xmax = self.xmax
#		ne_ymin = self.ymax - self.ymax/(2.**ne_divs)
		ne_ymin = np.mean([self.ymin, self.ymax])
		ne_ymax = self.ymax
		ne_idxs_x = np.logical_and(self.particles[:, 0] < ne_xmax, self.particles[:, 0] > ne_xmin)
		ne_idxs_y = np.logical_and(self.particles[:, 1] < ne_ymax, self.particles[:, 1] > ne_ymin)
		ne_idxs = np.logical_and(ne_idxs_x, ne_idxs_y)
		ne_particles = self.particles[ne_idxs, :]
#		print(ne_particles.shape)
		ne = Node(ne_xmin, ne_xmax, ne_ymin, ne_ymax, ne_divs, ne_particles)
		self.children.append(ne)
#		print('ne', ne_xmin, ne_xmax, ne_ymin, ne_ymax)

		se_divs = self.divs+1
#		se_xmin = self.xmax - self.xmax/(2.)
		se_xmin = np.mean([self.xmin, self.xmax])
		se_xmax = self.xmax
		se_ymin = self.ymin
		se_ymax = np.mean([self.ymin, self.ymax])
#		se_ymax = self.ymax - self.ymax/(2.**se_divs)
		se_idxs_x = np.logical_and(self.particles[:, 0] < se_xmax, self.particles[:, 0] > se_xmin)
		se_idxs_y = np.logical_and(self.particles[:, 1] < se_ymax, self.particles[:, 1] > se_ymin)
		se_idxs = np.logical_and(se_idxs_x, se_idxs_y)
		se_particles = self.particles[se_idxs, :]
#		print(se_particles.shape)
		se = Node(se_xmin, se_xmax, se_ymin, se_ymax, se_divs, se_particles)
		self.children.append(se)
#		print('se', se_xmin, se_xmax, se_ymin, se_ymax)

		sw_divs = self.divs+1
		sw_xmin = self.xmin
#		sw_xmax = self.xmax - self.xmax/(2.)
		sw_xmax = np.mean([self.xmin, self.xmax])
		sw_ymin = self.ymin
		sw_ymax = np.mean([self.ymin, self.ymax])
#		sw_ymax = self.ymax - self.ymax/(2.**sw_divs)
		sw_idxs_x = np.logical_and(self.particles[:, 0] < sw_xmax, self.particles[:, 0] > sw_xmin)
		sw_idxs_y = np.logical_and(self.particles[:, 1] < sw_ymax, self.particles[:, 1] > sw_ymin)
		sw_idxs = np.logical_and(sw_idxs_x, sw_idxs_y)
		sw_particles = self.particles[sw_idxs, :]
#		print(sw_particles.shape)
		sw = Node(sw_xmin, sw_xmax, sw_ymin, sw_ymax, sw_divs, sw_particles)
		self.children.append(sw)
#		print('sw', sw_xmin, sw_xmax, sw_ymin, sw_ymax)
		
		nw.subgrid()
		ne.subgrid()
		se.subgrid()
		sw.subgrid()

	def find_leaves(self, n):
		out = n
		if self.leaf == True:
#			print(self.particles)
			return n+1
		if len(self.children) == 0:
			pass
		for child in self.children:
			out += child.find_leaves(n)
		return out


coords = np.load('galaxies0.npy')

x0, y0 = coords[:, 0], coords[:, 1] # units of Mpc

print(x0.shape)

box = Node(0, 10, 0, 10, 0, np.array([x0, y0]).T)

box.subgrid()

out = box.find_leaves(0)
print(out)

# start with simulation box

#print(x0[:10], y0[:10])

#xiter1 = x0[np.logical_and(x0<xmin, y0<ymin)] # this is how to decide when to stop making further nodes
#yiter1 = y0[np.logical_and(x0<xmin, y0<ymin)] # if yields 0, stop making nodes until yields 1

# cut into 4s (8s?)
#sb1 = Node("sb1", parent=box, M=1e12, x=7.5, y=7.5) # this will be reformatted to produce child nodes based on how many particles in any given box
