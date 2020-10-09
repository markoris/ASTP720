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

		return

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

	def return_leaves(self, leaves):
		out = leaves
		if self.leaf == True:
#			print(self.particles)
			return np.append(leaves, self)
		if len(self.children) == 0:
			pass
		for child in self.children:
			out = np.append(out, child.return_leaves(leaves))
		return out

	def accel_calc(self):
		leaves = self.return_leaves(np.array([]))
		G = 4.301e-3 * (3.241e-14*3.154e7)**2 # pc M_sun^-1 pc/yr
		eps = 1e-3 # Plummer radius for force softening
		for alpha in range(leaves.shape[0]):
			ax = 0
			ay = 0
			for beta in range(leaves.shape[0]):
				if alpha == beta: continue
				r = (leaves[beta].particles[0] - leaves[alpha].particles[0]) * 1e6 #* 3.086e13 # Mpc to pc to km
				ax += G * leaves[beta].mass * (-1/np.sqrt(np.abs(r[0] + eps)))*(r[0])/(np.abs(r[0])**2)
				ay += G * leaves[beta].mass * (-1/np.sqrt(np.abs(r[1] + eps)))*(r[1])/(np.abs(r[1])**2)
				leaves[alpha].accel = np.array([ax, ay])
		return leaves

def verlet(x0, x1, y0, y1, t, nodes):
	x_next = np.zeros_like(x0)
	y_next = np.zeros_like(y0)
	for node in range(len(x0)):
		x_next[node] = 2*x1[node] - x0[node] + t**2*nodes[node].accel[0]
		y_next[node] = 2*y1[node] - y0[node] + t**2*nodes[node].accel[1]
	return x_next, y_next

coords0 = np.load('galaxies0.npy')
coords1 = np.load('galaxies1.npy')

x0, y0 = coords0[:, 0], coords0[:, 1] # units of Mpc
x1, y1 = coords1[:, 0], coords1[:, 1]

plt.scatter(x0, y0)
plt.savefig('figures/step1.png')
plt.close()

plt.scatter(x1, y1)
plt.savefig('figures/step2.png')
plt.close()

xs = np.array([x0, x1])
ys = np.array([y0, y1])

for _ in range(10):

	box = Node(0, 10, 0, 10, 0, np.array([xs[-1, :], ys[-1, :]]).T)

	box.subgrid()

	leaves = box.accel_calc()

	x_next, y_next = verlet(xs[-2, :], xs[-1, :], ys[-2, :], ys[-1, :], 1000, leaves)

	idxs = np.where((x_next < 0) | (x_next > 10) | (y_next < 0) | (y_next > 10))

	xs = np.delete(xs, idxs, axis=1)
	ys = np.delete(ys, idxs, axis=1)
	x_next = np.delete(x_next, idxs)
	y_next = np.delete(y_next, idxs)

	xs = np.append(xs, x_next[None, :], axis=0)
	ys = np.append(ys, y_next[None, :], axis=0)

	print(xs.shape)

	plt.scatter(xs[-1, :], ys[-1, :])
	plt.savefig('figures/step%d.png' % xs.shape[0])
	plt.close()

# things are working, but i keep losing particles :(