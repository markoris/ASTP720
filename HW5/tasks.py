import barnes_hut as bh
import numpy as np
import matplotlib.pyplot as plt

def verlet(x0, x1, y0, y1, t, nodes):
	'''
	calculates the next position of the galaxy given a period of time t and the acceleration, stored as an attribute in the node
	'''
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

	box = bh.Node(0, 10, 0, 10, 0, np.array([xs[-1, :], ys[-1, :]]).T)

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
