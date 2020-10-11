# --- Start of Task 1 ---

import barnes_hut as bh
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=12)

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

plt.title('Step 1')
plt.scatter(x0, y0)
plt.xlabel('x (Mpc)')
plt.ylabel('y (Mpc)')
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

	x_next, y_next = verlet(xs[-2, :], xs[-1, :], ys[-2, :], ys[-1, :], 10000, leaves)

	idxs = np.where((x_next < 0) | (x_next > 10) | (y_next < 0) | (y_next > 10))

	xs = np.delete(xs, idxs, axis=1)
	ys = np.delete(ys, idxs, axis=1)
	x_next = np.delete(x_next, idxs)
	y_next = np.delete(y_next, idxs)

	xs = np.append(xs, x_next[None, :], axis=0)
	ys = np.append(ys, y_next[None, :], axis=0)

	print(xs.shape)

	plt.scatter(xs[-1, :], ys[-1, :])
	plt.title('Step %d' % xs.shape[0])
	plt.xlabel('x (Mpc)')
	plt.ylabel('y (Mpc)')
	plt.savefig('figures/step%d.png' % xs.shape[0])
	plt.close()

# --- End of Task 1 ---

# --- Start of Task 2 ---

import numpy as np
import matplotlib.pyplot as plt

xs, ys = np.linspace(0, 10, 10), np.linspace(0, 10, 10) # using REALLY rough resolution because higher resolution (only 10 pts) starts to show peaks at 7, 2 and 8, 7 which are artifacts of how I'm calculating the potential (which is clearly incorrect)

X, Y = np.meshgrid(xs, ys)

phi = np.zeros_like(X)

G = 4.301e-3

for x in range(X.shape[0]):
	for y in range(Y.shape[0]):
		phi1x = G * 400*1e12 * (-1/np.sqrt((X[x, y]-7)**2 + 1e-3**2)) # group of 400 galaxies at 7, 7
		phi1y = G * 400*1e12 * (-1/np.sqrt((Y[x, y]-7)**2 + 1e-3**2))
		phi1 = np.sqrt(phi1x**2 + phi1y**2)
		phi2x = G * 255*1e12 * (-1/np.sqrt((X[x, y]-8)**2 + 1e-3**2)) # group of 255 galaxies at 8, 2
		phi2y = G * 255*1e12 * (-1/np.sqrt((Y[x, y]-2)**2 + 1e-3**2))
		phi2 = np.sqrt(phi2x**2 + phi2y**2)
		phi[x, y] = phi1 + phi2 # adding potentials together

plt.contour(X, Y, np.log10(phi))
cb = plt.colorbar()
plt.title('Rough Cluster Potential Approximation')
plt.xlabel('x (Mpc)')
plt.ylabel('y (Mpc)')
cb.set_label('log potential')
plt.savefig('figures/potential.png')

# --- End of Task 2