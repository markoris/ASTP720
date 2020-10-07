import numpy as np
import matplotlib.pyplot as plt
from anytree import Node, RenderTree # python3 -m pip install anytree

coords = np.load('galaxies0.npy')

x0, y0 = coords[:, 0], coords[:, 1] # units of Mpc

# start with simulation box
box = Node("box", x=5, y=5, size=10) # start with this

xiter1 = x0[np.logical_and(x0<box.x, y0<box.y)] # this is how to decide when to stop making further nodes
yiter1 = y0[np.logical_and(x0<box.x, y0<box.y)] # if yields 0, stop making nodes until yields 1

print(xiter1.shape, yiter1.shape)

# cut into 4s (8s?)
sb1 = Node("sb1", parent=box, M=1e12, x=7.5, y=7.5) # this will be reformatted to produce child nodes based on how many particles in any given box

print(box.children)

for node in box.children:
	print(node)

print(sb1.M, sb1.x, sb1.y)
