import sys
sys.path.append('../HW3/')
import odes
import numpy as np
import matplotlib.pyplot as plt

# --- Start of Task 1 ---

def hydro_eq(r, variables):
	G = 6.67e-8 # cgs units
	P, M_enc = variables
	if P < 0: P = 0 # weird behavior due to large time steps around rho = 1, forcing P=0 stop nans
	rho = 2.*(P/(1.e13))**(0.6) # white dwarf pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

radii = np.linspace(1e-2, 1e10+1e-2, 10) # cm

rs, ys = odes.rk4(hydro_eq, radii, np.array([1e13*((1e5)/2.)**(5./3), 1e-6]), dt=1e6)

#plt.plot(rs, ys[np.linspace(0, 19998, 10000).astype(int)])
#plt.plot(rs, ys[np.linspace(1, 19999, 10000).astype(int)])
#plt.show()

# --- End of Task 1 ---
