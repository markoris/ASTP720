import sys
#sys.path.append('../HW3/')
import odes
import numpy as np
import matplotlib.pyplot as plt

# --- Start of Task 1 ---

def hydro_eq_wd(r, variables):
	G = 6.67e-8 # cgs units
	P, M_enc = variables
	if P < 0:return np.array([0]) # weird behavior due to large time steps around rho = 1, forcing P=0 stops nans
	rho = 2.*(P/(1.e13))**(0.6) # white dwarf pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

#radii = np.linspace(1e-2, 1.9e9+1e-2, 10) # cm

#rs, ys = odes.rk4(hydro_eq_wd, radii, np.array([1e13*((1e4)/2.)**(5./3), 1e-6]), dt=1e6)

for rho in np.logspace(4, 6, 10):

	radii = np.linspace(1e-2, 6e9+1e-2, 10) # cm

	rs, ys = odes.rk4(hydro_eq_wd, radii, np.array([5.4e9*((rho))**(5./3), 1e-12]), dt=1e6)
	#print(rs.shape)

	plt.scatter(rs[-1]/1e5, ys[-1]/2e33)

	#plt.plot(rs, ys[np.linspace(0, 2598, 1300).astype(int)])
	#plt.plot(rs, ys[np.linspace(1, 2599, 1300).astype(int)])
	#plt.show()

plt.show()

#plt.plot(rs, ys[np.linspace(0, 19998, 10000).astype(int)])
#plt.plot(rs, ys[np.linspace(1, 19999, 10000).astype(int)])
#plt.show()

# --- End of Task 1 ---

# --- Start of Task 2 ---

def hydro_eq_ns(r, variables):
	G = 6.67e-8 # cgs units
	c = 2.99e10 # cm/s
	P, M_enc = variables
	if P < 0: return np.array([0]) # weird behavior due to large time steps around rho = 1, forcing P=0 stop nans
	rho = (P/(5.4e9))**(0.6) # white dwarf pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2)*(1+P/(rho*c**2))*(1+4.*np.pi*r**3*P/(M_enc*c**2))*(1-2*G*M_enc/(r*c**2))**(-1), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

for rho in np.logspace(14, 16, 10):

	radii = np.linspace(1e-2, 1e9+1e-2, 10) # cm

	rs, ys = odes.rk4(hydro_eq_ns, radii, np.array([5.4e9*((rho))**(5./3), 1e-12]), dt=1e3)
	#print(rs.shape)

	plt.scatter(rs[-1]/1e5, ys[-1]/2e33, label='%1.2e' % rho)

	#plt.plot(rs, ys[np.linspace(0, 2598, 1300).astype(int)])
	#plt.plot(rs, ys[np.linspace(1, 2599, 1300).astype(int)])
	#plt.show()
plt.legend()
plt.show()

#radii = np.linspace(1e-2, 2e6+1e-2, 10) # cm
#
#rs, ys = odes.rk4(hydro_eq_ns, radii, np.array([5.4e9*((1e15))**(5./3), 1e-6]), dt=1e3)
#
#print(rs.shape, ys.shape)

#plt.plot(rs, ys[np.linspace(0, 3998, 2000).astype(int)])
#plt.plot(rs, ys[np.linspace(1, 3999, 2000).astype(int)])
#plt.show()

# --- End of Task 2 ---
#
# --- Start of Task 3 ---

def hydro_eq_tov(r, variables):
	G = 6.67e-8 # cgs units
	c = 2.99e10 # cm/s
	P, M_enc = variables
	if P < 0: return np.array([0])
#		P = 1e-10 # weird behavior due to large time steps around rho = 1, forcing P=0 stop nans
	rho = (P/(5.4e9))**(0.6) # white dwarf pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2)*(1+P/(rho*c**2))*(1+4.*np.pi*r**3*P/(M_enc*c**2))*(1-2*G*M_enc/(r*c**2))**(-1), 4.*np.pi*(r**2)*rho]) 
# bottom pressure derivative is when you include terms of O(c^2) which actually recovers a mass close to what was found 
#	dvars_dt = np.array([-(G*M_enc*rho)/(r**2)*(1 + P/(rho*c**2) + 4.*np.pi*r**3*P/(M_enc*c**2) + 2*G*M_enc/(r*c**2)), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

for rho in np.logspace(14, 16, 10):

	radii = np.linspace(1e-2, 1e9+1e-2, 10) # cm

	rs, ys = odes.rk4(hydro_eq_tov, radii, np.array([5.4e9*((rho))**(5./3), 1e-12]), dt=1e3)
	#print(rs.shape)

	plt.scatter(rs[-1]/1e5, ys[-1]/2e33)

	#plt.plot(rs, ys[np.linspace(0, 2598, 1300).astype(int)])
	#plt.plot(rs, ys[np.linspace(1, 2599, 1300).astype(int)])
	#plt.show()

plt.show()

#print(ys.shape)

#print(ys[-1]/2e33)


# --- End of Task 3 ---