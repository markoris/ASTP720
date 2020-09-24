import sys
import odes
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=26)
plt.rc('lines', linewidth=3)

# --- Start of Task 1 ---

def hydro_eq_wd(r, variables):
	G = 6.67e-8 # cgs units
	P, M_enc = variables
	if P < 0:return np.array([0]) # weird behavior due to large time steps around rho = 1, force early return when P < 0
	rho = 2.*(P/(1.e13))**(0.6) # white dwarf pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

# looping over densities to create M-R diagram
for rho in np.logspace(4, 6, 10):

	radii = np.linspace(1e-2, 6e9+1e-2, 10) # cm

	rs, ys = odes.rk4(hydro_eq_wd, radii, np.array([5.4e9*((rho))**(5./3), 1e-12]), dt=1e6)

	plt.scatter(rs[-1]/1.e5, ys[-1]/2e33, s=500, label='rho = %1.2e' % rho)

plt.title('White dwarf mass-radius curve')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M_sun)')
plt.legend(fontsize='small')
plt.show()

# --- End of Task 1 ---

# --- Start of Task 2 ---

def hydro_eq_ns(r, variables):
	G = 6.67e-8 # cgs units
	c = 2.99e10 # cm/s
	P, M_enc = variables
	if P < 0: return np.array([0]) # weird behavior due to large time steps around rho = 1, force early return when P < 0
	rho = (P/(5.4e9))**(0.6) # neutron star pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2)*(1+P/(rho*c**2))*(1+4.*np.pi*r**3*P/(M_enc*c**2))*(1-2*G*M_enc/(r*c**2))**(-1), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

for rho in np.logspace(14, 16, 10):

	radii = np.linspace(1e-2, 1e9+1e-2, 10) # cm

	rs, ys = odes.rk4(hydro_eq_ns, radii, np.array([5.4e9*((rho))**(5./3), 1e-12]), dt=1e3)

	plt.scatter(rs[-1]/1.e5, ys[-1]/2.e33, s=500, label='rho = %1.2e' % rho)

plt.title('Neutron star mass-radius curve')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M_sun)')
plt.legend(fontsize='small')
plt.show()

# --- End of Task 2 ---
#
# --- Start of Task 3 ---

def hydro_eq_tov(r, variables):
	G = 6.67e-8 # cgs units
	c = 2.99e10 # cm/s
	P, M_enc = variables
	if P < 0: return np.array([0]) # weird behavior due to large time steps around rho = 1, force early return when P < 0
	rho = (P/(5.4e9))**(0.6) # neutron star pressure-density relation
	dvars_dt = np.array([-(G*M_enc*rho)/(r**2)*(1+P/(rho*c**2))*(1+4.*np.pi*r**3*P/(M_enc*c**2))*(1-2*G*M_enc/(r*c**2))**(-1), 4.*np.pi*(r**2)*rho]) 
	return dvars_dt

radii = np.linspace(1e-2, 1.302e6+1e-2, 10) # cm, going out to 13.02 km to get the mass of the neutron star

rs, ys = odes.rk4(hydro_eq_tov, radii, np.array([5.4e9*((1e15))**(5./3), 1e-12]), dt=1e3) # using density inferred from task 2 M-R curve

print(rs.shape[0], rs[-1]/1e5, ys[-1]/1.989e33)


# --- End of Task 3 ---