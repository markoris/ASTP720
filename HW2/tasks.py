import numpy as np
import num_calc as nc
import matplotlib.pyplot as plt

plt.rc('font', size=30)
plt.rc('lines', linewidth=3)

# --- Start of Task 1 ---
# See num_calc.py 
# --- End of Task 1 ---

# --- Start of Task 2 ---

def v_c_sq(r, r200, v200, c): # v_c squared function
	import numpy as np
	x = r/r200
	return v200**2/x*(np.log(1 + c*x)-c*x/(1+c*x))/(np.log(1+c)-c/(1+c))

def m_enc(r):
	r200 = 230e3 # 230 kpc base case
	v200 = 1 # 160 km/s base case
	c = 15
	G = 4.30091e-3 # pc (km/s)^2 M_sun^-1
	return r*v_c_sq(r, r200, v200, c)/G

rs = np.linspace(1, 300e3, 1000)

# M_enc(r)
plt.figure(figsize=(19.2, 10.8))
plt.title("Enclosed mass as a function of radius")
plt.xlabel('Distance from center of galaxy (pc)')
plt.ylabel('Enclosed mass (M_sun)')
plt.plot(rs, m_enc(rs))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()

# M_total at r=300kpc

print('Total mass at 300 kpc = %e Msun.' % (m_enc(300e3))) # total mass in solar masses, assuming 300 kpc distance limit as in hw sample figures

# M(r) mass profile

for idx in range(len(rs)-1): # takes mass between r_i+1 and r_i-1 where r_i+1 is r+dr and r_i-1 is r-dr
	if idx == 0 or idx == len(rs): continue
	m = m_enc(rs[idx+1])-m_enc(rs[idx-1])
	try:
		ms = np.append(ms, m)
	except NameError:
		ms = m

plt.title('Mass profile (mass contained between subsequent distance values)')
plt.xlabel('Distance from center of galaxy (pc)')
plt.ylabel('Mass contained in radial bin')
plt.plot(rs[1:-1], ms)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()

# dM(r)/dr
plt.title('Mass derivative using same radial bins as in the mass profile')
plt.xlabel('Distance from center of galaxy (pc)')
plt.ylabel('Mass contained in radial bin')
plt.plot(rs[1:-1], nc.symm_deriv(m_enc, rs))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()

# --- End of Task 2 ---

# --- Start of Task 3 ---

# See mat.py

# --- End of Task 3 ---