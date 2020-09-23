import mat
import odes
import numpy as np
from scipy import linalg
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rc('font', size=26)
plt.rc('lines', linewidth=3)

# --- Start of Task 1 ---
#
# See test_mat.py
#
# --- End of Task 1 ---

# --- Start of Task 2 ---

l, u, Aul = np.loadtxt('A_coefficients.dat', unpack=True, delimiter=',') # load coefficients for A_ul
l, u = l.astype(int), u.astype(int) # ensure ints for indexing

coeffs_aul = mat.Matrix(9, 9)
for value in range(Aul.shape[0]):
	coeffs_aul.values[l[value]-1, u[value]-1] = Aul[value] # subtract 1 from indices to go from matrix (1, 2, ...) to Python (0, 1, ...) indexing
coeffs_aul.values = coeffs_aul.values[:-1, 1:] # cut off the last row since lower goes from 1-8, cut off first col since upper goes from 2-9

# note: when computing energies/frequencies, make sure to add +2 to column index and +1 to row index for physically meaningful representation of upper/lower levels

def deltaE(i, j):
	'''
	For deltaE_ul, i = col+2, j = row+1, otherwise flip i and j.
	'''
	return -13.6*(1./i**2 - 1./j**2)

coeffs_bul = mat.Matrix(8, 8) # make matrices of same size as A_ul
coeffs_blu = mat.Matrix(8, 8) 
J_bar = mat.Matrix(8, 8) 

first_loop = True # used later for neat legend setup
temps = np.logspace(1, 6, 6) # 10 to 1e6 Kelvin temp range to see behavior as temp increases

for T in temps:

    k = 8.617e-5 # Boltzmann constant in eV / K units
    
    for row in range(coeffs_aul.values.shape[0]):
    	for col in range(coeffs_aul.values.shape[1]):
#    		J_bar.values[row, col] = 2*np.abs(deltaE(col+2, row+1)**3)/(1240.**2)*1./np.exp(np.abs(deltaE(col+2, row+1))-k*T) # populate Jbar
			# using non-absolute value of energy since negative sign does matter (absorbing or emitting energy)
    		J_bar.values[row, col] = 2*deltaE(col+2, row+1)**3/(1240.**2)*1./np.exp(deltaE(col+2, row+1)-k*T) # populate Jbar
    		if row > col: continue # A_ul is upper triangular, so skip lower triangular values (would be zero anyway, saves calculation time)
    		coeffs_bul.values[row, col] = coeffs_aul.values[row, col]*(1240**2)/(2*deltaE(col+2, row+1)**3)
    		coeffs_blu.values[col, row] = coeffs_bul.values[row, col]*(col+2)**2/(row+1)**2
    
    eqn_matrix = mat.Matrix(8, 8) # matrix for the actual equation components
    
    for row in range(eqn_matrix.n_rows):
    	for col in range(eqn_matrix.n_cols):
                # greater than and less than for row/col might be flipped, but if I didn't flip it I got a diagonal matrix which implied a trivial solution, so presumably it has to be this to produce an actual calculation...
    		if row > col: eqn_matrix.values[row, col] = -coeffs_blu.values[row, col]*J_bar.values[row, col]
    		if row < col: eqn_matrix.values[row, col] = -(coeffs_aul.values[row, col] + coeffs_bul.values[row, col]*J_bar.values[row, col])
    		if row == col: eqn_matrix.values[row, col] = np.sum(coeffs_blu.values[row, col:]*J_bar.values[row, col:]) + np.sum(coeffs_bul.values[row, :col]*J_bar.values[row, :col]) + np.sum(coeffs_aul.values[row, :col])
    
    eqn_matrix.invert() # invert equation matrix to get A^-1
    b = mat.Matrix(8, 1) # b vector in equation Ax = b
    b.populate(np.repeat(1e-10, 8)) # fill with almost zero values
    eqn_matrix.mult(b) # x = A^-1b
    
    normalization = np.sum(eqn_matrix.values) # normalize so that n1 + n2 + ... + n8 = N = 1 cm^-3
    
    ys = eqn_matrix.values/normalization # apply normalization
    colors = cm.rainbow(np.linspace(0, 1, 8)) # pretty colors
    for y, c in zip(ys, colors):
        if first_loop:
            plt.scatter(T, y, color=c, label='n%d' % int(np.where(ys==y)[0]+1))
        else: plt.scatter(T, y, color=c)
    first_loop=False
plt.yscale('log')
plt.xscale('log')
plt.title('Concentration of electrons per level in Hydrogen')
plt.xlabel('Temperature (K)')
plt.ylabel('Normalized Concentration')
plt.legend(ncol=2, fontsize='small')
plt.show()

# --- End of Task 2 ---

# --- Start of Task 3 ---
#
# See odes.py
#
# --- End of Task 3 ---

# --- Start of Task 4 ---

def pend(t, variables):
	# my pendulum function with b and c not taken as arguments but permanently set within
	b = 0.25
	c = 5.0
	theta, omega = variables
	dvars_dt = np.array([omega, -b*omega - c*np.sin(theta)])
	return dvars_dt

def pend_odeint(y, t, b, c):
	# pendulum function as defined in scipy's odeint method example
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

times = np.linspace(0, 10, 1001)

ts, ys = odes.euler(pend, times, np.array([np.pi-0.1, 0]), dt=0.001)
plt.plot(ts, ys[np.linspace(0, 19998, 10000).astype(int)], label='omega euler') # first var is in all the odd indices
plt.plot(ts, ys[np.linspace(1, 19999, 10000).astype(int)], label='theta euler') # second var is in all the even indices
ts, ys = odes.heun(pend, times, np.array([np.pi-0.1, 0]), dt=0.001)
plt.plot(ts, ys[np.linspace(0, 19998, 10000).astype(int)], label='omega heun')
plt.plot(ts, ys[np.linspace(1, 19999, 10000).astype(int)], label='theta heun')
ts, ys = odes.rk4(pend, times, np.array([np.pi-0.1, 0]), dt=0.001)
plt.plot(ts, ys[np.linspace(0, 19998, 10000).astype(int)], label='omega rk4')
plt.plot(ts, ys[np.linspace(1, 19999, 10000).astype(int)], label='theta rk4')

y0 = [np.pi-0.1, 0]
t = np.linspace(0, 10, 101)

sol = odeint(pend_odeint, y0, t, args=(0.25, 5.0)) # use odeint to test against my solvers

plt.plot(t, sol[:, 0], label='omega odeint')
plt.plot(t, sol[:, 1], label='theta odeint')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Angle and Angular Velocity of Damped Pendulum')
plt.legend(ncol=2)
plt.show()

# --- End of Task 4 ---

# --- Start of Task 5 ---

def stiff(t, y):
	lambd = 1000 # stiffness parameter
	dy_dt = np.array(-lambd*(y - np.cos(t)))
	return dy_dt

def stiff_real(t):
	lambd = 1000 # stiffness parameter
	return -lambd**2/(1+lambd**2)*np.exp(-lambd*t) + lambd/(1+lambd**2)*np.sin(t) + lambd**2/(1+lambd**2)*np.cos(t)

times = np.linspace(0, 10, 101)

ts, ys = odes.euler(stiff, times, np.array([1e-6]), dt=1e-3)
plt.plot(ts, ys, label='euler')
ts, ys = odes.heun(stiff, times, np.array([1e-6]), dt=1e-3)
plt.plot(ts, ys, label='heun')
ts, ys = odes.rk4(stiff, times, np.array([1e-6]), dt=1e-3)
plt.plot(ts, ys, label='rk4')
plt.plot(ts, stiff_real(ts), label='true')
plt.legend()
plt.xlabel('Times (s)')
plt.ylabel('Amplitude')
plt.title('Test of stiff ODE solving capability')
plt.show()

# --- End of Task 5 ---
