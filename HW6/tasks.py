import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import f

# --- Start of Task 1 ---

p, d, v, j, h, k, e, z = np.loadtxt('cepheid_data.txt', usecols=np.arange(1, 9, 1), delimiter=',', unpack=True)

p = np.log10(p) # take log of period
a_j = 3.1*(0.271)*e # calculate extinction coefficient
m_j = j + 5 - 5*np.log10(d) - a_j # obtain absolute magnitude 

#plt.scatter(p, m_j) # verifying that trend resembles that in Fig 2 on hw
#plt.gca().invert_yaxis()
#plt.show()

# equation is alpha + beta*log10(P) + gammma*Z
# params are alpha, beta, gamma
# design matrix columns are 1, log10(P) and Z, respectively

x = np.zeros((p.shape[0], 3))
x[:, 0] = 1 # column 1 = 1
x[:, 1] = p # column 2 = log10(P)
x[:, 2] = z # column 3 = Z

x = np.matrix(x)

theta = np.matmul(x.H, x) # equation 18 in lecture notes 13
theta = np.linalg.inv(theta)
theta = np.matmul(theta, x.H)
theta = np.matmul(theta, m_j).T

theta_errors = np.matmul(x.H, x) # equation 33
theta_errors = np.linalg.inv(theta_errors)
theta_errors = theta_errors.diagonal()
theta_errors = np.sqrt(theta_errors)
print('error-free model parameters and errors')
print(theta.T)
print(theta_errors)

# --- End of Task 1 ---

# --- Start of Task 2 ---

idxs = np.argsort(p)

fit = theta[0] + theta[1]*p[idxs] + theta[2]*z[idxs]

plt.plot(p[idxs], fit.T, 'r', label='fit') # checking fit against data
plt.scatter(p, m_j, label='data') 
plt.gca().invert_yaxis()
plt.title('Cepheid Period-Luminosity-Metallicity Relation (no errors)')
plt.xlabel('log Period (days)')
plt.ylabel('Absolute Magnitude')
plt.legend()
plt.savefig('figures/no_errors.png')
plt.close()

# --- End of Task 2 ---

# --- Start of Task 3 ---

del theta, theta_errors, fit # ensuring no cross-over from task 2

v = np.identity(m_j.shape[0])*(0.1**2) # equation 24

theta = np.matmul(x.H, np.linalg.inv(v)) # equation 37
theta = np.matmul(theta, x)
theta = np.linalg.inv(theta)
theta = np.matmul(theta, x.H)
theta = np.matmul(theta, np.linalg.inv(v))
theta = np.matmul(theta, m_j).T

theta_errors = np.matmul(x.H, np.linalg.inv(v)) # equation 38
theta_errors = np.matmul(theta_errors, x)
theta_errors = np.linalg.inv(theta_errors)
theta_errors = theta_errors.diagonal()
theta_errors = np.sqrt(theta_errors)

print('error-inclusive model parameters and errors')
print(theta.T)
print(theta_errors)

fit = theta[0] + theta[1]*p[idxs] + theta[2]*z[idxs]

plt.plot(p[idxs], fit.T, 'r', label='fit')
plt.scatter(p, m_j, label='data') # verifying that trend resembles that in Fig 2 on hw
plt.gca().invert_yaxis()
plt.title('Cepheid Period-Luminosity-Metallicity Relation (with errors)')
plt.xlabel('log Period (days)')
plt.ylabel('Absolute Magnitude')
plt.legend()
plt.savefig('figures/with_errors.png')
plt.close()

# --- End of Task 3---

# --- Start of Bonus Task ---

x_nested = np.zeros((p.shape[0], 2))
x_nested[:, 0] = 1 # column 1 = 1
x_nested[:, 1] = p # column 2 = log10(P), no third column for nested model

x_nested = np.matrix(x_nested)

v = np.identity(m_j.shape[0])*(0.1**2) # equation 24

# repeating all the same calculations as before just on nested model

theta_nested = np.matmul(x_nested.H, np.linalg.inv(v)) # equation 37
theta_nested = np.matmul(theta_nested, x_nested)
theta_nested = np.linalg.inv(theta_nested)
theta_nested = np.matmul(theta_nested, x_nested.H)
theta_nested = np.matmul(theta_nested, np.linalg.inv(v))
theta_nested = np.matmul(theta_nested, m_j).T

theta_nested_errors = np.matmul(x_nested.H, np.linalg.inv(v)) # equation 38
theta_nested_errors = np.matmul(theta_nested_errors, x_nested)
theta_nested_errors = np.linalg.inv(theta_nested_errors)
theta_nested_errors = theta_nested_errors.diagonal()
theta_nested_errors = np.sqrt(theta_nested_errors)

print('error-inclusive nested model parameters and errors')
print(theta_nested.T)
print(theta_nested_errors)

fit_nested = theta_nested[0] + theta_nested[1]*p[idxs]

plt.plot(p[idxs], fit_nested.T, 'r', label='fit')
plt.scatter(p, m_j, label='data') # verifying that trend resembles that in Fig 2 on hw
plt.gca().invert_yaxis()
plt.title('Cepheid Period-Luminosity-Metallicity Relation (with errors)')
plt.xlabel('log Period (days)')
plt.ylabel('Absolute Magnitude')
plt.legend()
plt.savefig('figures/nested.png')

fit = fit.T
fit_nested = fit_nested.T
fit.resize(fit.shape[0])
fit_nested.resize(fit_nested.shape[0])

test = np.array([])

fit_chisq_full = np.array([])
fit_chisq_nest = np.array([])
for val in range(fit.shape[0]):
	fit_chisq_full = np.append(fit_chisq_full, fit[val])
	fit_chisq_nest = np.append(fit_chisq_nest, fit_nested[val])

chisq_full = chisquare(fit_chisq_full, f_exp=m_j).statistic
chisq_nest = chisquare(fit_chisq_nest, f_exp=m_j).statistic

nu_full = m_j.shape[0] - 3
nu_nest = m_j.shape[0] - 2

numerator = (chisq_nest - chisq_full) / (nu_full - nu_nest)
denominator = chisq_full / (m_j.shape[0] - nu_full)

f_val = numerator/denominator

f_val_cdf = f.cdf(f_val, np.abs(nu_full - nu_nest), nu_full)

print('f-value and f-value CDF output')
print(f_val, f_val_cdf)

# --- End of Bonus Task ---
