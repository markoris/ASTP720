import numpy as np
import matplotlib.pyplot as plt

p, d, v, j, h, k, e, z = np.loadtxt('cepheid_data.txt', usecols=np.arange(1, 9, 1), delimiter=',', unpack=True)

p = np.log10(p) # take log of period
a_j = 3.1*(0.271)*e # calculate extinction coefficient
m_j = j + 5 - 5*np.log10(d) - a_j # obtain absolute magnitude 

#plt.scatter(p, m_j) # verifying that trend resembles that in Fig 2 on hw
#plt.gca().invert_yaxis()
#plt.show()

