import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt

group=np.array([109,65,22,3,1])
N_death=np.array([0,1,2,3,4])
freq=group/sum(group)
plt.scatter(N_death,freq)
plt.show()

mu = np.average(N_death,weights=freq)
plt.scatter(N_death, freq)

p = poisson(mu).pmf(N_death)
plt.plot(N_death, p)
plt.show()
