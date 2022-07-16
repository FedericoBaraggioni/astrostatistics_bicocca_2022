import numpy as np
import pylab as plt

N=10000
mu, sigma = 5, 3 # mean and standard deviation

data = np.random.normal(mu, sigma, N)

plt.hist(data,density=False,histtype='step',lw=2)
plt.xlabel("$data$")

plt.show()

x_mean=np.mean(data)
residuals=data-x_mean
s=np.sqrt(sum((residuals)**2)/(N-1))
sigma_x=s/np.sqrt(N)
sigma_s=s/np.sqrt(2*(N-1))

print('Mu_true =',5)
print('Sigma_true = ', 3)
print('Mu_samp =', round(x_mean,2),'+/-',round(sigma_x,2))
print('Sigma_samp = ', round(s,2),'+/-',round(sigma_s,2))

