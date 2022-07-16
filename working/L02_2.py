import numpy as np
import pylab as plt

N=100000
x=np.random.uniform(0.1,10,N)

plt.hist(x,density=True,histtype='step',lw=2)
plt.xlabel("$x$")

plt.show()

y=np.log10(x)
plt.hist(y,density=True,histtype='step',lw=2)

ygrid=np.linspace(-1,1,100)
theorical=10**ygrid*np.log(10)*(1/(10-0.1))
plt.xlabel("$y$")
plt.plot(ygrid,theorical);
plt.show()

print("Mean")
print("log10(x):",np.log10(np.mean(x)), "y:",np.mean(y))
print("Median")
print("log10(x):",np.log10(np.median(x)), "y:",np.median(y))

