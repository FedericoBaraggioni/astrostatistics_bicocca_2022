import requests
import numpy as np
import pylab as plt 
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn import cluster
from sklearn.cluster import estimate_bandwidth
from sklearn.model_selection import GridSearchCV


#-----------------------------------------------------------------------------------#
#Load Data
#-----------------------------------------------------------------------------------#
r = requests.get('https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt')
with open("Summary_table.txt", 'wb') as f:
    f.write(r.content)

data = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

# Read headers
with open("Summary_table.txt",'r') as f:
    names= np.array([n.strip().replace(" ","_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])
print(names)
#Initializing the arrays
T90=np.array(data[6])
T90_err=np.array(data[7])
Fluence=np.array(data[9])
F_err=np.array(data[10])
redshift=np.array(data[11])

#Converting from strings to float
T90=T90.astype(float)
Fluence=Fluence.astype(float)
T90_err=T90_err.astype(float)
F_err=F_err.astype(float)
redshift=redshift.astype(float)
#errortimes= grb['T90_error'][~np.isnan(np.log10(grb['T90']))]

#Removing -999 values
sel1=T90 >-999
sel2=Fluence > -999
sel3=T90!=0
sel4=Fluence != 0

T90=T90[np.logical_and(np.logical_and(sel1, sel2),np.logical_and(sel3, sel4))]
T90_err=T90_err[np.logical_and(np.logical_and(sel1, sel2),np.logical_and(sel3, sel4))]
Fluence=Fluence[np.logical_and(np.logical_and(sel1, sel2),np.logical_and(sel3, sel4))]
F_err=F_err[np.logical_and(np.logical_and(sel1, sel2),np.logical_and(sel3, sel4))]

lT90=np.log10(T90)
lFlu=np.log10(Fluence)
TF1=np.stack((lT90,lFlu), axis=-1)


#-----------------------------------------------------------------------------------#
#Plot Data
#-----------------------------------------------------------------------------------#

#Plotting Fluence Vs T90
plt.scatter(x=lT90,y=lFlu,alpha=0.5,s=3)
plt.xlabel('log10(T90)')
plt.ylabel('log10(Fluence)')
plt.show()

#plotting relative errors
plt.scatter(T90,(T90_err/T90),alpha=0.1)
plt.axhline(1,c='red')
plt.ylim(-1,20)
plt.semilogx()
plt.xlabel('T90')
plt.ylabel('T90_err/T90')

plt.show()

plt.scatter(Fluence,(F_err/Fluence),alpha=0.1)
plt.xlabel('Fluence')
plt.ylabel('Fluence_err/Fluence')
plt.ylim(-1,20)
plt.axhline(1,c='red')
plt.semilogx()
plt.show()
#-----------------------------------------------------------------------------------#
#Looking at the Histogram using KDE
#-----------------------------------------------------------------------------------#

X_plot = np.linspace(-3, 3, 1000)[:, np.newaxis]

kde = KernelDensity(kernel="gaussian", bandwidth=0.15).fit(lT90[:, np.newaxis])
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:,0], np.exp(log_dens),label="log(T90)")
plt.title('log10(T90) Gaussian KernelDensity')
plt.show()


#-----------------------------------------------------------------------------------#
#Identifying the number of processes with GaussianMixture
#-----------------------------------------------------------------------------------#
print("Gaussian Mixture")
N = np.arange(1, 7)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GaussianMixture(N[i],n_init=15).fit(lT90[:, np.newaxis])

AIC= [model.aic(lT90[:, np.newaxis]) for model in models]

plt.plot(N, AIC)
plt.scatter(N, AIC)

plt.xlabel('n. components')
plt.ylabel('information criterion')
#plt.legend(loc=2);
plt.show()

M_best = models[np.argmin(AIC)]

x = np.linspace(-10, 10, 1000)
logprob = M_best.score_samples(x.reshape(-1, 1))
pdf = np.exp(logprob)

responsibilities = M_best.predict_proba(x.reshape(-1, 1))
pdf_individual = responsibilities * pdf[:, np.newaxis]

plt.hist(lT90, 50, density=True, histtype='step', alpha=0.4,color='black')

for i in range(N[np.argmin(AIC)]):
    plt.plot(x, pdf_individual[:,i], c='C0')

plt.plot(x, pdf, '--k')
plt.show()


#-----------------------------------------------------------------------------------#
#Searching for 2 cluster -- Kmeans
#-----------------------------------------------------------------------------------#

clf = KMeans(n_clusters=2)
clf.fit(TF1)
labels = clf.predict(TF1)

# plot the data color-coded by cluster id
fig2, ax2 = plt.subplots(figsize = (9, 6))
colors = ['C0', 'C1']
for ii in range(2):
    plt.scatter(T90[labels==ii], Fluence[labels==ii], color=colors[ii],alpha=0.5,s=3)

plt.title('KMeans')
ax2.set_xscale("log")
ax2.set_yscale("log")

plt.show()


#-----------------------------------------------------------------------------------#
#Searching for N cluster -- MeanShift
#-----------------------------------------------------------------------------------#
scaler = preprocessing.StandardScaler()
bandwidth = 0.54

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False,n_jobs=-1)
ms.fit(scaler.fit_transform(TF1))

labels_unique = np.unique(ms.labels_)
n_clusters = len(labels_unique[labels_unique >= 0])

print("number of estimated clusters :", n_clusters)


#-----------------------------------------------------------------------------------#
# Plotting MeanShift Resuslts
#-----------------------------------------------------------------------------------#

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()

# Compute a 2D histogram  of the input
H, xedges, yedges = np.histogram2d(lT90, lFlu, 50)

plt.scatter(x=lT90,y=lFlu,alpha=0.5,s=3)
# plot cluster centers
cluster_centers = scaler.inverse_transform(ms.cluster_centers_)
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=40, c='w', edgecolors='k')

# plot cluster boundaries
x_centers = 0.5 * (xedges[1:] + xedges[:-1])
y_centers = 0.5 * (yedges[1:] + yedges[:-1])

Xgrid = np.meshgrid(x_centers, y_centers)
Xgrid = np.array(Xgrid).reshape((2, 50 * 50)).T

H = ms.predict(scaler.transform(Xgrid)).reshape((50, 50))

for i in range(n_clusters):
    Hcp = H.copy()
    flag = (Hcp == i)
    Hcp[flag] = 1
    Hcp[~flag] = 0

    ax.contour(x_centers, y_centers, Hcp, [-0.5, 0.5],
               linewidths=1, colors='k')
 
    H = ms.predict(scaler.transform(Xgrid)).reshape((50, 50))
    
ax.set_xlim(xedges[0], xedges[-1])
ax.set_ylim(yedges[0], yedges[-1])

plt.show()

#-----------------------------------------------------------------------------------#
#
#-----------------------------------------------------------------------------------#

