#import
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from astroML.datasets import generate_mu_z
from astroML.linear_model import LinearRegression
from astroML.linear_model import PolynomialRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#

def linea(x,coeff):
	return coeff[1]*x+coeff[0]

def geterror(X,y,classifier):
    return np.sqrt( np.sum(( y - classifier.predict(X) )**2) / len(X) )

def fitanderror(classifier):
    classifier.fit(X_train, y_train,dy_train)
    error_train = geterror(X_train,y_train,classifier)
    error_validation  = geterror(X_validation, y_validation, classifier)
    return error_train, error_validation

#----------------------------------------------------------------------------#
#Load Data
#----------------------------------------------------------------------------#
cosmo = LambdaCDM(H0=71, Om0=0.27, Ode0=1-0.27)
z = np.linspace(0.01, 2, 100)
mu_true = cosmo.distmod(z)

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)
z_sample1=z_sample[:,np.newaxis]

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)

#----------------------------------------------------------------------------#
#Linear Regression (Whole dataset)
#----------------------------------------------------------------------------#
model = LinearRegression()
model.fit(z_sample1, mu_sample, dmu)
print(model.coef_)

xgrid=np.linspace(0,2,100)
xgrid1=xgrid[:,np.newaxis]
plt.plot(xgrid,linea(xgrid,model.coef_),label='d = 1')


#----------------------------------------------------------------------------#
#Polynomial Regression (Whole dataset)
#----------------------------------------------------------------------------#
deg_list=[2,4,10]

for degree in deg_list:
	model_pol = PolynomialRegression(degree)
	model_pol.fit(z_sample1, mu_sample,dmu)
	plt.plot(xgrid,model_pol.predict(xgrid[:,np.newaxis]),label='d = '+str(degree))
plt.legend()
plt.show()

#----------------------------------------------------------------------------#
#Polynomial Regression (Training and Testing)
#----------------------------------------------------------------------------#
X = z_sample1
y = mu_sample
dy = dmu

X_train, X_validation, ydy_train, ydy_validation = train_test_split(X, np.array([y,dy]).T, test_size=0.3, random_state=42)
y_train,dy_train = ydy_train.T
y_validation,dy_validation = ydy_validation.T

nrange = np.arange(1,10)

etrain, etest= [], []
for n in nrange:
    classifier = PolynomialRegression(n)
    error_train, error_validation = fitanderror(classifier)
    print("Polynomial Regression n=",n, error_train, error_validation)
    etrain.append(error_train)
    etest.append(error_validation)


#----------------------------------------------------------------------------#
#Plot Errors
#----------------------------------------------------------------------------#
plt.plot(nrange,etest, label='Validation set')
plt.plot(nrange,etrain, label='Training set')

plt.xlabel('n')
plt.ylabel('Error')
plt.title('Poly Degree Vs Error')
plt.legend()
plt.show()


#----------------------------------------------------------------------------#
#Fit Best Poly
#----------------------------------------------------------------------------#
print("Best", nrange[np.argmin(etest)], min(etest))

migliore=nrange[np.argmin(etest)]

classifier = PolynomialRegression(migliore)

classifier.fit(X, mu_sample,dmu)
mu_fit = classifier.predict(xgrid1)

mu_fit = classifier.predict(xgrid1)

plt.plot(xgrid1, mu_fit, label='Fit using all data')

classifier.fit(X_train, y_train,dy_train)
mu_fit = classifier.predict(xgrid1)

plt.plot(xgrid1, mu_fit, label='Fit using only traning data')

#plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)

plt.errorbar(X_train, y_train, dy_train, fmt='.k', ecolor='gray', lw=1,label='Training data')
plt.errorbar(X_validation, y_validation, dy_validation, fmt='.r', ecolor='red', lw=1, label='Validation data')

plt.plot(z, mu_true, '--', c='gray',label='true')

plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)
plt.legend(loc='lower right')
plt.show()


#----------------------------------------------------------------------------#
#Learning curve
#----------------------------------------------------------------------------#
classifier = PolynomialRegression(migliore)
# Ten training sample sizes from 10% to 100%
train_sizes, train_scores_linreg, test_scores_linreg = \
    learning_curve(classifier, z_sample1, mu_sample, train_sizes=np.linspace(0.1, 1, 10), \
    scoring="neg_mean_squared_error", cv=10)

fig = plt.figure(figsize=(8, 8))

plt.plot(train_sizes, -test_scores_linreg.mean(1), 'o-', color="r", label="Val")
plt.plot(train_sizes, -train_scores_linreg.mean(1), 'o-', color="g", label="Train")

plt.xlabel("Train size",fontsize=14)
plt.ylabel("Mean Squared Error",fontsize=14)
plt.title('Learning curves',fontsize=14)
plt.legend(loc="best")
plt.ylim(0,3)

plt.show()

#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#
