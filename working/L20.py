import numpy as np
import matplotlib.pyplot as plt
import corner
import h5py
import sklearn.model_selection
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from astroML.utils import completeness_contamination
from sklearn.model_selection import GridSearchCV
from  sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis)
from astroML.utils import pickle_results

np.random.seed(18)

#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#

def compute_models(*args):
	names = []
	probs = []
	comp_list = []
	cont_list = []
	acc_list = []
	for classifier, kwargs in args:
		print(classifier.__name__)
		clf = classifier(**kwargs)
		clf.fit(X_train, y_train)

		y_probs = clf.predict_proba(X_test)[:, 1]
		y_pred = clf.predict(X_test)

		names.append(classifier.__name__)
		probs.append(y_probs)
		completeness, contamination = completeness_contamination(y_pred, y_test)
		Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
		acc_list.append(np.sum(Confusion.diagonal())/len(y_test))
		comp_list.append(completeness)
		cont_list.append(contamination)
	return names, probs, comp_list, cont_list, acc_list


#----------------------------------------------------------------------------#
#Load Data
#----------------------------------------------------------------------------#

filename = "sample_2e7_design_precessing_higherordermodes_3detectors.h5"
f= h5py.File(filename, "r")

N_righe=100000

mtot = f['mtot'][:N_righe]
q = f['q'][:N_righe]
z = f['z'][:N_righe]
psi = f['psi'][:N_righe]
iota = f['iota'][:N_righe]
"""
chi1x= f['chi1x'][:N_righe]
chi1y= f['chi1y'][:N_righe]
chi1z= f['chi1z'][:N_righe]
chi2x= f['chi2x'][:N_righe]
chi2y= f['chi2y'][:N_righe]
chi2z=f['chi2z'][:N_righe]
ra = f['ra'][:N_righe]
dec = f['dec'][:N_righe]
"""

det=f['det'][:N_righe]


X=np.stack((mtot,q,z,psi,iota)).T

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, det, test_size=0.25, random_state=8)


#----------------------------------------------------------------------------#
#Classification
#----------------------------------------------------------------------------#

print("----------------")
names, probs, completness, contamination, accuracies = compute_models(
									(GaussianNB, {}),
									(LinearDiscriminantAnalysis, {}),
									(QuadraticDiscriminantAnalysis, {}),
									(LogisticRegression,dict(class_weight='balanced', random_state=123, max_iter=1000)),
									(KNeighborsClassifier,dict(n_neighbors=10)),
									(DecisionTreeClassifier,dict(random_state=0, max_depth=12,criterion='entropy')),
									(RandomForestClassifier,dict(n_estimators=30, n_jobs=-1)),
									(BaggingClassifier,dict(base_estimator=DecisionTreeClassifier(random_state=42), bootstrap=True, random_state=42, n_jobs=-1)))


#----------------------------------------------------------------------------#
#Plot ROC curves and completeness/efficiency
#----------------------------------------------------------------------------#

fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25)

# ax1 will show roc curves
ax1 = plt.subplot(121)

# ax2 will show completeness/efficiency
ax2 = plt.subplot(122)

labels = dict(GaussianNB='GNB',
              LinearDiscriminantAnalysis='LDA',
              QuadraticDiscriminantAnalysis='QDA',
              LogisticRegression='LR',
              KNeighborsClassifier='KNN',
              DecisionTreeClassifier='DT',
              RandomForestClassifier='RFC',
              BaggingClassifier='BC'
              )

thresholds = np.linspace(0, 1, 1001)[:-1]

# iterate through and show results
for name, y_prob in zip(names, probs):
	fpr, tpr, thresh = roc_curve(y_test, y_prob)
	# add (0, 0) as first point
	fpr = np.concatenate([[0], fpr])
	tpr = np.concatenate([[0], tpr])
	#plot false/true positive rate
	ax1.plot(fpr, tpr, label=labels[name])
	
	#Note that the range of threshhold values here is 0% to 100% (0.0 to 1.0)
	comp = np.zeros_like(thresholds)
	cont = np.zeros_like(thresholds)
	for i, t in enumerate(thresholds):
		y_pred = (y_prob >= t)
		comp[i], cont[i] = completeness_contamination(y_pred, y_test)
	ax2.plot(1 - cont, comp, label=labels[name])

ax1.set_xlim(0, 0.04)
ax1.set_ylim(0, 1.02)
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
ax1.legend(loc=4)

ax2.set_xlabel('efficiency')
ax2.set_ylabel('completeness')
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0.2, 1.02)

plt.show()


#----------------------------------------------------------------------------#
#Print Results
#----------------------------------------------------------------------------#

for name,complete, contamin,accuracy in zip(names,completness,contamination,accuracies):
	print("----------------")
	print(name+" :")
	print("completeness: ", complete)
	print("contamination: ", contamin)
	print("accuracy: ",accuracy)


#----------------------------------------------------------------------------#
#Cross Validation Decision Tree
#----------------------------------------------------------------------------#
print('----------')
print('Decision Tree - CV:')
clf=DecisionTreeClassifier()
drange = np.arange(1,10)
grid = GridSearchCV(clf, param_grid={'max_depth': drange}, cv=5 , verbose=0,n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_params_['max_depth']
print("best parameter choice:", best)

clf=DecisionTreeClassifier(random_state=0, max_depth=best,criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
completeness, contamination = completeness_contamination(y_pred, y_test)
Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)
print("completeness", completeness)
print("contamination", contamination)
print("accuracy: ",accuracy)


#----------------------------------------------------------------------------#
#sklearn.neural_network.MLPClassifier
#----------------------------------------------------------------------------#
print('-------------------------')
clf = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=(5), 
    activation='logistic',
    solver='adam',
    alpha=0,
    learning_rate_init=0.001,
    max_iter=200000)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
completeness, contamination = completeness_contamination(y_pred, y_test)
Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)

print('NN MLP:')
print("completeness", completeness)
print("contamination", contamination)
print("accuracy: ",accuracy)
print("Confusion Matrix:")
print(Confusion)


#----------------------------------------------------------------------------#
#sklearn.neural_network.MLPClassifier --- Cross Validation
#----------------------------------------------------------------------------#
print('-------------------------')
@pickle_results('L20_CV.pkl')
def Cross_Val(X_train,y_train):
	drange=np.arange(start=5,stop=15)
	tuple_list=[]
	l_range=[0.001,0.005,0.01]
	for j in drange:
		thistuple=()
		for i in range (3):
			y = list(thistuple)
			y.append(j)
			thistuple = tuple(y)
			tuple_list.append(thistuple)

	grid = GridSearchCV(clf, param_grid={'hidden_layer_sizes': tuple_list,'learning_rate_init':l_range}, cv=3 , verbose=3,n_jobs=-1)
	grid.fit(X_train, y_train)

	return grid

CV=Cross_Val(X_train,y_train)
print(CV.best_params_)
bestclf=CV.best_estimator_ 
bestclf.fit(X_train, y_train)
y_pred=bestclf.predict(X_test)
completeness, contamination = completeness_contamination(y_pred, y_test)
Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)
print('NN MLP - CV:')
print("completeness", completeness)
print("contamination", contamination)
print("accuracy: ",accuracy)
print("Confusion Matrix:")
print(Confusion)
#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#
