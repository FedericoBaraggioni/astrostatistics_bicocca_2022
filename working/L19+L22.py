import numpy as np
import matplotlib.pyplot as plt
import corner
from  sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from astroML.utils import completeness_contamination

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from astroML.classification import GMMBayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from astroML.utils import pickle_results

#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#

def compute_models(*args):
	names = []
	probs = []
	comp_list = []
	cont_list = []
	accuracy=[]
	for classifier, kwargs in args:
		print(classifier.__name__)
		clf = classifier(**kwargs)
		clf.fit(X_train, y_train)

		y_probs = clf.predict_proba(X_test)[:, 1]
		y_pred = clf.predict(X_test)

		names.append(classifier.__name__)
		probs.append(y_probs)
		completeness, contamination = completeness_contamination(y_pred, y_test)
		comp_list.append(completeness)
		cont_list.append(contamination)
		Confusion = confusion_matrix(y_test, y_pred)
		accuracy.append(np.sum(Confusion.diagonal())/len(y_test))
	return names, probs, comp_list, cont_list,accuracy


#----------------------------------------------------------------------------#
#Load Data
#----------------------------------------------------------------------------#

arr = np.loadtxt("galaxyquasar.csv", delimiter=",", dtype=str)
#print(arr[0])

u=arr[1:,0].astype(float)
g=arr[1:,1].astype(float)
r=arr[1:,2].astype(float)
i=arr[1:,3].astype(float)
z=arr[1:,4].astype(float)
classific=arr[1:,5]
z1=arr[1:,6].astype(float)
zerr=arr[1:,7].astype(float)

classification=np.where(classific=='QSO',np.ones(len(u)), np.zeros(len(u))).astype(int)

u_g=u-g
g_r=g-r
r_i=r-i
i_z=i-z

corner.corner(np.array([u_g,g_r,r_i,i_z]).T, labels=['ug', 'gr', 'ri', 'iz']);
plt.show()

bins=np.linspace(0,3,100)

plt.hist(u_g[classification==1],histtype='step',bins=bins,density=False)
plt.hist(u_g[classification==0],histtype='step',bins=bins,density=False)
plt.xlim(0,3)
plt.show()

color=np.stack((u_g,g_r,r_i,i_z)).T
X_train, X_test, y_train, y_test = train_test_split(color, classification, test_size=0.33, random_state=8)


#----------------------------------------------------------------------------#
#Classification
#----------------------------------------------------------------------------#
print("----------------")

names, probs, completness, contamination, accuracies = compute_models(
									(GaussianNB, {}),
									(LinearDiscriminantAnalysis, {}),
									(QuadraticDiscriminantAnalysis, {}),
									(LogisticRegression,dict(class_weight='balanced')),
									(KNeighborsClassifier,dict(n_neighbors=10)),
									(DecisionTreeClassifier,dict(random_state=0, max_depth=12,criterion='entropy')),
									(RandomForestClassifier,dict(n_estimators=30, n_jobs=-1)),
									(BaggingClassifier,dict(base_estimator=DecisionTreeClassifier(random_state=42), bootstrap=True, random_state=42, n_jobs=-1)),
									(GMMBayes, dict(n_components=3, tol=1E-5,covariance_type='full')))

#----------------------------------------------------------------------------#
#Print Results
#----------------------------------------------------------------------------#

for name,complete, contamin,accuracy in zip(names,completness,contamination,accuracies):
	print("----------------")
	print(name+" :")
	print("completeness", complete)
	print("contamination", contamin)
	print("accuracy: ",accuracy)
	
#----------------------------------------------------------------------------#
#Classification sklearn.neural_network
#----------------------------------------------------------------------------#
print('-------------------------')
clf = MLPClassifier(
    hidden_layer_sizes=(5), 
    activation='relu',
    solver='adam',
    alpha=0,
    learning_rate_init=0.001,
    max_iter=20000)

clf.fit(X_train[:1000], y_train[:1000])
y_pred = clf.predict(X_test)
completeness, contamination = completeness_contamination(y_pred, y_test)
Confusion = confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)
#----------------------------------------------------------------------------#
#Print Results
#----------------------------------------------------------------------------#
print('MLPClassifier:')
print("completeness", completeness)
print("contamination", contamination)
print("accuracy: ",accuracy)
print("Confusion Matrix:")
print(Confusion)


#----------------------------------------------------------------------------#
#Cross Validation
#----------------------------------------------------------------------------#
@pickle_results('L19_CV.pkl')
def Cross_Val(X_train,y_train):
	print('-------------------------')
	drange=np.arange(start=5,stop=15)
	tuple_list=[]
	l_range=[0.001,0.005,0.01]
	for j in drange:
		thistuple=()
		for i in range (4):
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
Confusion = confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)
print('MLPClassifier (CV):')
print("completeness", completeness)
print("contamination", contamination)
print("accuracy: ",accuracy)
print("Confusion Matrix:")
print(Confusion)

probs_mlp=bestclf.predict_proba(X_test)[:,1]
#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
#Plot ROC curves and completeness/efficiency
#----------------------------------------------------------------------------#

fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25)

# ax1 will show roc curves
ax1 = plt.subplot(121)

# ax2 will show completeness/efficiency
ax2 = plt.subplot(122)

labels = ['GNB','LDA','QDA','LR','KNN','DT','RFC','BC','GMMB','MLP']

thresholds = np.linspace(0, 1, 1001)[:-1]
names.append(MLPClassifier)
probs.append(probs_mlp)
# iterate through and show results
for name, y_prob in zip(labels, probs):
    fpr, tpr, thresh = roc_curve(y_test, y_prob)
    # add (0, 0) as first point
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])
	#plot false/true positive rate
    ax1.plot(fpr, tpr, label=name)

    #Note that the range of threshhold values here is 0% to 100% (0.0 to 1.0)
    comp = np.zeros_like(thresholds)
    cont = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t)
        comp[i], cont[i] = completeness_contamination(y_pred, y_test)
    ax2.plot(1 - cont, comp, label=name)
    
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
