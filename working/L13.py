from sklearn import datasets
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift

import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from astroML.utils import completeness_contamination
from sklearn.ensemble import RandomForestClassifier

from astroML.utils import pickle_results

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
from scipy.sparse import (SparseEfficiencyWarning, csr_matrix, lil_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

np.random.seed(18)

#----------------------------------------------------------------------------#
#Function Definitions
#----------------------------------------------------------------------------#
def Transformation(algo,nn=4,nc=2,plot=1):
	if algo=='Isomap':
		Reduct = Isomap(n_neighbors=nn,n_components=nc)
	elif algo=='TSNE':
		Reduct = TSNE(n_components=2,learning_rate=400,init='pca')
	X_transformed = Reduct.fit_transform(digits.data)
	if(plot==1):
		plt.title(algo)
		plt.scatter(x=X_transformed[:,0],y=X_transformed[:,1],c=digits.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
		plt.colorbar(label='digit label', ticks=range(10))
		plt.clim(-0.5, 9.5)
		plt.show()
	
	return X_transformed

def Classifier(*args):
	names = []
	acc_list = []
	conf_list = []
	for classifier, kwargs in args:
		print(classifier.__name__)
		clf = classifier(**kwargs)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		names.append(classifier.__name__)
		Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
		acc_list.append(np.sum(Confusion.diagonal())/len(y_test))
		conf_list.append(Confusion)
	return names, conf_list, acc_list

def Cluster(cluster,plot=1):
	if cluster=='Kmeans':
		clf = KMeans(n_clusters=10)
		clf.fit_transform(X_transformed)
		labels = clf.predict(X_transformed)
	
	#labels for each of the points
	labels_matched = np.empty_like(labels)
	# For each cluster label...
	for k in np.unique(labels):
		# ...find and assign the best-matching truth label
		match_nums = [np.sum((labels==k)*(digits.target==t)) for t in np.unique(digits.target)]
		labels_matched[labels==k] = np.unique(digits.target)[np.argmax(match_nums)]
	if(plot==1):
		plt.scatter(x=X_transformed[:,0], y=X_transformed[:,1], c=labels_matched,edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('jet', 10))
		plt.colorbar(label='digit label', ticks=range(10))
		plt.clim(-0.5, 9.5)
		plt.title(cluster)
		plt.show()
	
	return (accuracy_score((labels_matched), (digits.target))),(confusion_matrix(labels_matched, digits.target))


#----------------------------------------------------------------------------#
#Load Data
#----------------------------------------------------------------------------#

digits = datasets.load_digits()
#print(digits.keys())

sc = StandardScaler()
X_scaled = sc.fit_transform(digits.data)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_scaled, digits.target, test_size=0.33, random_state=42)

#----------------------------------------------------------------------------#
#Classifiers
#----------------------------------------------------------------------------#
names,matrices, accuracies = Classifier(
									(LogisticRegression,dict(solver='sag',max_iter=1000)),
									(RandomForestClassifier,dict(n_estimators=30, n_jobs=-1)),
									(SVC,dict(gamma=0.001, C=100)))


#----------------------------------------------------------------------------#
#Unsupervised ---- (Dimensionality reduction + Clustering)
#----------------------------------------------------------------------------#

X_transformed=Transformation('Isomap')
Isomap_sc,Isomap_Conf=Cluster('Kmeans')

X_transformed = Transformation('TSNE')
TSNE_sc,TSNE_Conf=Cluster('Kmeans')
#----------------------------------------------------------------------------#
#Print Accuracy
#----------------------------------------------------------------------------#

Methods=['Isomap+Kmeans: ','TSNE+Kmeans: ']
Score=[Isomap_sc,TSNE_sc]
Conf_Matr=[Isomap_Conf,TSNE_Conf]

print('@@@@@@@@@@@@@@@@@@@@@@@@@')
for name,accuracy,matrix in zip(names,accuracies,matrices):
	print("------------------------------")
	print(name,":")
	print("accuracy: ",accuracy)
	print("confusion matrix:\n",matrix)

for i in range(len(Score)):
	print(Methods[i],':')
	print("accuracy: ",Score[i])
	print("confusion matrix:\n",Conf_Matr[i])
	
print('---------------------------')

#----------------------------------------------------------------------------#
#sklearn.neural_network
#----------------------------------------------------------------------------#
print('MLPClassifier:')

clf = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=(5), 
    activation='relu',
    solver='lbfgs',
    alpha=0.1,
    learning_rate_init=0.001,
    max_iter=20000)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)

print("accuracy: ",accuracy)
print("Confusion Matrix:")
print(Confusion)

print('-------------------------')
#----------------------------------------------------------------------------#
#sklearn.neural_network --- Cross Validation
#----------------------------------------------------------------------------#
@pickle_results('L13_CV.pkl')
def Cross_Val(X_train,y_train):
	print('MLPClassifier - Cross Validation:')
	drange=np.arange(start=55,stop=75)
	tuple_list=[]
	l_range=[0.001,0.01]
	for j in drange:
		thistuple=()
		for i in range (2):
			y = list(thistuple)
			y.append(j)
			thistuple = tuple(y)
			tuple_list.append(thistuple)

	grid = GridSearchCV(clf, param_grid={'hidden_layer_sizes': tuple_list,'learning_rate_init':l_range}, cv=3 , verbose=0,n_jobs=-1)
	grid.fit(X_train, y_train)
	return grid
	
CV=Cross_Val(X_train,y_train)
print(CV.best_params_)
bestclf=CV.best_estimator_ 
bestclf.fit(X_train, y_train)
y_pred=bestclf.predict(X_test)
Confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy=np.sum(Confusion.diagonal())/len(y_test)

print("accuracy: ",accuracy)
print("Confusion Matrix:")
print(Confusion)

"""
#----------------------------------------------------------------------------#
#Isomap: Accuracy VS Number of components
#----------------------------------------------------------------------------#

a=np.arange(start=1,stop=20)
print('--------------------------')
scores=[]
for i in a:
	X_transformed=Transformation('Isomap',nc=i, plot=0)
	Score,Conf=Cluster('Kmeans', plot=0)
	scores.append(Score)
	
plt.plot(a,scores)
plt.xlabel('Number of components')
plt.ylabel('Accuracy')
plt.show()
#----------------------------------------------------------------------------#
#
#----------------------------------------------------------------------------#
"""
