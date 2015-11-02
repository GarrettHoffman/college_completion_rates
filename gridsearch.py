import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import BernoulliRBM
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sqlite3
import time

data = pd.read_csv('data.csv')

data['completion_class'] = 'temp'
data['completion_class'][(data['C150_4'] < data['C150_4'].mean())] = 'Below Average'
data['completion_class'][(data['C150_4'] >= data['C150_4'].mean())] = 'Above Average'


y = data.completion_class
X = data.drop(['C150_4', 'completion_class', 'Year'], axis=1).values
y_train, y_test, X_train, X_test = train_test_split(y, X, 
                                                    test_size=.3, 
                                                    random_state=515)

RBM = BernoulliRBM()
Log = LogisticRegression()
classifier = Pipeline([("rbm", RBM), ("logistic", Log)])

params = {
    "rbm__learning_rate": [0.1, 0.01, 0.001],
    "rbm__n_iter": [20, 40, 80],
    "rbm__n_components": [50, 100, 200],
    "logistic__C": [1.0, 10.0, 100.0]}

start = time.time()
gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1)
gs.fit(X_train, y_train)

print "\ndone in %0.3fs" % (time.time() - start)
print "best score: %0.3f" % (gs.best_score_)
print "RBM + LOGISTIC REGRESSION PARAMETERS"
bestParams = gs.best_estimator_.get_params()

for p in sorted(params.keys()):
    print "\t %s: %f" % (p, bestParams[p])
