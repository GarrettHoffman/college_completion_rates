import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
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
import time
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

data = pd.read_csv('data.csv')

data['completion_class'] = 'temp'
data['completion_class'][(data['C150_4'] < data['C150_4'].mean())] = 'Below Average'
data['completion_class'][(data['C150_4'] >= data['C150_4'].mean())] = 'Above Average'


y = data.completion_class
X = data.drop(['Unnamed: 0','C150_4', 'completion_class', 'Year'], axis=1).values
y_train, y_test, X_train, X_test = train_test_split(y, X, 
                                                    test_size=.3, 
                                                    random_state=515)

X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

ADM_RATE_ALL_train = X_train[:,0]
COSTT4_A_train = X_train[:,1]
UGDS_BLACK_train = X_train[:,2]
INC_PCT_LO_train = X_train[:,3]
PAR_ED_PCT_1STGEN_train = X_train[:,4]
PCTFLOAN_train = X_train[:,5]

ADM_RATE_ALL_test = X_test[:,0]
COSTT4_A_test = X_test[:,1]
UGDS_BLACK_test = X_test[:,2]
INC_PCT_LO_test = X_test[:,3]
PAR_ED_PCT_1STGEN_test = X_test[:,4]
PCTFLOAN_test = X_test[:,5]

ADM_RATE_ALL_scaled_train = X_train_scaled[:,0]
COSTT4_A_scaled_train = X_train_scaled[:,1]
UGDS_BLACK_scaled_train = X_train_scaled[:,2]
INC_PCT_LO_scaled_train = X_train_scaled[:,3]
PAR_ED_PCT_1STGEN_scaled_train = X_train_scaled[:,4]
PCTFLOAN_scaled_train = X_train_scaled[:,5]

ADM_RATE_ALL_scaled_test = X_test_scaled[:,0]
COSTT4_A_scaled_test = X_test_scaled[:,1]
UGDS_BLACK_scaled_test = X_test_scaled[:,2]
INC_PCT_LO_scaled_test = X_test_scaled[:,3]
PAR_ED_PCT_1STGEN_scaled_test = X_test_scaled[:,4]
PCTFLOAN_scaled_test = X_test_scaled[:,5]

y_train = [1 if y == 'Above Average' else 0 for y in y_train ]
y_test = [1 if y == 'Above Average' else 0 for y in y_test ]
X_train = np.column_stack((COSTT4_A_scaled_train, INC_PCT_LO_scaled_train))
X_test = np.column_stack((COSTT4_A_scaled_test, INC_PCT_LO_scaled_test))

h = .02  # step size in the mesh

names = ["Decision Tree", "Random Forest", 
         "Logistic Regression", "Gradient Boosting Machine", 
         "RBF SVM", "Nearest Neighbors"]
classifiers = [
    DecisionTreeClassifier(max_depth=6),
    RandomForestClassifier(max_depth=6, n_estimators=100),
    LogisticRegression(),
    GradientBoostingClassifier(learning_rate = .05, subsample = .9),
    SVC(),
    KNeighborsClassifier(n_neighbors = 15)]

figure = plt.figure(figsize=(20, 12))
x_min, x_max = min(X_train[:, 0].min(),X_test[:, 0].min()) - .5, max(X_train[:, 0].max(),X_test[:, 0].max()) + .5
y_min, y_max = min(X_train[:, 1].min(),X_test[:, 1].min()) - .5, max(X_train[:, 1].max(),X_test[:, 1].max()) + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) 

i = 1
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(2, 3, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.5)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlabel('Average Annual Cost')
    ax.set_ylabel('% Low Income Family')
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    print name + ": Done"
    i += 1
    
figure.subplots_adjust(left=.02, right=.98)
plt.savefig('clfsrf_v2.png', bbox_inches='tight')
plt.savefig('clfsrf_v2.pdf', bbox_inches='tight')

print "Done"

