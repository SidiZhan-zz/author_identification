# -*- coding: utf-8 -*-
"""
execfile("estimator_selection.py")
"""
print(__doc__)

import db
import author_identification as ai
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize,scale
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
##from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier, Layer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

def decision2proba(sc): # decision for svm
    sc = sc - np.min(sc,axis=1).reshape((-1,1)) * np.ones((1,nClass))
    sc = sc / (np.sum(sc,axis=1).reshape((-1,1)) * np.ones((1,nClass)))
    return sc


def RBF_C_gamma_GridSearchCV():
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
##    from sklearn.model_selection import StratifiedShuffleSplit
##    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
##    from sklearn.model_selection import GridSearchCV
##    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(SVC(), param_grid=param_grid) #cv=None inputs, StratifiedKFold is used.
    grid.fit(data, target)
    print("The best parameters are %s with a score of %0.2f"%(grid.best_params_, grid.best_score_))
    return grid.best_params_

def RBF_C_gamma():
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    max_score = 0
    max_param = None
    for C in C_range:
        for gamma in gamma_range:
            clf = SVC(C=C, gamma=gamma)
            score = clf.fit(trainingSet, trainingLabels).score(testSet,testLabels)
##            print C,gamma,score
            if score > max_score:
                max_param = (C,gamma)
                max_score = score
    print max_param,max_score
    return max_param



def logistic():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(trainingSet,trainingLabels)
    sc = clf.predict_proba(testSet)
    print evaluation(testLabels, classes[np.argmax(sc,axis=1)])
    prob_pos = clf.decision_function(testSet)
    sc = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    evaluation(testLabels, classes[np.argmax(sc,axis=1)])


if __name__ == "__main__":
    nFold = 5
    skf = StratifiedKFold(n_splits=nFold)
    for train_index, test_index in skf.split(data, target):
        pass
    trainingSet = data[train_index]
    trainingLabels = target[train_index]
    testSet = data[test_index]
    testLabels = target[test_index]
    st = time.time()
##    clf.fit(trainingSet,trainingLabels)
##    sc = clf.predict_proba(testSet)
##    print evaluation(testLabels, classes[np.argmax(sc,axis=1)])
##    featureSelection(trainingSet,trainingLabels,testSet,testLabels,fea_alg = 'dt')

##    from sklearn.decomposition import PCA
##    from sklearn.neural_network import MLPClassifier
##    clf = MLPClassifier()#(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
##    clf.fit(trainingSet,trainingLabels)
##    sc = clf.predict_proba(testSet)
##    print 'predict_proba',evaluation(testLabels, classes[np.argmax(sc,axis=1)])
##    print 'score', clf.score(testSet,testLabels)
##    from sklearn.calibration import CalibratedClassifierCV
##    clf = clf = OneVsRestClassifier(LinearSVC(random_state=0)) #slow
##    clf = SVC(gamma = 0.00001, C = 1000., probability = True, decision_function_shape = 'ovr')
##    clf = KNeighborsClassifier(n_neighbors=[5,len(test_index)/10,10][int(len(test_index)>50) + int(len(test_index)>100)])
    clf = ExtraTreesClassifier(n_estimators=300, max_features= "sqrt")
##    clf = GaussianNB()
##    clf = LogisticRegression()
##    clf = MLPClassifier()
##    clf.fit(trainingSet,trainingLabels)
##    sc = clf.predict_proba(testSet)
##    print ai.evaluation(testLabels, classes[np.argmax(sc,axis=1)]),time.time()-st
    st = time.time()
    c_clf = CalibratedClassifierCV(clf, method="sigmoid")#, cv="prefit")
    c_clf.fit(trainingSet,trainingLabels)
    sc = c_clf.predict_proba(testSet) # 'LinearSVC' object has no attribute 'predict_proba'
    print ai.evaluation(testLabels, classes[np.argmax(sc,axis=1)]),time.time()-st
##    recommList = ['lda']#['knn','svm','et','nb','lr','mlp','lsvm','lda','qda']
##    classifiers = {
##        'lsvm':OneVsRestClassifier(LinearSVC(random_state=0)),
##        'svm': SVC(gamma = 0.00001, C = 1000., probability = True, decision_function_shape = 'ovr'),
##        'knn': KNeighborsClassifier(n_neighbors=[5,len(test_index)/10,10][int(len(test_index)>50) + int(len(test_index)>100)]),
##        'et': ExtraTreesClassifier(n_estimators=300, max_features= "sqrt"),
##        'nb': GaussianNB(),
##        'lr': LogisticRegression(),
##        'mlp': MLPClassifier(),
##        'lda': LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True),
##        'qda': QuadraticDiscriminantAnalysis(store_covariances=True)
##    }
##    for recom in recommList:
##        st = time.time()
##        clf = classifiers[recom]
##        clf.fit(trainingSet,trainingLabels)
##        try:
##            df = clf.decision_function(testSet)
##            df = df - np.min(df,axis=1).reshape((-1,1)) * np.ones((1,nClass))
##            df = df / (np.sum(df,axis=1).reshape((-1,1)) * np.ones((1,nClass)))
##        except:
##            print 'clf.decision_function(testSet) error!'
##            df = clf.predict_proba(testSet)
##        try:
##            pp = clf.predict_proba(testSet)
##        except:
##            print 'clf.predict_proba(testSet) error!'
##            df = clf.decision_function(testSet)
##            df = df - np.min(df,axis=1).reshape((-1,1)) * np.ones((1,nClass))
##            pp = df / (np.sum(df,axis=1).reshape((-1,1)) * np.ones((1,nClass)))
##        edf = evaluation(testLabels, classes[np.argmax(df,axis=1)])
##        epp = evaluation(testLabels, classes[np.argmax(pp,axis=1)])
##        if np.mean(evaluation(testLabels, classes[np.argmax(df,axis=1)])) > np.mean(evaluation(testLabels, classes[np.argmax(pp,axis=1)])):
##            sample_class = df
##        else:
##            sample_class = pp
##        print 'estimator:',recom,"['Reciprocal Rank','Precision','Recall','Fmeasure']:",evaluation(testLabels, classes[np.argmax(sample_class,axis=1)]),'running time:',time.time()-st,'sample_class:',sample_class
