# -*- coding: utf-8 -*-
'''
execfile("feature_selection.py")
'''
import numpy as np
import db
from db import tags, askers, questions, comments, answers
from db import feature_path, clf_path, equiv_path, plot_path, lda_path
from author_identification import evaluation
from equivalence import Equivalence

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import itertools
from collections import Counter

def feature_selection():
    print clf.fit(trainingSet, trainingLabels).score(testSet, testLabels)
    feature_rank = clf.sigma_
    ind = np.argsort(np.sum(feature_rank,axis=0))
    max_score = 0
    max_i = 0
    for i in range(0,len(ind)+1,10):
        score = clf.fit(trainingSet[:,ind[:i+1]], trainingLabels).score(testSet[:,ind[:i+1]], testLabels)
        if score>max_score:
            max_score = score
            max_i = i
    print 'ranked by sigma_ asc, max_feature=ind[:%d+1], max_score=%d'%(max_i,max_score)


def dimensionReduction(data, target, fea_alg = 'et'): #featureSelection(trainingSet,trainingLabels,testSet,testLabels,fea_alg = 'dt'):
    nFold = 5
    skf = StratifiedKFold(n_splits=nFold)
    for train_index, test_index in skf.split(data, target):
        pass
    trainingSet = data[train_index]
    trainingLabels = target[train_index]
    testSet = data[test_index]
    testLabels = target[test_index]
    
    # random forest, feature_importances_, feature importances
    if fea_alg == 'et':
        clf = ExtraTreesClassifier(n_estimators=300, random_state=0, max_features= "sqrt")
        clf.fit(trainingSet,trainingLabels)
        select  = clf.feature_importances_
        score0 = clf.score(testSet,testLabels)
        model = SelectFromModel(clf, prefit=True)
        train_new = model.transform(trainingSet)
        test_new = model.transform(testSet)
        score1 = clf.fit(train_new, trainingLabels).score(test_new,testLabels)
        print train_new.shape[1], score0, score1, select[:5]

    if fea_alg == 'lsvc':
        clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, target)
        clf.fit(trainingSet,trainingLabels)
        select  = clf.coef_
        score0 = clf.score(testSet,testLabels)
        model = SelectFromModel(clf, prefit=True)
        train_new = model.transform(trainingSet)
        test_new = model.transform(testSet)
        score1 = clf.fit(train_new, trainingLabels).score(test_new,testLabels)
        print train_new.shape[1], score0, score1, select[:5]
        
    # naive bayesian, sigma_ : array, shape (n_classes, n_features), variance of each feature per class
    elif fea_alg == 'nb':
        clf = GaussianNB()
        clf.fit(trainingSet,trainingLabels)
        feature_rank = clf.sigma_
        ind = np.argsort(np.sum(feature_rank,axis=0))
        max_score = 0
        max_i = 0
        for i in range(0,len(ind)+1,10):
            score = clf.fit(trainingSet[:,ind[:i+1]], trainingLabels).score(testSet[:,ind[:i+1]], testLabels)
            if score>max_score:
                max_score = score
                max_i = i
        feature_ind = ind[:max_i+1]
        data = data[feature_ind]

    return data


def featureSelection_single():
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    featureList = ['up']#['createTime', 'stylometryDis', 'topicDis', 'scoreMean', 'timeSpan', 'qA_LastActTime', 'qA_CommentCount', 'q_Tags']
    figure = plt.figure(figsize=(27, 9))
    st = figure.suptitle('feature selection - single')
    i = 1
    for fea in featureList:
        results = np.zeros(nRound)
        for r in range(nRound):
            features = {}
            createTime = np.loadtxt(feature_path + 'CreateTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['createTime'] = createTime
            stylometryDis = np.loadtxt(feature_path + 'StylometryDis[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['stylometryDis'] = stylometryDis
            topicDis = np.loadtxt(feature_path + 'TopicDis_new[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['topicDis'] = topicDis
            scoreMean = np.loadtxt(feature_path + 'ScoreMean[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['scoreMean'] = scoreMean
            timeSpan = np.loadtxt(feature_path + 'TimeSpan[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['timeSpan'] = timeSpan
            qA_LastActTime= np.loadtxt(feature_path + 'QA_LastActTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_LastActTime'] = qA_LastActTime
            qA_CommentCount= np.loadtxt(feature_path + 'QA_CommentCount[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_CommentCount'] = qA_CommentCount
            q_Tags = np.loadtxt(feature_path + 'Q_Tags[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['q_Tags'] = q_Tags
            data = features[fea]
            clf = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True)
            clf.fit(data,target)
            classes = clf.classes_
            sc = clf.predict_proba(data)
            results[r] = ai.evaluation(target, classes[np.argmax(sc,axis=1)])[0]
        ax = plt.subplot(2, len(featureList)/2, i)
        ax.set_title(fea)
        ax.plot(results)
        i += 1
    figure.subplots_adjust(left=0.05, bottom=0.04, right=0.98, top=0.85, wspace=0.25, hspace=0.16)
    st.set_y(0.95)
    figure.savefig(plot_path + 'feature selection - single' + '.png')
    plt.close()

def featureSelection_multiple():
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    featureList = ['up']#['createTime', 'stylometryDis', 'topicDis', 'scoreMean', 'timeSpan', 'qA_LastActTime', 'qA_CommentCount', 'q_Tags']
    figure = plt.figure()
    i = 1
    for fea in featureList:
        results = np.zeros(nRound)
        for r in range(nRound):
            features = {}
            createTime = np.loadtxt(feature_path + 'CreateTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['createTime'] = createTime
            stylometryDis = np.loadtxt(feature_path + 'StylometryDis[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['stylometryDis'] = stylometryDis
            topicDis = np.loadtxt(feature_path + 'TopicDis_new[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['topicDis'] = topicDis
            scoreMean = np.loadtxt(feature_path + 'ScoreMean[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['scoreMean'] = scoreMean
            timeSpan = np.loadtxt(feature_path + 'TimeSpan[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['timeSpan'] = timeSpan
            qA_LastActTime= np.loadtxt(feature_path + 'QA_LastActTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_LastActTime'] = qA_LastActTime
            qA_CommentCount= np.loadtxt(feature_path + 'QA_CommentCount[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_CommentCount'] = qA_CommentCount
            q_Tags = np.loadtxt(feature_path + 'Q_Tags[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['q_Tags'] = q_Tags
        ##    feature_names = ['CreateTime[%d]'%i for i in range(createTime.shape[1])]
        ##    feature_names.extend(['StylometryDis[%d]'%i for i in range(stylometryDis.shape[1])])
        ##    feature_names.extend(['TopicDis[%d]'%i for i in range(topicDis.shape[1])])
        ##    feature_names.extend(['ScoreMean[%d]'%i for i in range(scoreMean.shape[1])])
        ##    feature_names.extend(['TimeSpan[%d]'%i for i in range(timeSpan.shape[1])])
        ##    feature_names.extend(['QA_LastActTime[%d]'%i for i in range(qA_LastActTime.shape[1])])
        ##    feature_names.extend(['QA_CommentCount[%d]'%i for i in range(qA_CommentCount.shape[1])])
        ##    feature_names.extend(['Q_Tags[%d]'%i for i in range(q_Tags.shape[1])])
        ##    data = np.column_stack((createTime, stylometryDis, topicDis, scoreMean, timeSpan, qA_LastActTime, qA_CommentCount, q_Tags))
            data = np.column_stack((stylometryDis, topicDis, timeSpan, qA_LastActTime, q_Tags))
##            data = np.delete(data, np.where((data== np.zeros((data.shape[0],1))).all(0)), 1)
        ##    feature_matrix = np.column_stack((createTime, stylometryDis, topicDis, scoreMean, timeSpan, qA_LastActTime, qA_CommentCount, q_Tags))
##            data = features[fea]
            clf = LinearDiscriminantAnalysis()#solver="lsqr", store_covariance=True)
            clf.fit(data,target)            
            classes = clf.classes_
            sc = clf.predict_proba(data)
            results[r] = ai.evaluation(target, classes[np.argmax(sc,axis=1)])[0]
##            print r,ai.evaluation(target, classes[np.argmax(sc,axis=1)]),sc
        plt.title('feature selection - up %d'%data.shape[1])
        plt.plot(results)
        i += 1
##    figure.subplots_adjust(left=0.05, bottom=0.04, right=0.98, top=0.95, wspace=0.25, hspace=0.16)
    figure.savefig(plot_path + 'feature selection - up %d'%data.shape[1] + '.png')
    plt.close()



def dimension_reduction():
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    for r in range(nRound):
        features = {}
        createTime = np.loadtxt(feature_path + 'CreateTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['createTime'] = createTime
        stylometryDis = np.loadtxt(feature_path + 'StylometryDis[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['stylometryDis'] = stylometryDis
        topicDis = np.loadtxt(feature_path + 'TopicDis_new[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['topicDis'] = topicDis
        scoreMean = np.loadtxt(feature_path + 'ScoreMean[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['scoreMean'] = scoreMean
        timeSpan = np.loadtxt(feature_path + 'TimeSpan[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['timeSpan'] = timeSpan
        qA_LastActTime= np.loadtxt(feature_path + 'QA_LastActTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['qA_LastActTime'] = qA_LastActTime
        qA_CommentCount= np.loadtxt(feature_path + 'QA_CommentCount[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['qA_CommentCount'] = qA_CommentCount
        q_Tags = np.loadtxt(feature_path + 'Q_Tags[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
        features['q_Tags'] = q_Tags
        data = np.column_stack((createTime, stylometryDis, topicDis, scoreMean, timeSpan, qA_LastActTime, qA_CommentCount, q_Tags))

        dimensionReduction(data, target, fea_alg = 'et')


def feature_1():
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    featureList = ['createTime', 'stylometryDis', 'topicDis', 'scoreMean', 'timeSpan', 'qA_LastActTime', 'qA_CommentCount', 'q_Tags']
    opt_mean = 0
    opt_fea = 'null'

##    for fea1, fea2 in itertools.product(featureList, featureList):
    for fea in featureList:
        results = np.zeros(nRound)
        for r in range(nRound):
            features = {}
            createTime = np.loadtxt(feature_path + 'CreateTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['createTime'] = createTime
            stylometryDis = np.loadtxt(feature_path + 'StylometryDis[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['stylometryDis'] = stylometryDis
            topicDis = np.loadtxt(feature_path + 'TopicDis_new[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['topicDis'] = topicDis
            scoreMean = np.loadtxt(feature_path + 'ScoreMean[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['scoreMean'] = scoreMean
            timeSpan = np.loadtxt(feature_path + 'TimeSpan[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['timeSpan'] = timeSpan
            qA_LastActTime= np.loadtxt(feature_path + 'QA_LastActTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_LastActTime'] = qA_LastActTime
            qA_CommentCount= np.loadtxt(feature_path + 'QA_CommentCount[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_CommentCount'] = qA_CommentCount
            q_Tags = np.loadtxt(feature_path + 'Q_Tags[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['q_Tags'] = q_Tags
            
            data = features[fea]
            clf = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True)
##            clf = ExtraTreesClassifier(n_estimators=300, random_state=0, max_features= "sqrt")
            clf.fit(data,target)
            classes = clf.classes_
            sample_class = clf.predict_proba(data)
            equiv = Equivalence(target)
            equiv.set_class(range(nSample),classes,sample_class)
            equiv.equivalence_class(range(nSample), 'jump points')
            sample_classes_predict = [equiv.equiv_class_question(ind) for ind in range(nSample)]
            results[r] = np.mean(evaluation(target, sample_classes_predict))
            
        slope = results[1:] - results[:-1]
        mean = np.mean(slope)
        std  = np.std(slope)
        print fea,data.shape[1],mean,std,slope,results
        if mean > opt_mean:
            opt_mean = mean
            opt_fea = fea
    return opt_fea

def feature_2():
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    featureList = ['createTime', 'stylometryDis', 'topicDis', 'scoreMean', 'timeSpan', 'qA_LastActTime', 'qA_CommentCount', 'q_Tags']
    opt_mean = 0
    opt_fea = ('null','null')

    for fea in itertools.combinations(featureList, 2):
        results = np.zeros(nRound)
        for r in range(nRound):
            features = {}
            createTime = np.loadtxt(feature_path + 'CreateTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['createTime'] = createTime
            stylometryDis = np.loadtxt(feature_path + 'StylometryDis[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['stylometryDis'] = stylometryDis
            topicDis = np.loadtxt(feature_path + 'TopicDis_new[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['topicDis'] = topicDis
            scoreMean = np.loadtxt(feature_path + 'ScoreMean[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['scoreMean'] = scoreMean
            timeSpan = np.loadtxt(feature_path + 'TimeSpan[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['timeSpan'] = timeSpan
            qA_LastActTime= np.loadtxt(feature_path + 'QA_LastActTime[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_LastActTime'] = qA_LastActTime
            qA_CommentCount= np.loadtxt(feature_path + 'QA_CommentCount[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['qA_CommentCount'] = qA_CommentCount
            q_Tags = np.loadtxt(feature_path + 'Q_Tags[%d].csv'%r, dtype=float, delimiter=',').reshape(nSample,-1)
            features['q_Tags'] = q_Tags
            
            data = np.column_stack((features[fea[0]],features[fea[1]]))           
            clf = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True)
##            clf = ExtraTreesClassifier(n_estimators=300, random_state=0, max_features= "sqrt")
            clf.fit(data,target)
            classes = clf.classes_
            sample_class = clf.predict_proba(data)
            equiv = Equivalence(target)
            equiv.set_class(range(nSample),classes,sample_class)
            equiv.equivalence_class(range(nSample), 'jump points')
            sample_classes_predict = [equiv.equiv_class_question(ind) for ind in range(nSample)]
            results[r] = np.mean(evaluation(target, sample_classes_predict))
            
        slope = results[1:] - results[:-1]
        mean = np.mean(slope)
        std  = np.std(slope)
        print fea,data.shape[1],mean,std,slope,results
        if mean > opt_mean:
            opt_mean = mean
            opt_fea = fea
    return opt_fea


'''
select best combination of feature sets. number of feature sets can be k = 1,2,...,8
criteria are: maximum and positive slope(diff), maximum metric result.
'''
def feature_n(k, fil):
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    featureList = ['createTime', 'stylometryDis', 'topicDis', 'scoreMean', 'timeSpan', 'qA_LastActTime', 'qA_CommentCount', 'q_Tags']
    opt_mean = 0
    opt_fea = None
    nFold = 5
    skf = StratifiedKFold(n_splits=nFold)
##    skf = StratifiedShuffleSplit(n_splits=nFold,test_size = nClass)

    for fea in itertools.combinations(featureList, k):
        results = np.zeros(nRound)
        for r in range(nRound):
            data = np.empty((nSample,0))
            for f in fea:
                feature = np.loadtxt(feature_path + '%s[%d].csv'%(f,r), dtype=float, delimiter=',').reshape(nSample,-1)
                data = np.column_stack((data,feature))
            for train_index, test_index in skf.split(data, target):
                break
            trainingLabels = target[train_index]
            testLabels = target[test_index]
            trainingSet = data[train_index]
            testSet = data[test_index]
##            clf = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True)
            clf = ExtraTreesClassifier(n_estimators=300, random_state=0, max_features= "sqrt")
            clf.fit(trainingSet,trainingLabels)
##            classes = clf.classes_ # sometimes classes!=np.unique(testLabels)
            sample_class = clf.predict_proba(testSet)
            equiv = Equivalence(testLabels)
            equiv.set_class(range(len(test_index)),np.unique(testLabels),sample_class)
            equiv.equivalence_class(range(len(test_index)), 'jump points')
            sample_classes_predict = [equiv.equiv_class_question(ind) for ind in range(len(test_index))]
            results[r] = np.mean(evaluation(testLabels, sample_classes_predict))
        slope = results[1:] - results[:-1]
        mean = np.mean(slope)
        std  = np.std(slope)
        minus = np.sum([i for i in slope if i < 0])
        print >>fil, fea,testSet.shape[1],mean,std,minus,slope,results
        if mean > opt_mean:
            opt_mean = mean
            opt_fea = (fea,testSet.shape[1],mean,std,minus,slope,results)
    return opt_fea

if __name__ == "__main__":
    print 'StratifiedKFold,first,ExtraTreesClassifier'
    fil = open('log.txt','w')
    for k in range(1,9):
        print k, feature_n(k,fil)
    fil.close()
    
##    print 'fea,data.shape[1],mean,std,slope,results'
##    fil = open('log2.txt','w')
##    print feature_n(8,fil)
##    fil.close()

    
    import os
    os.system('say -v "Anna" "Wunderbar"')
