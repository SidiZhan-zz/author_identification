# -*- coding: utf-8 -*-
'''
main control module
import os
os.chdir('/Users/sindyjen/Documents/codes/python/author_identification/v7')
execfile("author_identification.py")
'''
print(__doc__)

import db
from db import feature_path, clf_path, equiv_path, plot_path
import feature_extraction as fe
from equivalence import Equivalence
from analyse import fig0

from collections import Counter
import itertools
import time
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize,scale
##from random import shuffle
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
##from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
##from sknn.mlp import Classifier, Layer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
##from sklearn.calibration import CalibratedClassifierCV


# parameters and method selection...
featureList = ['createTime', 'stylometryDis', 'timeSpan', 'qA_CommentCount', 'q_Tags']
'''
all ['createTime', 'stylometryDis', 'topicDis', 'scoreMean', 'timeSpan', 'qA_LastActTime', 'qA_CommentCount', 'q_Tags']
first ['stylometryDis', 'topicDis', 'timeSpan', 'qA_LastActTime', 'Q_Tags']
second ['createTime', 'stylometryDis', 'timeSpan', 'qA_CommentCount', 'q_Tags']
'''
recommList = ['et']#['lda','et','svm','mlp']#['knn','svm','et','nb','lr','mlp','sknnmlp']
classifiers = {
    'lsvm':OneVsRestClassifier(LinearSVC(random_state=0)),
    'svm': SVC(gamma = 0.00001, C = 1000., probability = True, decision_function_shape = 'ovr'),
    'knn': KNeighborsClassifier(n_neighbors=10),#(n_neighbors=[5,len(test_index)/10,10][int(len(test_index)>50) + int(len(test_index)>100)]),
    'et': ExtraTreesClassifier(n_estimators=300, max_features= "sqrt"),
    'nb': GaussianNB(),
    'lr': LogisticRegression(),
    'mlp': MLPClassifier(),
    'lda': LinearDiscriminantAnalysis(),#solver="lsqr", store_covariance=True),
    'qda': QuadraticDiscriminantAnalysis(store_covariances=True)
##    'sknnmlp1': Classifier(
##        layers=[
##            Layer("Maxout", units=100, pieces=2),
##            Layer("Softmax")],
##        learning_rate=0.001,
##        n_iter=25)
##    'sknnmlp2': Classifier(layers=[
##            # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
##        Layer('Rectifier', units=200),
##        Layer('Softmax')],
##        learning_rate=0.01,
##        learning_rule='nesterov',
##        learning_momentum=0.9,
##        batch_size=300,
##        valid_size=0.0,
##        n_stable=10,
##        n_iter=10,
##        verbose=True)
}
threshList = ['mean']#,'median','percentile','mode']
equivaList = ['jump points']#['jump points','median','percentile','mode','mean'] # methods of generating equivalence class: equiv_alg
distriList = ['hist','boxplot','mode','mean','median','percentile'] #plot the round-equivalence value matric
identiList = ['uniqueness', 'accuracy','inclusion','closeness'] # transfer from question equivalence to asker identifiability
metriQList = ['Reciprocal Rank','Precision','Recall','Fmeasure']

nSample = 0
nClass = 0
nFeature = 0
nRound = 0
nFold = 5


def dataProcessing(data,r):
    data = np.delete(data, np.where((data== np.zeros((data.shape[0],1))).all(0)), 1)
    data = scale(data) # mean=0, std=1
##    data = normalize(data, norm = 'l2', axis=1)
##    data = normalize(data, norm = 'l2', axis=0)
##    data = normalize(data, norm = 'l1', axis=1)   
##    np.savetxt(feature_path + "data1[%d].csv"%r,data,delimiter=',')
    return data


##feature_ind = None
def dimensionReduction(data, target, fea_alg = 'et'): #featureSelection(trainingSet,trainingLabels,testSet,testLabels,fea_alg = 'dt'):
##    global feature_ind
##    if  feature_ind != None:
##        return feature_ind

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
        feature_rank = clf.feature_importances_
        ind = np.argsort(feature_rank)[::-1]
        max_score = 0
        max_i = 0
        for i in range(0,len(ind)+1,10):
            score = clf.fit(trainingSet[:,ind[:i+1]], trainingLabels).score(testSet[:,ind[:i+1]], testLabels)
            if score>max_score:
                max_score = score
                max_i = i
        feature_ind = ind[:max_i+1]
        
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
        
    print 'ranked by sigma_ asc, max_feature=ind[:%d+1], max_score=%f'%(max_i,max_score)#, 'max_feature_ind=',feature_ind
    return feature_ind


# post_equivalence = 1*nSample
def threshold(jump_points, sample_class_diff, metric=''):
    nSample = sample_class_diff.shape[0]
    diffs = sample_class_diff.flatten()
    post_equivalence = [nSample for i in range(nSample)]
    thres = 0
    if metric == 'mean':
        thres = sample_class_diff.mean()
    elif metric == 'median':
        thres = np.median(diffs)
    elif metric == 'mode':
        around = np.around(diffs,decimals=5)
        thres= stats.mode(around).mode[0]
        if thres == 0:
            around = [a for a in around if a != 0]
            thres = np.min(around)
    elif metric == 'percentile':
        thres = np.percentile(diffs, 75)
        
    for i in range(nSample):
        ind,value = jump_points[i]
        if value>=thres:
            post_equivalence[i] = ind + 1
    return post_equivalence


def equivalence_class(sample_class, equiv = "jump points"):
    sample_classes_predict = None
    if equiv == "jump points":
        sample_class_sorted = np.sort(sample_class, axis=1)[:,::-1] # larger, more precise
        sample_class_ind_sorted = np.argsort(sample_class, axis=1)[:,::-1]
        sample_class_diff = sample_class_sorted - np.column_stack((sample_class_sorted[:,1:],sample_class_sorted[:,-1]))
        inds = np.argmax(sample_class_diff, axis=1)
        jump_points = [(inds[i],sample_class_diff[i][inds[i]]) for i in range(len(inds)) ]
        
    ##    sample_classes_sorted = []# label of all classes sorted by score
        sample_classes_predict = []# top label of recommendations sorted by score
        for i in range(len(test_index)):
            sample_classes_predict.append([classes[sample_class_ind_sorted[i,ind]] for ind in range(inds[i]+1)])
    ##        sample_classes_sorted.append(classes[sample_class_ind_sorted[i,:]])
    ##    post_equivalence = threshold(jump_points,sample_class_diff,thresh)
    return sample_classes_predict


'''
id_alg in ['uniqueness', 'accuracy','inclusion','closeness']
'''
def identifiability(equiv, id_alg = "uniqueness"):
    iden = 0.0
    nUser = len(equiv.classes)
    nQuestion = equiv.sample_class.shape[0]
    if id_alg == "uniqueness":
        for user in equiv.classes:
            q_indices = np.where(equiv.target == user)[0]
            add = len([row for row in equiv.sample_eqClass[q_indices] if len(np.where(row != 0)[0])==1])
            iden += float(add)/len(q_indices)
    elif id_alg == "accuracy":
        for user in equiv.classes:
            q_indices = np.where(equiv.target == user)[0]
            add = len([row for row in equiv.sample_eqClass[q_indices] if len(np.where(row != 0)[0])==1 and equiv.classes[np.argmax(row)] == user])
            iden += float(add)/len(q_indices)
    elif id_alg == "inclusion":
        for user in equiv.classes:
            q_indices = np.where(equiv.target == user)[0]
            add = len([row for row in equiv.sample_eqClass[q_indices] if user in equiv.classes[np.where(row!=0)[0]]])
            iden += float(add)/len(q_indices)
    elif id_alg == "closeness":
        for user in equiv.classes:
            q_indices = np.where(equiv.target == user)[0]
            add = np.sum([1.0/(np.where(equiv.classes[np.argsort(row)[::-1]] == user)[0][0]+1 + len(np.where(row != 0)[0])) for row in equiv.sample_eqClass[q_indices] if user in equiv.classes[np.where(row!=0)[0]]])
            iden += float(add)/len(q_indices)
    iden = iden / nUser
    return iden


def evaluation(trueLists, predictedLists):
    accur = []
   
##    metric = 'Reciprocal Rank', avg(1/rank/predict size)
    avg = []
    for i in range(len(trueLists)):
        trueList = trueLists[i]
        if type(trueList) != list:
            trueList = [trueList]
        else:
            trueList = list(trueList)
        predictList = predictedLists[i]
        if type(predictList) != list:
            predictList = [predictList]
        else:
            predictList = list(predictList)
            
        count = 0.0
        for label in predictList:
            if label in trueList:
                count = count + 1.0/(predictList.index(label)+1)
        if len(predictList) !=0:
            avg.append(count/len(predictList))
        else:
            avg.append(0)
    accur.append(np.mean(avg))
    
##    metric = 'Precision', avg(hit/predict size)
    avg = []
    for i in range(len(trueLists)):
        trueList = trueLists[i]
        if type(trueList) != list:
            trueList = [trueList]
        else:
            trueList = list(trueList)
        predictList = predictedLists[i]
        if type(predictList) != list:
            predictList = [predictList]
        else:
            predictList = list(predictList)
            
        count = 0.0
        for label in predictList:
            if label in trueList:
                count = count+1.0
        if len(predictList) !=0:
            avg.append(count/len(predictList))
        else:
            avg.append(0)
    accur.append(np.mean(avg))

##    metric = "Recall", avg(hit/true size)
    avg = []
    for i in range(len(trueLists)):
        trueList = trueLists[i]
        if type(trueList) != list:
            trueList = [trueList]
        else:
            trueList = list(trueList)
        predictList = predictedLists[i]
        if type(predictList) != list:
            predictList = [predictList]
        else:
            predictList = list(predictList)
            
        count = 0.0
        for label in predictList:
            if label in trueList:
                count = count+1.0
        if len(trueList) !=0:
            avg.append(count/len(trueList))
        else:
            avg.append(0)
    accur.append(np.mean(avg))
       
##    metric = "Fmeasure"
    value = 0.0
    if accur[1]!=0 and accur[2]!=0:
        value = 2/(1/accur[1]+1/accur[2])
    accur.append(value)
##    print accur,trueLists,predictedLists
    return accur



'''
['mean','median','percentile','mode','hist','boxplot']
'''
def plotDistri(plt, i, results, distri):
    nRound,nSample = results.shape
    xlabel = 'equivalence'
    ylabel = 'frequency'
    ax = plt.subplot(2,3,i)

    if distri == 'hist':
        yaxis = None
        bins = range(int(np.min(results)), int(np.max(results)) + 2)
        for r in range(nRound):
            yaxis0, xaxis0 = np.histogram(results[r,:], bins = bins)
            if r == 0:
                yaxis = yaxis0
            else:
                yaxis = np.vstack((yaxis,yaxis0))
        xaxis = xaxis0[:-1] #(xaxis0[:-1] + xaxis0[1:]).astype(float)/2
        for r in range(yaxis.shape[0]):
            ax.plot(xaxis, yaxis[r], '-', alpha = 1 - 0.7/nRound*r, label='r=%d'%r)
            plt.axvline(x=np.percentile(yaxis[r],95), color='c', linestyle='--')
        distri = 'histogram (v-line = 95%)' #draw vertical line of 95% percentile
        plt.legend(loc='upper center', shadow=False, ncol=2, fontsize=5)

    elif distri == 'boxplot':
        yaxis = np.transpose(results) # So one column in values generate one box in plot
        ax.boxplot(yaxis)
        xlabel = 'round'
        ylabel = 'equivalence'
        
    elif distri == 'mean':
        yaxis = np.mean(results, axis = 1)
        ax.plot(yaxis)
        xlabel = 'round'
        ylabel = 'equivalence'
        plt.xticks(range(nRound))
##        plt.yticks(range(int(np.min(results)), int(np.max(results)) + 1, int(np.max(results)/20)))
            
    elif distri == 'median':
        yaxis = np.median(results, axis = 1)
        ax.plot(yaxis)
        xlabel = 'round'
        ylabel = 'equivalence'
        plt.xticks(range(nRound))
##        plt.yticks(range(int(np.min(results)), int(np.max(results)) + 1, int(np.max(results)/20)))
        
    elif distri == 'mode':
        yaxis = [i[0] for i in stats.mode(results, axis = 1).mode]
        ax.plot(yaxis)
        xlabel = 'round'
        ylabel = 'equivalence'
        plt.xticks(range(nRound))
##        plt.yticks(range(int(np.min(results)), int(np.max(results)) + 1, int(np.max(results)/20)))

    elif distri == 'percentile':
        yaxis = np.percentile(results, 75, axis = 1)
        ax.plot(yaxis)
        xlabel = 'round'
        ylabel = 'equivalence'
        distri == 'percentile = 75%'
        plt.xticks(range(nRound))
##        plt.yticks(range(int(np.min(results)), int(np.max(results)) + 1, int(np.max(results)/20)))

##    np.savetxt(plot_path + title + ".csv", yaxis, delimiter=',')
    ax.set_title(distri)
    plt.xlabel(xlabel, fontsize=10, color='black')
    plt.ylabel(ylabel, fontsize=10, color='black')



def plotMetricUser(values, metric, title):
##    np.savetxt(plot_path + title + '.csv', values, delimiter=',')
    fig = plt.figure() # create a figure
    plt.plot(values)
    plt.xlabel('round', fontsize=10, color='black')
    plt.ylabel(metric, fontsize=10, color='black')
    plt.title(title)
    fig.savefig(plot_path + title + '.png')
    plt.close(fig)
    
# results = nRound * nMetric
def plotMetric(results,metricList, title):
##    np.savetxt(plot_path + title + '.csv', results, delimiter=',')
    if len(metricList)>4:
        for i in range(len(metriQList)):
            plotMetricUser(metrics[:,i], metriQList[i], title+' metric='+metricList[i])
    else:
        fig = plt.figure() # default plt.figure(figsize=(6.4,6.4),dpi=100)
        for i in range(len(metricList)):
            plt.subplot(len(metricList),1,i+1)
            plt.plot(results[:,i].flatten())
            plt.ylabel(metricList[i], fontsize=10, color='black')
            if i == 0:
                plt.title(title)
            if i == len(metricList)-1:
                plt.xlabel('round', fontsize=10, color='black')
                plt.xticks(range(results.shape[0]))
        fig.savefig(plot_path + title + '.png')
        plt.close(fig)




'''
print itemList of questions under the author of the quesion
'''
def checkInText(equiv,q_index=0):
    f = open(equiv_path + 'compare_itemLists_of_true_predict_user.txt','w')
    print >> f, '============real user itemlist============'
    u_label = target[q_index]
    print >> f, '========author.id',u_label
    q_indices = equiv.equiv_question_user_true(u_label=u_label)
    for ind in q_indices:
        print >> f, '==question.id',db.questions[ind].get("Id")
        for item in db.itemLists[ind]:
            print >> f, ET.tostring(item)
    print >> f, '============predicted users itemlists============'
    u_labels = equiv.equiv_class_question(q_index=q_index)
    for u_label in u_labels:
        print >> f, '========author.id',u_label
        q_indices = equiv.equiv_question_user_true(u_label=u_label)
        for ind in q_indices:
            print >> f, '==question.id',db.questions[ind].get("Id")
            for item in db.itemLists[ind]:
                print >> f, ET.tostring(item)
    f.close()
    
if __name__ == "__main__":
    logFile = open(feature_path + "log.txt",'w')
##    st = time.time()
    
    # target, dtype='|S7'
##    target = [question.get("OwnerUserAccountId") for question in db.questions]
##    nSample = len(target)
##    nClass = len( set(target) )
##    target = np.array(target)
##    np.savetxt(clf_path + "target.csv", target,  fmt='%s', delimiter=',')
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))

    
##    print 'target = ',time.time() - st
##    st = time.time()
    nRound = db.nRound()
##    print 'nRound = ',time.time() - st
    
    
    # calculate all 
    for recom, thresh, equiv_alg in itertools.product(recommList, threshList, equivaList):
        iterTitle = 'recom=%s thresh=%s equiv=%s'%(recom,thresh,equiv_alg)
        ev = np.zeros((nRound,nSample)) # each post: size of equivalence class
        ev_qq = np.zeros((nRound,nSample)) # each post: size of equivalence class
        metrics = np.zeros((nRound,len(metriQList))) 
        identi = np.zeros((nRound,len(identiList)))
        for r in range(nRound):
            print '==========BEGIN: round:%d, %s'%(r,iterTitle)
            print >> logFile, '==========BEGIN: round:%d, %s'%(r,iterTitle)
            # =======================feature, data, preprocessing===================================
            st = time.time()
            data = np.empty((nSample,0))
            for f in featureList:
                feature = np.loadtxt(feature_path + '%s[%d].csv'%(f,r), dtype=float, delimiter=',').reshape(nSample,-1)
                data = np.column_stack((data,feature))
            print 'data.shape',data.shape
            data = data[:,dimensionReduction(data, target, fea_alg = 'et')]
            print 'after featureSelection data.shape',data.shape
            np.savetxt(feature_path + "data[%d].csv"%r, data, delimiter=',') # raw feature matrix
            
##            createTime = fe.CreateTime(r)
##            stylometryDis = fe.StylometryDis(r)
##            topicDis = fe.TopicDis(r)
##            scoreMean = fe.ScoreMean(r)
##            timeSpan = fe.TimeSpan(r)
##            qA_LastActTime= fe.QA_LastActTime(r)
##            qA_CommentCount= fe.QA_CommentCount(r)
##            q_Tags = fe.Q_Tags(r) 
##            data = np.column_stack((createTime, stylometryDis, topicDis, scoreMean, timeSpan, qA_LastActTime, qA_CommentCount, q_Tags))
##            np.savetxt(feature_path + "data[%d].csv"%r, data, delimiter=',') # raw feature matrix
            
##            data = np.loadtxt(feature_path + 'data1[%d].csv'%r, dtype=float, delimiter=',')
            
            if recom == 'svm':  # scores for each class are quite similar, sometimes smallest the best??
                data = dataProcessing(data,r)
            elif recom == 'knn':
                data = dataProcessing(data,r)
            elif recom == "et":
                data = dataProcessing(data,r)
            elif recom == "nb": # sometimes only 0 or 1??
                data = dataProcessing(data,r)
            elif recom == 'rl':
                data = dataProcessing(data,r)
            elif recom == 'mlp':
                data = np.delete(data, np.where((data== np.zeros((data.shape[0],1))).all(0)), 1)
            nFeature = data.shape[1]
            print 'after dataProcessing data.shape',data.shape,'runtime',time.time()-st

            # =======================cross validation, scoring by clf===================================
            #cross-validation (5-fold)
            #nFold = 5
            n = 0
            equiv = Equivalence(target)
            metric1 = np.empty((5,4))
            metric2 = np.empty((5,4))
            skf = StratifiedKFold(n_splits=nFold)
            st = time.time()
            for train_index, test_index in skf.split(data, target):
                trainingSet = data[train_index]
                trainingLabels = target[train_index]
                testSet = data[test_index]
                testLabels = target[test_index]

                clf = classifiers[recom]
                clf.fit(trainingSet,trainingLabels)
                classes = clf.classes_
                try:
                    df = clf.decision_function(testSet)
                    df = df - np.min(df,axis=1).reshape((-1,1)) * np.ones((1,nClass))
                    df = df / (np.sum(df,axis=1).reshape((-1,1)) * np.ones((1,nClass)))
                except:
                    print 'clf.decision_function(testSet) error!'
                    df = clf.predict_proba(testSet)
                try:
                    pp = clf.predict_proba(testSet)
                except:
                    print 'clf.predict_proba(testSet) error!'
                    df = clf.decision_function(testSet)
                    df = df - np.min(df,axis=1).reshape((-1,1)) * np.ones((1,nClass))
                    pp = df / (np.sum(df,axis=1).reshape((-1,1)) * np.ones((1,nClass)))
                edf = evaluation(testLabels, classes[np.argmax(df,axis=1)])
                epp = evaluation(testLabels, classes[np.argmax(pp,axis=1)])
                if np.mean(evaluation(testLabels, classes[np.argmax(df,axis=1)])) > np.mean(evaluation(testLabels, classes[np.argmax(pp,axis=1)])):
                    sample_class = df
                else:
                    sample_class = pp
                
                metric1[n] = evaluation(testLabels,classes[np.argmax(sample_class,axis=1)])
                metric2[n] = evaluation(testLabels, clf.predict(testSet))
##                print r,n,metric1[n-1],metric2[n-1]                   
                equiv.set_class(test_index,classes,sample_class)
                n += 1
            print 'runtime',time.time()-st
            # =======================equivalence, identification, evaluation===================================
            # equiv from equiv.sample_eqClass:
            equiv.equivalence_class(range(nSample), equiv_alg)
            ev[r] = equiv.equiv_value(entity = 'q')
            # metric
            sample_classes_predict = [equiv.equiv_class_question(ind) for ind in range(nSample)]
            metrics[r] = evaluation(target, sample_classes_predict)
            print 'predict_proba', metric1.mean(axis=0)
            print 'predict', metric2.mean(axis=0)
            print 'equiv.sample_eqClass metrics',metrics[r]
            print 'equiv.sample_class metric',evaluation(target,equiv.classes[np.argmax(equiv.sample_class,axis=1)])
            print >> logFile, 'predict_proba', metric1.mean(axis=0)
            print >> logFile, 'predict', metric2.mean(axis=0)
            print >> logFile, 'equiv.sample_eqClass metrics',metrics[r]
            print >> logFile, 'equiv.sample_class metric',evaluation(target,equiv.classes[np.argmax(equiv.sample_class,axis=1)])
            # identifi
            for i in range(len(identiList)):
                identi[r,i] = identifiability(equiv, id_alg = identiList[i])
            # save the components of equiv: sample_eqClass, sample_class
            np.savetxt(equiv_path + "sample_classes_predict[%d][recom=%s equiv=%s].csv"%(r,recom,equiv_alg), sample_classes_predict,  fmt='%s', delimiter=',')
            np.savetxt(equiv_path + "equiv_sample_eqClass[%d][recom=%s equiv=%s].csv"%(r,recom,equiv_alg), equiv.sample_eqClass,  delimiter=',')
            np.savetxt(clf_path + "sample_class[%d][%s].csv"%(r,recom), equiv.sample_class,  delimiter=',')
            np.savetxt(clf_path + "classes[%d][%s].csv"%(r,recom), equiv.classes,  fmt='%s', delimiter=',')
            # middle results:3 values: 3 * nSample = [size of EC, top matches, in, ]
            middle_results = np.zeros((3,nSample))
            middle_results[0] = ev[r]
            middle_results[1] = [[0,1][target[i] == sample_classes_predict[i][0]] for i in range(nSample)]
            middle_results[2] = [[0,1][target[i] in sample_classes_predict[i]] for i in range(nSample)]
            np.savetxt(equiv_path + "middle_results[%d][recom=%s equiv=%s].csv"%(r,recom,equiv_alg), middle_results, delimiter=',')
            
            print '==========END: round:%d, %s'%(r,iterTitle)
            print >> logFile, '==========END: round:%d, %s'%(r,iterTitle)
##            input('...')

        # =======================visualization analysis: equiv, identif, metric===================================
        figure = plt.figure(figsize=(10,5.1))
        st = figure.suptitle("equivalence - " + iterTitle, fontsize="x-large")
        for i in range(0,len(distriList)):
            distri = distriList[i]
            plotDistri(plt,i+1, ev, distri)
        figure.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.85, wspace=0.34, hspace=0.75)
        st.set_y(0.95)
        figure.savefig(plot_path + "equivalence - " + iterTitle + '.png')
        plt.close(figure)
        
        fig0()

        plotMetric(metrics,metriQList, title = "metric - %s"%(iterTitle))
        
        plotMetric(identi,identiList, title = "identifiability - %s"%(iterTitle))
    logFile.close()
