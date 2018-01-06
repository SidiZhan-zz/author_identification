# -*- coding: utf-8 -*-
"""
execfile("analyse.py")
analyse middle results by calling functions here
generate the five figures here
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import itertools

import db
from db import feature_path, clf_path, equiv_path, plot_path
import author_identification as ai
from equivalence import Equivalence

fig = None

##equiv_path = '/Users/sindyjen/Documents/codes/python/author_identification/v7/equiv_small/'

def toFile():
    fil = open('analyse.txt','w')
    nRound = db.nRound()
    thresh = 'mean'
    for recom, equiv_alg in itertools.product(recommList, equivaList):
        for r in range(nRound):
            print 'r=%d, recom=%s, equiv=%s, thresh=%s'%(r,recom,equiv_alg,thresh)
            fil.write('r=%d, recom=%s, equiv=%s, thresh=%s'%(r,recom,equiv_alg,thresh))
            middle_results = np.loadtxt(equiv_path + "middle_results[%d][recom=%s equiv=%s].csv"%(r,recom,equiv_alg), delimiter=',')
            x,y,z = 0,0,0
            for i in range(middle_results.shape[1]):
                if middle_results[0,i]==1 and middle_results[1,i]==1:
                    x = x + 1
                if middle_results[2,i]==1:
                    y = y + 1
                if middle_results[0,i]==1:
                    z = z + 1
            print float(x)/middle_results.shape[1],float(y)/middle_results.shape[1],float(z)/middle_results.shape[1] 
            fil.write('\t%f\t%f\t%f\n'%(float(x)/middle_results.shape[1], float(y)/middle_results.shape[1], float(z)/middle_results.shape[1]))

def plotMiddleResults(plt,i, key, plotType, xticks, yticks, xlabel, ylabel):
    global fig
    values = fig[key]
    ax = plt.subplot(4, 3, i)
    if plotType == 'hist':
        nRound,nSample = values.shape
        yaxis = None
        bins = range(int(np.min(values)), int(np.max(values)) + 2)
        for r in range(nRound):
            yaxis0, xaxis0 = np.histogram(values[r,:], bins = bins)
            if r == 0:
                yaxis = yaxis0
            else:
                yaxis = np.vstack((yaxis,yaxis0))
        xaxis = xaxis0[:-1]
        for r in range(yaxis.shape[0]):
            plt.plot(xaxis, yaxis[r], '-', alpha = 1 - 0.7/nRound*r, label='Round%d'%r)
    elif plotType == 'plot':
        ax.plot(values)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=10, color='black')
    plt.ylabel(ylabel, fontsize=10, color = 'black')
    ax.set_title(key)


def fig0():
    global fig
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    nRound = db.nRound()
    thresh = 'mean'
    for recom, equiv_alg in itertools.product(['lda','et','svm','mlp'], ['jump points','median','percentile','mode','mean']):
        title = 'recom=%s equiv=%s thresh=%s'%(recom,equiv_alg,thresh)
        print title
        figure = plt.figure(figsize=(15, 8))
        st = figure.suptitle('f0 ' + title, fontsize="x-large")
        fig = dict(('f%d'%i,np.zeros((nRound),dtype=float)) for i in range(1,9))
        fig['f1.5'] = np.zeros((nRound,nSample))
        fig['f6.5'] = np.zeros((nRound,nSample))
        for r in range(nRound):
            middle_results = np.loadtxt(equiv_path + "middle_results[%d][recom=%s equiv=%s].csv"%(r,recom,equiv_alg), delimiter=',')
            classes = np.loadtxt(clf_path + "classes[%d][%s].csv"%(r,recom), dtype=str, delimiter=',')
            sample_class = np.loadtxt(clf_path + "sample_class[%d][%s].csv"%(r,recom), delimiter=',')
            fig['f1.5'][r] = middle_results[0]
            fig['f2'][r] = np.mean(middle_results[0])
            fig['f3'][r] = np.std(middle_results[0])
            rankInds = np.zeros((nSample))
            for i in range(nSample):
                if middle_results[0,i]==1:
                    fig['f1'][r] = fig['f1'][r] + 1.0/nSample
                if middle_results[0,i]==1 and middle_results[1,i]==1:
                    fig['f4'][r] = fig['f4'][r] + 1.0/nSample
                if middle_results[2,i]==1:
                    fig['f5'][r] = fig['f5'][r] + 1.0/nSample
                rankList = [classes[ind] for ind in np.argsort(sample_class[i])[::-1]]
                fig['f6'][r] = fig['f6'][r] + (rankList.index(target[i])+1)/nSample
                rankInds[i] = rankList.index(target[i])+1
            fig['f6.5'][r] = rankInds
            fig['f7'][r] = np.mean(rankInds)
            fig['f8'][r] = np.std(rankInds)
        # (key, plotType, title, ticks, ticks, xlabel, ylabel)
        plotMiddleResults(plt,1,'f1', 'plot', range(nRound), np.linspace(0,1,11), 'round', "proportion of |EC|=1")
        plotMiddleResults(plt,2,'f1.5', 'hist', range(1,nRound+1), range(0,nSample+1,nSample/10), 'equivalence', 'frequency')
        plotMiddleResults(plt,3,'f2', 'plot', range(nRound), range(0,nRound+1), 'round', "mean of |EC|")
        plotMiddleResults(plt,4,'f3', 'plot', range(nRound), range(0,10), 'round', "standard deviation of |EC|")
        plotMiddleResults(plt,5,'f4', 'plot', range(nRound), np.linspace(0,1,11), 'round', "proportion of EC=TL")
        plotMiddleResults(plt,6,'f5', 'plot', range(nRound), np.linspace(0,1,11), 'round', "proportion of TL in EC")
        plotMiddleResults(plt,7,'f6', 'plot', range(nRound), np.linspace(0,1,11), 'round', "proportion of EC[0]=TL")
        plotMiddleResults(plt,8,'f6.5', 'hist', range(1,nClass+1,nClass/15), range(0,nSample+1,nSample/10), "rank", "frequency")
        plotMiddleResults(plt,9,'f7', 'plot', range(nRound), range(1,nClass+1,nClass/10), 'round', "mean of rank")
        plotMiddleResults(plt,10,'f8', 'plot', range(nRound), range(0,10), 'round', "standard deviation of rank")
        figure.subplots_adjust(left=0.05, bottom=0.08, right=0.99, top=0.85, wspace=0.17, hspace=0.75)
        st.set_y(0.95)
        figure.savefig(plot_path + 'f0 ' + title + '.png')
        plt.close(figure)

'''
['mean','median','percentile','mode','hist','boxplot']
'''
def plotDistri(plt, i, results, distri):
    nRound,nSample = results.shape
    
    # plot
    xlabel = 'equivalence'
    ylabel = 'frequency'
    ax = plt.subplot(2,3,i)
    # plot methods
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


'''
call plotDistri() to plot equivalence
'''
def figEquivClass():
    from equivalence import Equivalence
    from scipy import stats
    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nRound = db.nRound()
    distriList = ['hist','boxplot','mode','mean','median','percentile']
    for recom, thresh, equiv_alg in itertools.product(['et','svm','mlp'], ['mean'], ['jump points','median','percentile','mode','mean']):
        title = 'recom=%s equiv=%s thresh=%s'%(recom,equiv_alg,thresh)
        print title
        ev = np.zeros((nRound,nSample))
        figure = plt.figure(figsize=(10,5.1))
        st = figure.suptitle("equivalence - " + title, fontsize="x-large")
        for r in range(nRound):
            equiv = Equivalence(target)
            equiv.sample_class = np.loadtxt(clf_path + "sample_class[%d][%s].csv"%(r,recom),  delimiter=',')
            equiv.classes = np.loadtxt(clf_path + "classes[%d][%s].csv"%(r,recom),  dtype= str, delimiter=',')
            equiv.equivalence_class(range(nSample), equiv_alg)
            ev[r] = equiv.equiv_value(entity = 'q')
        for i in range(0,len(distriList)):
            distri = distriList[i]
            plotDistri(plt,i+1, ev, distri)
        figure.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.85, wspace=0.34, hspace=0.75)
        st.set_y(0.95)
        figure.savefig(plot_path + "equivalence - " + title + '.png')
##        plt.show()
        plt.close(figure)
##        input('...')


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
    
if __name__ == "__main__":
##    fig0()
##    figEquivClass()
    # equivalence
##    figEquivClass()
    # identifiability

    target = np.loadtxt(clf_path + "target.csv", dtype=str, delimiter=',')
    nSample = len(target)
    nClass = len(np.unique(target))
    
    nRound = db.nRound()
    
    for recom, thresh, equiv_alg in itertools.product(['et'], ['mean'], ['jump points']):
        iterTitle = 'recom=%s thresh=%s equiv=%s'%(recom,thresh,equiv_alg)
        identi = np.zeros((nRound,len(ai.identiList)))
        for r in range(nRound):
            sample_class = np.loadtxt(clf_path + "sample_class[%d][%s].csv"%(r,recom),delimiter=",")
            classes = np.loadtxt(clf_path + "classes[%d][%s].csv"%(r,recom), dtype=str, delimiter = ',')
            equiv = Equivalence(target)
            equiv.set_class(range(nSample),classes,sample_class)
            equiv.equivalence_class(range(nSample),equiv_alg)
            for i in range(len(ai.identiList)):
                identi[r,i] = identifiability(equiv, id_alg = ai.identiList[i])
        tmp = input(identi)
        ai.plotMetric(identi,ai.identiList, title = "identifiability - %s"%(iterTitle))
