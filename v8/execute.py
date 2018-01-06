# -*- coding: utf-8 -*-
"""
execfile("execute.py")
"""
print(__doc__)

def decision2proba(sc):
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

def RBF_C_gamma():fe
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

if __name__ == "__main__":
##    sc_n = decision2proba(sc)
##    evaluation(testLabels, classes[np.argmax(sc_n,axis=1)])

##    C_range = np.logspace(-2, 10, 13)
##    gamma_range = np.logspace(-9, 3, 13)
##    param_grid = dict(gamma=gamma_range, C=C_range)
##    from sklearn.model_selection import StratifiedShuffleSplit
##    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
##    from sklearn.model_selection import GridSearchCV
##    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
##    grid.fit(data, target)
##    print("The best parameters are %s with a score of %0.2f"%(grid.best_params_, grid.best_score_))
    for r in range(nRound):
        createTime = fe.CreateTime(r)
        stylometryDis = fe.StylometryDis(r)
        topicDis = fe.TopicDis(r)
        scoreMean = fe.ScoreMean(r)
        timeSpan = fe.TimeSpan(r)
        qA_LastActTime= fe.QA_LastActTime(r)
        qA_CommentCount= fe.QA_CommentCount(r)
        q_Tags = fe.Q_Tags(r) 
        data = np.column_stack((createTime, stylometryDis, topicDis, scoreMean, timeSpan, qA_LastActTime, qA_CommentCount, q_Tags))
        np.savetxt(feature_path + "data[%d].csv"%r, data, delimiter=',') # raw feature matrix
