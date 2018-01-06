# -*- coding: utf-8 -*-
'''
execfile("feature_extraction.py")
'''
import nltk
import numpy as np
from datetime import datetime
import re
from db import tags, askers, questions, comments, answers
from db import feature_path, clf_path, equiv_path, plot_path, lda_path
import db
import lda

nQuestion = len(questions)
timeInterval = [(0,6),(7,11),(12,16),(17,17),(18,18),(19,19),(20,20),(21,23)]

# lda
nTopic = 500
nTopTopic = 20
nTopWord = 50
corpusKey = None
modelTrain = None
modelTest = None

# accumulate, or define once
CallTimes = dict.fromkeys(['CreateTime','StylometryDis','TopicDis','ScoreMean','TimeSpan','QA_LastActTime','QA_CommentCount','Q_Tags'], np.zeros((nQuestion,1)))
CreateTimeCount = None
LastTime = None
TimeDiff = None
TopicDistribution = None
ldaTestset = None
LastActTimeDiff = None
StylometryAccu = None
ScoreSum = None
TagDis = None
TagList = None
CommentCount = None

# for multicolumn feature, divide rows by their sum
def np_row(f):
##    f = f.astype(float)
##    if f.shape[1]<=1:
##        return f
##    f = f / np.vstack(f.sum(axis=1))
##    return np.nan_to_num(f)
    return f.astype(float)

def itemsByRound(r):
    return db.itemsByRound(r)
    # temp
##    tmp = [itemLists[212],itemLists[235]]
##    lis = []
##    for itemList in tmp:
##        if r < len(itemList):
##            lis.append(itemList[r])
##        else:
##            item = ET.fromstring("""<row Id="-1" PostId="" Score="" Text="" CreationDate="" UserId="" />""")
##            lis.append(item)
##    return lis


'''
[8,11],[12,16],[17,21],[22,7]
'''
def timeSlot(timeStrs):
    f = np.zeros((len(timeStrs),len(timeInterval)),dtype = np.int)
    for i in range(len(timeStrs)):
        if timeStrs[i] == "":
            continue
        h = datetime.strptime(timeStrs[i], "%Y-%m-%dT%H:%M:%S.%f").time().hour # # "2016-01-12T18:57:48.103"
        for j in range(len(timeInterval)):
            t = timeInterval[j]
            if h>=t[0] and h<=t[1]:
                f[i,j]=1
                break
    return f


def ldaTrain():
    global modelTrain
    global corpusKey
    import time
    st = time.time()
    # preparing word keys as features
    documents = [re.findall('[\w\'-]*[A-Za-z]+',re.sub('<[\w/]+>|\\n','',(question.get("Body") + " " + question.get("Title","")).lower())) for question in questions]
    documents.extend([re.findall('[\w\'-]*[A-Za-z]+',re.sub('<[\w/]+>|\\n','',(answer.get("Body")).lower())) for answer in answers])
    documents.extend([re.findall('[\w\'-]*[A-Za-z]+',re.sub('<[\w/]+>|\\n','',(comment.get("Text")).lower())) for comment in comments]) # len(documents)=21615
    text = []
    for document in documents:
        text.extend(document)
    from collections import Counter
    corpus = Counter(text) # 1769684 -> 70564
    stopwords = re.split("\n",re.sub('\r','',open(lda_path + "stopwords.txt",'r').read())) #427
    corpusKey = [item[0] for item in corpus.items() if item[0] not in stopwords and item[1]>=10 and item[1]<10000] # 70564 -> 9068
    # save corpusKey
    open(lda_path + "lda_corpusKey.txt",'w').write("\n".join(corpusKey))
    print("--- keys %s seconds ---" % (time.time() - st))
    st = time.time()

    # trainset(post,key) = word count
    trainset = np.zeros((len(documents),len(corpusKey)),dtype=int)
    for i in range(len(documents)):
        for word in documents[i]:
            if word in corpusKey:
                trainset[i,corpusKey.index(word)] = trainset[i,corpusKey.index(word)]+1
    print("--- trainset %s seconds ---" % (time.time() - st))
    st = time.time()
              
    model = lda.LDA(n_topics=nTopic, n_iter=1000, random_state=1)
    model.fit(trainset)
    
    model.components_.tofile(lda_path + 'lda_components.dat') # when model.topic_word_ is defined, the model can be used to transform testset, with alias model.topic_word_, (topic,word) = value is higher means word is more important in the topic

    np.array(model.loglikelihoods_).tofile(lda_path + 'lda_loglikelihoods.dat')  # as looping, likelihoods go up (negative number)
    model.doc_topic_.tofile(lda_path + 'lda_doc_topic.dat') # probability of doc belongs to the topic
    
    fil = open(lda_path + 'lda_topics.txt','w')
    for i, topic_dist in enumerate(model.topic_word_): # model.components_ also works
        topic_words = np.array(corpusKey)[np.argsort(topic_dist)][:-(nTopWord+1):-1]
        fil.write('Topic {}: {}'.format(i, ' '.join(topic_words)))
        fil.write('\n')
    fil.close()
    print("--- train %s seconds ---" % (time.time() - st))
    modelTrain = model
    

# input document list, output document-topicFrequency matrix
def ldaTest(documents):
    global nTopic
    global nTopTopic
    global corpusKey
    global modelTest
    global modelTrain
    global ldaTestset
    model = modelTest
    if corpusKey is None:
        corpusKey = re.split('\n',open(lda_path + "lda_corpusKey.txt",'r').read())

    if ldaTestset is None:
        ldaTestset = np.zeros((len(documents),len(corpusKey)),dtype=int)

    if model is None:
        if modelTrain is not None:
            model = modelTrain
        else:
            model = lda.LDA(n_topics = nTopTopic, n_iter=1000, random_state=1)
            model.components_ = np.fromfile(lda_path + 'lda_components.dat',dtype="float64").reshape((-1,len(corpusKey)))
            model.doc_topic_ = np.fromfile(lda_path + 'lda_doc_topic.dat',dtype="float64").reshape((-1,nTopic))
        from collections import Counter
        top_topic_tuple = Counter([dt.argmax() for dt in model.doc_topic_]).most_common(nTopTopic)
        top_topic_index = [item[0] for item in top_topic_tuple]
##        np.array(top_topic_index).tofile('lda_top%d_topic_index.dat'%(nTopTopic))
        np.savetxt(feature_path + 'lda_top%d_topic_index.csv'%(nTopTopic), np.array(top_topic_index), delimiter=",")
        model.components_ = model.components_[top_topic_index,:]
        model.doc_topic_ = model.doc_topic_[:,top_topic_index]
        
    # testset(question, key)
    for i in range(len(documents)):
        if documents[i] =="":
            continue
        text = re.findall('[\w\'-]*[A-Za-z]+',re.sub('<[\w/]+>|\\n','',documents[i].lower()))
        for j in range(len(corpusKey)):
            if corpusKey[j] in text:
                ldaTestset[i,j] = ldaTestset[i,j]+1
                
    return model.transform(ldaTestset)

def stylometry(content):
    f = np.zeros((10),dtype = int)
    content = re.sub('<[\w/]+>|\\n','',content)
    words = re.findall('[\w\'-]*[A-Za-z]+',content)
    # frequency of '
    f[0]=content.count("'")
    # number of char
    f[1]=len(content)
    # Freq. of words only first letter uppercased
    f[2]=len(re.findall('\W[A-Z]\w*',content))
    # number of words
    f[3]=len(words)
    # freq. of (NP,PRP)-->bi-gram with JJ+PRP pairs: PRP = personal noun
    pos_tags = [tag[1] for tag in nltk.pos_tag(words)]
    f[4] = len([tag for tag in pos_tags if tag == 'PRP'])
    # freq. of .
    f[5]=content.count(".")
    # freq. of lowercase words
    f[6]=len(re.findall('[a-z-]+',content))
    # freq. of (NP,NNP) --> JJ+NNP: NNP = singular proper noun
    f[7] = len([tag for tag in pos_tags if tag == 'NNP'])
    # freq of uppercase words
    f[8]=len(re.findall('[A-Z-]+',content))
    # freq of ,
    f[9]=content.count(",")
    return f

# ============== feature matrix generating function below:

# tell the type of item: Q,A,C ??? 
def CreateTime(r):
    global CreateTimeCount
    global timeInterval
    global CallTimes
    f = np.zeros((nQuestion,len(timeInterval)),dtype = float)
    items = itemsByRound(r)
    timeStrs = [item.get("CreationDate") for item in items]
    for i in range(len(timeStrs)):
        if timeStrs[i] != "":
            CallTimes['CreateTime'][i] = CallTimes['CreateTime'][i] + 1
    f = timeSlot(timeStrs)
    if r == 0:
        CreateTimeCount = f
    else:
##        f = CreateTimeCount + f
        f = CreateTimeCount + f
        CreateTimeCount = f
##    f = np_row(f)
    f = f / (CallTimes['CreateTime'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'CreateTime[%d].csv'%(r), f, delimiter=",")
    return f


'''
time span from last item
'''
def TimeSpan(r):
    global LastTime
    global TimeDiff
    global CallTimes
    items = itemsByRound(r)
    f = np.zeros( (nQuestion,1), dtype=float)
    if r == 0:
        TimeDiff = f
        LastTime = np.zeros( (nQuestion,1), dtype=datetime)
        for i in range(nQuestion):
            if items[i].get("CreationDate")=='':
                continue
            CallTimes['TimeSpan'][i] = CallTimes['TimeSpan'][i] + 1
            LastTime[i,0] = datetime.strptime(items[i].get("CreationDate"), "%Y-%m-%dT%H:%M:%S.%f")
    else:
        for i in range(nQuestion):
            if items[i].get("CreationDate")=='':
                f[i,0] = TimeDiff[i,0]
                continue
            CallTimes['TimeSpan'][i] = CallTimes['TimeSpan'][i] + 1
            dt = datetime.strptime(items[i].get("CreationDate"), "%Y-%m-%dT%H:%M:%S.%f")
            f[i,0] = (dt - LastTime[i,0]).seconds + TimeDiff[i,0] #datetime diff = datetime.timedelta(d.days, d.seconds, d.microseconds)
            LastTime[i,0] = dt
            TimeDiff[i,0] = f[i,0]
##    f = np_row(f)
    f = f / (CallTimes['TimeSpan'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'TimeSpan[%d].csv'%(r), f, delimiter=",")
    return f


'''
first round, it is all zero
'''
def QA_LastActTime(r):
    global LastActTimeDiff
    global CallTimes
##    items = db.itemsByRound(r)
    items = itemsByRound(r)
    f = np.zeros( (nQuestion,1), dtype=float)
    if r ==0:
        LastActTimeDiff=f
    for i in range(nQuestion):
        lastStr = items[i].get("LastActivityDate",default='')
        createStr = items[i].get("CreationDate",default='')
        if createStr == '' or lastStr == "":
            f[i,0] = LastActTimeDiff[i,0]
            continue
        CallTimes['QA_LastActTime'][i] = CallTimes['QA_LastActTime'][i] + 1
        last = datetime.strptime(lastStr, "%Y-%m-%dT%H:%M:%S.%f")
        create = datetime.strptime(createStr, "%Y-%m-%dT%H:%M:%S.%f")
        f[i,0] = (last-create).seconds+LastActTimeDiff[i,0]
        LastActTimeDiff[i,0] = f[i,0]
##        print r,i,last,create,(last-create).seconds,LastActTimeDiff[i,0],f[i,0]
##    f = np_row(f) # to float
    f = f / (CallTimes['QA_LastActTime'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'QA_LastActTime[%d].csv'%(r), f, delimiter=",")
    return f


def TopicDis(r):
    items = db.itemsByRound(r)
    documents = []
    for i in range(len(questions)):
        string = ""
        string = string + items[i].get("Text",default="").lower()
        string = string + items[i].get("Body",default="").lower()
        string = string + items[i].get("Title",default="").lower()
        documents.append(string)
    f = ldaTest(documents)
    np.savetxt(feature_path + 'TopicDis_new[%d].csv'%(r), f, delimiter=",")
    return f

def TopicDis_v1(r):
    global CallTimes
    global nTopic
    global nTopTopic
    global corpusKey
    global modelTest
    global modelTrain
    global TopicDistribution
    model = modelTest
    if corpusKey is None:
        corpusKey = re.split('\n',open(lda_path + "lda_corpusKey.txt",'r').read())

    if model is None:
        if modelTrain is not None:
            model = modelTrain
        else:
            model = lda.LDA(n_topics = nTopTopic, n_iter=1000, random_state=1)
            model.components_ = np.fromfile(lda_path + 'lda_components.dat',dtype="float64").reshape((-1,len(corpusKey)))
            model.doc_topic_ = np.fromfile(lda_path + 'lda_doc_topic.dat',dtype="float64").reshape((-1,nTopic))
        from collections import Counter
        top_topic_tuple = Counter([dt.argmax() for dt in model.doc_topic_]).most_common(nTopTopic)
        top_topic_index = [item[0] for item in top_topic_tuple]
        np.savetxt(feature_path + 'lda_top%d_topic_index.csv'%(nTopTopic), np.array(top_topic_index), delimiter=",")
        model.components_ = model.components_[top_topic_index,:]
        model.doc_topic_ = model.doc_topic_[:,top_topic_index]
        
    items = itemsByRound(r)
    f = np.zeros((nQuestion, nTopTopic))
    if r == 0:
        TopicDistribution = f
    else:
        f = TopicDistribution
    for i in range(nQuestion):
        document = ""
        document = document + items[i].get("Text",default="").lower()
        document = document + items[i].get("Body",default="").lower()
        document = document + items[i].get("Title",default="").lower()
        if document =="":
            continue
        CallTimes['TopicDis'][i] = CallTimes['TopicDis'][i] + 1
        text = re.findall('[\w\'-]*[A-Za-z]+',re.sub('<[\w/]+>|\\n','',document.lower()))
        ldaTestset = np.zeros((len(corpusKey)),dtype=int)
        for j in range(len(corpusKey)):
            if corpusKey[j] in text:
                ldaTestset[j] = ldaTestset[j] + 1
        results = model.transform(ldaTestset)
        f[i] = f[i] + results
    TopicDistribution = f
##    f = np_row(f)
    f = f / (CallTimes['TopicDis'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'TopicDis[%d].csv'%(r), f, delimiter=",")
    return f

def StylometryDis(r):
    global StylometryAccu
    global CallTimes
    items = itemsByRound(r)
    f = np.zeros((nQuestion,10),dtype = float)
    for i in range(nQuestion):
        content = items[i].get("Text",default="") + " " + items[i].get("Body",default="") + " " + items[i].get("Title",default="")
        if len(re.sub('\s','',content)) == 0:
            continue
        CallTimes['StylometryDis'][i] = CallTimes['StylometryDis'][i] + 1
        content = re.sub('<.*?>|\\n',' ',content)
        f[i,:] = stylometry(content)
    if r == 0:
        StylometryAccu = f
    else:
        f = StylometryAccu + f
        StylometryAccu  = f
##    f = np_row(f)
    f = f / (CallTimes['StylometryDis'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'StylometryDis[%d].csv'%(r), f, delimiter=",")
    return f


def ScoreMean(r):
    global ScoreSum
    global CallTimes
##    items = db.itemsByRound(r)
    items = itemsByRound(r)
    f = np.zeros( (nQuestion,1), dtype=float)
    for i in range(len(items)):
        scoreStr = items[i].get("Score")
        if scoreStr == "":
            continue
        CallTimes['ScoreMean'][i] = CallTimes['ScoreMean'][i] + 1
        f[i,0] = float(scoreStr)
    if r == 0:
        ScoreSum = f
    else:
        f = ScoreSum + f
        ScoreSum = f
##    f = np_row(f) # to float
    f = f / (CallTimes['ScoreMean'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'ScoreMean[%d].csv'%(r), f, delimiter=",")
    return f

def Q_Tags(r):
    global TagDis
    global TagList
    global CallTimes
    if TagList is None: # only the 0th round is question and question has tags
        TagList = [tag.get("TagName") for tag in tags]
    f = np.zeros( (nQuestion,len(TagList)), dtype=float)
    if r == 0:
        TagDis = f
    else:
        f = TagDis
    items = itemsByRound(r)
    for i in range(nQuestion):
        tagStr = items[i].get("Tags", default = "")
        if tagStr != "":
            CallTimes['Q_Tags'][i] = CallTimes['Q_Tags'][i] + 1
            questionTags = [tag for tag in re.split('[<>]', tagStr) if len(tag)>0]
            for questionTag in questionTags:
                if questionTag in TagList:
                    j = TagList.index(questionTag)
##                    f[i,j] = (TagDis[i,j] + 1.0)/2# 1 or 0
                    f[i,j]  = f[i,j] + 1.0
    TagDis = f
##    f = f/(np.sum(f,axis=1).reshape(-1,1)*np.ones((1,len(TagList))))
    f = f / (CallTimes['Q_Tags'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'Q_Tags[%d].csv'%(r), f, delimiter=",")
    return f


def QA_CommentCount(r):
    global CommentCount
    global CallTimes
##    items = db.itemsByRound(r)
    items = itemsByRound(r)
    f = np.zeros( (nQuestion,1), dtype=float)
    for i in range(len(items)):
        countStr = items[i].get("CommentCount",default="")
        if countStr == "":
            continue
        CallTimes['QA_CommentCount'][i] = CallTimes['QA_CommentCount'][i] + 1
        f[i,0] = float(countStr)
    if r == 0:
        CommentCount = f
    else:
##        f = (CommentCount + f)/2
        f = CommentCount + f
        CommentCount = f
##    f = np_row(f) # to float
    f = f / (CallTimes['QA_CommentCount'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'QA_CommentCount[%d].csv'%(r), f, delimiter=",")
    return f


def Q_Tags_select(r=0):
    global TagDis
    global TagList
    global CallTimes
    tagLabels = []
    for i in range(nQuestion):
        tagStr = questions[i].get("Tags", default = "")
        if tagStr != "":
            tagLabels.extend([tag for tag in re.split('[<>]', tagStr) if len(tag)>0])
    TagList = Counter(tagLabels).keys()
    f = np.zeros( (nQuestion,len(TagList)), dtype=float)
    if r == 0:
        TagDis = f
##        items = db.itemsByRound(r)
    items = itemsByRound(r)
    for i in range(nQuestion):
        tagStr = questions[i].get("Tags", default = "")
        if tagStr != "":
            CallTimes['Q_Tags'][i] = CallTimes['Q_Tags'][i] + 1
            questionTags = [tag for tag in re.split('[<>]', tagStr) if len(tag)>0]
            for questionTag in questionTags:
                if questionTag in TagList:
                    j = TagList.index(questionTag)
##                    f[i,j] = (TagDis[i,j] + 1.0)/2# 1 or 0
                    f[i,j]  = TagDis[i,j] + 1.0
    TagDis = f
##    f = f/(np.sum(f,axis=1).reshape(-1,1)*np.ones((1,len(TagList))))
    f = f / (CallTimes['Q_Tags'] * np.ones((1,f.shape[1])))
    np.savetxt(feature_path + 'Q_Tags[%d].csv'%(r), f, delimiter=",")
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier()
    print clf.fit(f, target).score(f,target)
    feature_ind = np.argsort(clf.feature_importances_)[-350:] #np.where(clf.feature_importances_>0)[0]
    clf0 = ExtraTreesClassifier()
    f0 = f[:,feature_ind]
    print clf0.fit(f0,target).score(f0,target)
    tags= ET.parse('/Users/sindyjen/Documents/codes/python/author_identification/dataset/Tags_full.xml').getroot()
    tagList = [TagList[i] for i in feature_ind]
    tagsNew = ET.Element("tags")
    tagsNew.extend([tag for tag in tags if tag.get("TagName") in tagList]) # 
    tagsNewTree=ET.ElementTree(tagsNew)
    tagsNewTree.write("/Users/sindyjen/Documents/codes/python/author_identification/dataset_3items/Tags_three.xml")

if __name__ == "__main__":
    pass
