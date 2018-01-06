# -*- coding: utf-8 -*-
"""
this is a file to manage dataset stored in xml files,
execfile("db.py")
from db import questions
from db import feature_path, clf_path, equiv_path, plot_path, lda_path
"""
import xml.etree.ElementTree as ET
import re
from datetime import datetime
import time
import numpy as np

root_path = "/Users/sindyjen/Documents/codes/python/author_identification/"
lda_path = "/Users/sindyjen/Documents/codes/python/author_identification/lda/"
##communities = ["3dprinting", "ai", "beer", "bricks", "coffee", "computergraphics", "crafts", "ebooks", "esperanto"]

dataset_path = "serverfault.com/"
feature_path = '/Users/sindyjen/Documents/codes/python/author_identification/v8/feature/'
clf_path = '/Users/sindyjen/Documents/codes/python/author_identification/v8/clf/'
equiv_path = '/Users/sindyjen/Documents/codes/python/author_identification/v8/equiv/'
plot_path = '/Users/sindyjen/Documents/codes/python/author_identification/v8/plot/'

##askers = ET.parse(root_path + dataset_path+'Users.xml').getroot() 
##questions = ET.parse(root_path + dataset_path+'Questions.xml').getroot() 
##answers = ET.parse(root_path + dataset_path+'Answers.xml').getroot()
##comments = ET.parse(root_path + dataset_path+'Comments.xml').getroot()
##tags = ET.parse(root_path + dataset_path+'Tags.xml').getroot() 
itemLists = None


# query all comments (to the question or the answers) under questions by questions' authors
def queryItemListsByQuestions():
    global itemLists
    if itemLists is not None:
        return itemLists
    itemLists = []
    for question in questions:
        uId = question.get("OwnerUserId")
        qCommunity = question.get("Community")
        qId = question.get("Id")
        pIds = [qId]
        pIds.extend([answer.get("Id") for answer in answers if answer.get("ParentId") == qId and answer.get("Community") == qCommunity])
        itemList = [question] # question
        itemList.extend([answer for answer in answers if answer.get("ParentId") == qId and answer.get("OwnerUserId") == uId and answer.get("Community") == qCommunity]) # answers
        itemList.extend([comment for comment in comments if comment.get("PostId") in pIds and comment.get("UserId") == uId and comment.get("Community") == qCommunity]) # comments
        itemList = sorted(itemList, key = lambda content: datetime.strptime( content.get("CreationDate"), "%Y-%m-%dT%H:%M:%S.%f"))
        itemLists.append(itemList)
    return itemLists

# question + n comments ordered by creationTime
def itemsByRound(r):
    itemLists = queryItemListsByQuestions()
    lis = []
    for itemList in itemLists:
        if r < len(itemList):
            lis.append(itemList[r])
        else:
            item = ET.fromstring("""<row Id="-1" PostId="" Score="" Text="" CreationDate="" UserId="" />""")
            lis.append(item)
    return lis


def nRound():
    itemList = queryItemListsByQuestions()
    maxLen = 0
    for itemList in itemLists:
        if maxLen < len(itemList):
            maxLen = len(itemList)
    return maxLen


    
if __name__ == "__main__":
##    users = ET.parse(root_path + dataset_path + 'Users.xml').getroot()
##    posts = ET.parse(root_path + dataset_path + 'Posts.xml').getroot()
##    questions = [post for post in posts if post.get("PostTypeId")=="1" and post.get("OwnerUserId") != None]
##    answers = [post for post in posts if post.get("PostTypeId")=="2" and post.get("OwnerUserId") != None]
##    comments = ET.parse(root_path + dataset_path + 'Comments.xml').getroot()
##    tags = ET.parse(root_path + dataset_path + 'Tags.xml').getroot()
##    print len(users),len(questions),len(answers),len(comments),len(tags) #315531 233964 401631 813288 3536

##    userid_questions = [question.get("OwnerUserId") for question in questions]
##    from collections import Counter
##    userid_questions = Counter(userid_questions)
##    userids = [item[0] for item in userid_questions.items() if item[1] >=5]

    '''
    usersN = ET.Element("users")
    usersN.extend([user for user in users if user.get("Id") in userids])
    print len(usersN)#9628

    questionsN = ET.Element("questions")
    questionsN.extend([question for question in questions if question.get("OwnerUserId") in userids])
    questionid_items = [question.get("Id") for question in questionsN]
    questionids = questionid_items
    print len(questionids)#108740

    answersN = ET.Element("answers")
    answersN.extend([answer for answer in answers if answer.get("ParentId") in questionids])
    questionid_items.extend([answer.get("ParentId") for answer in answers])
    answeridDic = {answer.get("Id"):answer.get("ParentId") for answer in answers}
    answerids = answeridDic.keys()
    print len(answerids)#401631

    postids = questionids
    postids.extend(answerids)
    print len(postids)#912002

    commentsN = ET.Element("comments")
    commentsN.extend([comment for comment in comments if comment.get("PostId") in postids])
    questionid_items.extend([comment.get("PostId") for comment in commentsN if comment.get("PostId") in questionids])
    questionid_items.extend([answeridDic[comment.get("PostId")] for comment in commentsN if comment.get("PostId") in answerids])
    print  len(questionid_items)
    questionid_items = Counter(questionid_items)
    print len(questionid_items)
    questionids = [item[0] for item in questionid_items.items() if item[1] >=5]
    print len(questionids)
    '''

    questionDic = {}
    for question in questionsN:
        if question.get("OwnerUserId") in userids:
            answerstmp = answers.findall('row[@ParentId="%s"][@OwnerUserId="%s"]'%(question.get("Id"),question.get("OwnerUserId")))
            questionDic[question] = 1 + len(answerstmp) + len(comments.findall('row[@PostId="%s"][@UserId="%s"]'%(question.get("Id"),question.get("OwnerUserId"))))
            for answer in answerstmp:
                questionDic[question] += len(comments.findall('row[@PostId="%s"][@UserId="%s"]'%(answer.get("Id"),answer.get("OwnerUserId"))))
    userids = [item[0].get("OwnerUserId") for item in questionDic.items() if item[1]>=5]
    print len(userids)
    usersN = ET.Element("users")
    usersN.extend([user for user in users if user.get("Id") in userids])
    questionsN = ET.Element("questions")
    questionsN.extend([question for question in questionDic.keys() if question.get("OwnerUserId") in userids])
    print len(questionsN)
    postkeys = [(question.get("Id"),question.get("OwnerUserId")) for question in questionsN]
    answersN = ET.Element("answers")
    answersN.extend([answer for answer in answers if (answer.get("ParentId"),answer.get("OwnerUserId")) in postkeys])
    print len(answersN)
    postkeys.extend([(answer.get("Id"),answer.get("OwnerUserId")) for answer in answersN])
    commentsN = ET.Element("comments")
    commentsN.extend([comment for comment in comments if (comment.get("PostId"),comment.get("UserId")) in postkeys])
    print len(commentsN)
    

    # ()
##    userid_questions = Counter([question.get("OwnerUserId") for question in questionsN if question.get("Id") in questionids])
##    print len(userid_questions)
##    userids = [item[0] for item in userid_questions.items() if item[1] >=5]
##    usersNew = ET.Element("users")
##    usersNew.extend([user for user in usersN if user.get("Id") in userids])
##    len(usersNew)
##    questionsNew = ET.Element("questions")
##    questionsNew.extend([question for question in questionsN if question.get("OwnerUserId") in userids])
##    len(questionsNew)
##    postkeys = [(question.get("Id"),question.get("OwnerUserId")) for question in questionsNew]
##    answersNew = ET.Element("answers")
##    

    
        
##    import os
##    os.system('say -v "Anna" "Wunderbar"')
