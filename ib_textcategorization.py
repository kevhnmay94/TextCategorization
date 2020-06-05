#! /usr/local/bin/python3.7
import pandas as pd
import numpy as np
import scikitprocessing
import kerasprocessing
import mypreprocessing
import json
import sys
from pathlib import Path
import os
import time
import psutil
import pymysql.cursors
import urllib.parse

path = str(Path(sys.argv[0]).parent) + str(os.sep)
MNB_FILENAME = path + 'mnb_classifier.pkl'
SVM_FILENAME = path + 'svm_classifier.pkl'
MLP_FILENAME = path + 'mlp_classifier.pkl'
CORPUS_VECTOR = path + 'tfidf_vector.pkl'
trainMode = False
fitCorpus = False
fitTrainModel = False
writeCorpus = False
useScikit = True
useScikitMNB = False
useScikitSVM = False
useScikitMLP = True
useKeras = False
verbose = False
headline = None
content = None
probaResult = True
partialTrain = True
useSQL = True

def vprint(*data):
    if verbose:
        print(*data)


skipArg = False
for i,s in enumerate(sys.argv[1:]):
    if skipArg:
        skipArg = False
        continue
    if s[:2] == '--':
        arg = s[2:]
        if arg == 'train':
            trainMode = True
        elif arg == 'test':
            trainMode = False
        elif arg == 'scikit':
            useScikit = True
        elif arg == 'no-scikit':
            useScikit = False
        elif arg == 'keras':
            useKeras = True
        elif arg == 'no-keras':
            useKeras = False
        elif arg == 'proba':
            probaResult = True
        elif arg == 'no-proba':
            probaResult = False
        elif arg == 'no-partial':
            partialTrain = False
        elif arg == 'headline':
            headline = sys.argv[i+2]
            skipArg = True
        elif arg == 'content':
            content = sys.argv[i+2]
            skipArg = True
        if useScikit:
            if arg == 'mlp':
                useScikitMLP = True
            elif arg == 'no-mlp':
                useScikitMLP = False
            elif arg == 'svm':
                useScikitSVM = True
            elif arg == 'no-svm':
                useScikitSVM = False
            elif arg == 'mnb':
                useScikitMNB = True
            elif arg == 'no-mnb':
                useScikitMNB = False
        if trainMode:
            if arg == 'fit-model':
                fitTrainModel = True
            elif arg == 'no-fit-model':
                fitTrainModel = False
            elif arg == 'write-corpus':
                writeCorpus = True
            elif arg == 'no-write-corpus':
                writeCorpus = False
            elif arg == 'fit-corpus':
                fitCorpus = True
            elif arg == 'no-fit-corpus':
                fitCorpus = False
    elif s[0] == '-':
        for arg in s[1:]:
            if 't' == arg:
                trainMode = True
            elif 'T' == arg:
                trainMode = False
            elif 'C' == arg:
                useScikit = True
            elif 'c' == arg:
                useScikit = False
            elif 'K' == arg:
                useKeras = True
            elif 'k' == arg:
                useKeras = False
            elif 'v' == arg:
                verbose = True
            elif 'q' == arg:
                verbose = False
            if useScikit:
                if 'N' == arg:
                    useScikitMLP = True
                elif 'n' == arg:
                    useScikitMLP = False
                if 'S' == arg:
                    useScikitSVM = True
                elif 's' == arg:
                    useScikitSVM = False
                if 'B' == arg:
                    useScikitMNB = True
                elif 'b' == arg:
                    useScikitMNB = False
            if trainMode:
                if 'M' == arg:
                    fitTrainModel = True
                elif 'm' == arg:
                    fitTrainModel = False
                elif 'C' == arg:
                    writeCorpus = True
                elif 'c' == arg:
                    writeCorpus = False
                elif 'F' == arg:
                    fitCorpus = True
                elif 'f' == arg:
                    fitCorpus = False

if not trainMode:
    fitCorpus = False
    fitTrainModel = False
    writeCorpus = False

def unqoute(text):
    res = urllib.parse.unquote(text)
    res = urllib.parse.unquote_plus(res)
    return res
def insertCategory(post_id,category):
    category_id = 15
    with open("database.txt") as f:
        props = [line.rstrip() for line in f]

    # Connect to the database
    connection = pymysql.connect(host=props[0],
                                 user=props[1],
                                 password=props[2],
                                 db=props[3])
    try:
        with connection.cursor() as cursor:
            # Read a single record
            sql = "SELECT `ID` FROM `CATEGORY` where `CODE` = %s"
            cursor.execute(sql, (str(category),))
            category_id = cursor.fetchone()[0]
            # category_id = int(result)

        with connection.cursor() as cursor:
            vprint("Post ID: {}, Category: {}".format(post_id,category_id))
            sql = "INSERT INTO `CONTENT_CATEGORY` (`POST_ID`,`CATEGORY`) values (%s,%s)"
            cursor.execute(sql,(str(post_id),category_id))

        connection.commit()
    finally:
        connection.close()
def deleteOthers(postID):

    with open("database.txt") as f:
        props = [line.rstrip() for line in f]

    # Connect to the database
    connection = pymysql.connect(host=props[0],
                                 user=props[1],
                                 password=props[2],
                                 db=props[3])
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM `CONTENT_CATEGORY` where `POST_ID` = %s AND `CATEGORY` = 15"
            cursor.execute(sql,(str(postID)))

        connection.commit()
    finally:
        connection.close()






def fetch_unlabeled_SQL():
    with open("database.txt") as f:
        props = [line.rstrip() for line in f]

    # Connect to the database
    connection = pymysql.connect(host=props[0],
                                 user=props[1],
                                 password=props[2],
                                 db=props[3])

    query = "SELECT * FROM CATEGORY where CODE = 'Others'"
    vprint('db others')
    others = pd.read_sql(query, connection)['ID'][0]
    vprint('db others 2')
    query = "select P.`POST_ID`, P.`TITLE`, P.`DESCRIPTION` from `POST` P,CONTENT_CATEGORY C WHERE P.POST_ID = C.POST_ID AND C.CATEGORY = {}".format(others)
    vprint('db data')
    data = pd.read_sql(query,connection)
    vprint('db data 2')
    data.rename(columns={'POST_ID': 'story_id', 'TITLE': 'title', 'DESCRIPTION': 'description'},
                inplace=True)
    title = data['title'].tolist()
    desc = data['description'].tolist()
    title = [unqoute(x) for x in title]
    desc = [unqoute(x) for x in desc]
    result = []
    databaru = pd.DataFrame(columns=['post_id','title','description'])
    for val1,val2,val3 in zip(data['story_id'],data['title'],data['description']):
        databaru = databaru.append(pd.DataFrame([[val1,unqoute(val2),unqoute(val3)]],columns=['post_id','title','description']))
    for val1,val2 in zip(title,desc):
        result.append(val1 + " " + val2)
    connection.close()
    vprint("Result: {}".format(result))
    vprint("Data: {}".format(data.to_string))
    return result,databaru
    # except IndexError as e:
    #     vprint(e)
    #     return None
    # except:
    #     return None

totalstart = time.time()
np.random.seed()
if writeCorpus:
    corpus = None
else:
    try:
        corpus = pd.read_csv(path + "dataset_final.csv")
    except FileNotFoundError:
        corpus = None
if corpus is None or writeCorpus:
    writeCorpus = True
    fitTrainModel = True
    corpus = mypreprocessing.write_corpus(path, fix_contractions=False)
if (not useScikit) and (not useKeras):
    useScikit = True
if (not useScikitMNB) and (not useScikitSVM) and (not useScikitMLP):
    useScikitMLP = True
if useScikit:
    test_str = None
    vprint("Headline: ",headline)
    vprint("Content: ",content)
    if headline is not None and content is not None:
        test_str = headline + " " + content
    elif useSQL:
        test_str,unc = fetch_unlabeled_SQL()
        if not test_str:
            exit()
    vprint("Test: ",test_str)
    start = time.time()
    scikitprocessing.prepare(corpus, path, write_corpus=writeCorpus, fit_corpus=fitCorpus,
                             fit_train_model=fitTrainModel, partial=partialTrain,
                             proba=probaResult, verbose=verbose, new_data=test_str)
    end = time.time()
    elapse = end - start
    vprint("Prepare time: {}".format(elapse))
    if useScikitMNB:
        result = scikitprocessing.test_mnb()
    if useScikitSVM:
        result = scikitprocessing.test_svm()
    if useScikitMLP:
        start = time.time()
        result = scikitprocessing.test_mlp()
        end = time.time()
        elapse = end - start
        vprint("TEST MLP TIME: {}".format(elapse))

    if probaResult:
        print(result[0])
        print(result[2])
        if headline is not None and content is not None:
            dataset = pd.DataFrame(data={'category': result[1], 'headline': [headline], 'content': [content]})
            dataset.to_csv(path + 'dataset-all.csv',mode='a',header=False,index=False)
        elif useSQL:
            for x,y,z in zip(result[1],unc['title'].tolist(),unc['description'].tolist()):
                dataset = pd.DataFrame(data={'category': x, 'headline': y, 'content': z}, index=[0])
                dataset.to_csv(path + 'dataset-all.csv', mode='a', header=False, index=False)
            for a,b in zip(result[0],unc['post_id'].tolist()):
                for category in a:
                    insertCategory(b,category)
                deleteOthers(b)




    else:
        print(result)
        if headline is not None and content is not None:
            dataset = pd.DataFrame(data={'category': [result], 'headline': [headline], 'content': [content]})
            dataset.to_csv(path + 'dataset-all.csv',mode='a',header=False,index=False)
    totalend = time.time()
    totalelapse = totalend - totalstart
    vprint("Total python time: {}".format(totalelapse))



if useKeras:
    kerasprocessing.exec(corpus, path, write_corpus=writeCorpus, fit_corpus=fitCorpus, fit_train_model=fitTrainModel,
                         verbose=verbose, new_data=test_str)
