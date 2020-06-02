#! python
import pymysql.cursors
import pandas as pd
import numpy as np
import scikitprocessing
import kerasprocessing
import cupreprocessing
import sys
from pathlib import Path
import os

path = str(Path(sys.argv[0]).parent) + str(os.sep)
MNB_FILENAME = path + 'mnb_classifier.pkl'
SVM_FILENAME = path + 'svm_classifier.pkl'
MLP_FILENAME = path + 'mlp_classifier.pkl'
CORPUS_VECTOR = path + 'tfidf_vector.pkl'
trainMode = True
fitCorpus = False
fitTrainModel = False
writeCorpus = False
useScikit = True
useScikitMNB = False
useScikitSVM = False
useScikitMLP = True
useKeras = False
verbose = True
headline = None
content = None
f_pin = None
probaResult = True
partialTrain = True
fetchSQL = True


def vprint(*data):
    if verbose:
        print(*data)


for i,s in enumerate(sys.argv[1:]):
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
        elif arg == 'content':
            content = sys.argv[i+2]
        elif arg == 'f_pin':
            f_pin = sys.argv[i+2]
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

np.random.seed()

def fSQL():
    try:
        with open("database.txt") as f:
            props = [line.rstrip() for line in f]

        # Connect to the database
        connection = pymysql.connect(host=props[0],
                                     user=props[1],
                                     password=props[2],
                                     db=props[3])

        query = props[4]
        data = pd.read_sql(query, connection)
        data.rename(columns={'POST_ID':'story_id','F_PIN':'f_pin','TITLE':'title','DESCRIPTION':'description'},inplace=True)
        connection.close()
        return data
    except IndexError as e:
        vprint(e)
        return None
    except:
        return None


if writeCorpus:
    corpus = None
elif fetchSQL:
    corpus = fSQL()
else:
    try:
        corpus = pd.read_csv(path+"dataset_final_cu_raw.csv")
    except FileNotFoundError:
        corpus = None
if corpus is None or writeCorpus:
    writeCorpus = True
    fitTrainModel = True
    corpus = cupreprocessing.write_corpus(path, fix_contractions=False)

if useScikit:
    test_str = None
    vprint("Title: ",headline)
    vprint("Description: ",content)
    vprint("F_Pin ID: ",f_pin)
    if f_pin is not None and headline is not None and content is not None:
        test_str = f_pin+" "+headline + " " + content
    vprint("Testing Result: ",test_str)
    scikitprocessing.prepare(corpus, path, write_corpus=writeCorpus, fit_corpus=fitCorpus,
                             fit_train_model=fitTrainModel, partial=True,
                             proba=probaResult, verbose=verbose, new_data=test_str)
    if useScikitMNB:
        result = scikitprocessing.test_mnb()
    if useScikitSVM:
        result = scikitprocessing.test_svm()
    if useScikitMLP:
        result = scikitprocessing.test_mlp()
    if isinstance(result, list):
        print(result[0])
        if f_pin is not None and headline is not None and content is not None:
            dataset = pd.DataFrame(data={'story_id': result[1],'f_pin':[f_pin], 'title': [headline], 'description': [content]})
            dataset.to_csv(path + 'dataset-all-cu.csv',mode='a',header=False,index=False)
    else:
        print(result)
        if f_pin is not None and headline is not None and content is not None:
            dataset = pd.DataFrame(data={'story_id': [result],'f_pin':[f_pin], 'title': [headline], 'description': [content]})
            dataset.to_csv(path + 'dataset-all-cu.csv',mode='a',header=False,index=False)


if useKeras:
    kerasprocessing.exec(corpus, path, write_corpus=writeCorpus, fit_corpus=fitCorpus, fit_train_model=fitTrainModel,
                         verbose=verbose, new_data=test_str)
