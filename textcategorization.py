#! python
import pandas as pd
import numpy as np
import scikitprocessing
import kerasprocessing
import mypreprocessing
import json
import sys

MNB_FILENAME = 'mnb_classifier.pkl'
SVM_FILENAME = 'svm_classifier.pkl'
MLP_FILENAME = 'mlp_classifier.pkl'
CORPUS_VECTOR = 'tfidf_vector.pkl'
trainMode = True
fitCorpus = trainMode
fitTrainModel = trainMode
writeCorpus = trainMode
useScikit = True
useScikitMNB = True
useScikitSVM = True
useScikitMLP = True
useKeras = False
verbose = False


def vprint(data):
    if verbose:
        print(data)


for s in sys.argv:
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
        else:
            fitTrainModel = False
            writeCorpus = False
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
            else:
                fitCorpus = False
                fitTrainModel = False
                writeCorpus = False

np.random.seed()
if writeCorpus:
    corpus = None
else:
    try:
        corpus = pd.read_csv("dataset_final.csv")
    except FileNotFoundError:
        corpus = None
if corpus is None or writeCorpus:
    writeCorpus = True
    fitTrainModel = True
    corpus = mypreprocessing.write_corpus(fix_contractions=False)

if useScikit:
    scikitprocessing.prepare(corpus, write_corpus=writeCorpus, fit_corpus=fitCorpus, fit_train_model=fitTrainModel,
                             proba=True, verbose=verbose)
    if useScikitMNB:
        result = scikitprocessing.test_mnb()
    if useScikitSVM:
        result = scikitprocessing.test_svm()
    if useScikitMLP:
        result = scikitprocessing.test_mlp()
    if isinstance(result, list):
        dump = json.dumps(result)
        print(dump)
    else:
        print(result)

if useKeras:
    kerasprocessing.exec(corpus, write_corpus=writeCorpus, fit_corpus=fitCorpus, fit_train_model=fitTrainModel,
                         verbose=verbose)
