import pandas as pd
import numpy as np
import scikitprocessing
import kerasprocessing
import mypreprocessing
import json

MNB_FILENAME = 'mnb_classifier.pkl'
SVM_FILENAME = 'svm_classifier.pkl'
KNN_FILENAME = 'knn_classifier.pkl'
RNC_FILENAME = 'rnc_classifier.pkl'
DT_FILENAME = 'dt_classifier.pkl'
MLP_FILENAME = 'mlp_classifier.pkl'
CORPUS_VECTOR = 'tfidf_vector.pkl'
multilabel = False
fitCorpus = False
fitTrainModel = False
writeCorpus = False
useScikit = False
useScikitMNB = True
useScikitSVM = True
useScikitMLP = True
useKeras = True

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
    scikitprocessing.prepare(corpus, write_corpus=False, fit_corpus=False, fit_train_model=False, proba=True)
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
    kerasprocessing.exec(corpus, write_corpus=False, fit_corpus=False, fit_train_model=False)
