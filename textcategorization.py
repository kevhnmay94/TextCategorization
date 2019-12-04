import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import contractions
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes as nb_generator, svm as svm_generator
from sklearn.neural_network import multilayer_perceptron
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

MNB_FILENAME = 'mnb_classifier.pkl'
SVM_FILENAME = 'svm_classifier.pkl'
KNN_FILENAME = 'knn_classifier.pkl'
RNC_FILENAME = 'rnc_classifier.pkl'
DT_FILENAME = 'dt_classifier.pkl'
MLP_FILENAME = 'mlp_classifier.pkl'
CORPUS_VECTOR = 'tfidf_vector.pkl'
fixContractions = False
multilabel = False
fitCorpus = False
fitTrainModel = False
writeCorpus = False

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
np.random.seed()
try:
    corpus = pd.read_csv("dataset_final.csv")
except FileNotFoundError:
    corpus = None
if corpus is None or writeCorpus:
    writeCorpus = True
    fitTrainModel = True
    corpus = pd.read_csv("dataset-all.csv")
    corpus['text'] = corpus['headline'] + " " + corpus["content"]
    corpus['text'].dropna(inplace=True)
    corpus['text'] = [entry.lower() for entry in corpus['text']]
    if fixContractions:
        corpus['text'] = [contractions.fix(entry) for entry in corpus['text']]
    print('Tokenize words...')
    corpus['text'] = [word_tokenize(entry) for entry in corpus['text']]
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    print('Lemmatize words...')
    for index, entry in enumerate(corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        corpus.loc[index, 'text_final'] = str(Final_words)

    corpus = corpus.loc[:, ['category', 'text_final']]
    corpus.to_csv('dataset_final.csv')

print('Preparing train and test data sets...')
test_size = 0.25 if fitTrainModel else corpus['text_final'].size-1
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'], corpus['category'],
                                                                    test_size=test_size)
print('Encoding labels...')
Encoder = LabelEncoder()
Train_Y_Encoded = Encoder.fit_transform(Train_Y)
Test_Y_Encoded = Encoder.fit_transform(Test_Y)
if writeCorpus is True:
    Tfidf_vect = None
else:
    try:
        Tfidf_vect = joblib.load(CORPUS_VECTOR)
    except FileNotFoundError:
        Tfidf_vect = None
if Tfidf_vect is None:
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    fitCorpus = True
if fitCorpus:
    Tfidf_vect.fit(corpus['text_final'])
    joblib.dump(Tfidf_vect, CORPUS_VECTOR, compress=9)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
print(Test_X)
unlabeled = Tfidf_vect.transform(
    ["The first 2020 issue of Hakusensha's Hana to Yume magazine revealed on Thursday that Julietta Suzuki will draw "
     "a one-shot manga in the magazine's third issue on January 4. The magazine describes the one-shot manga as a "
     "avant-garde mystery, and it will have a color opening page. Suzuki is also drawing a new spinoff chapter from "
     "her Kamisama Kiss (Kamisama Hajimemashita) manga in Hakusensha's The Hana to Yume magazine (a sister magazine "
     "to the main Hana to Yume magazine) on January 25. Suzuki launched the Kamisama Kiss manga in Hana to Yume in "
     "2008, and ended the series in May 2016 with 25 volumes. Viz Media has published the manga in English. "])
print('Classify dataset using Naive Bayes...')
try:
    mnb_model = joblib.load(MNB_FILENAME)
except FileNotFoundError:
    mnb_model = None
if mnb_model is None:
    mnb_model = nb_generator.MultinomialNB()
    fitTrainModel = True
if fitTrainModel:
    mnb_model.fit(Train_X_Tfidf, Train_Y)
predictions_NB = mnb_model.predict(Test_X_Tfidf)
# print(predictions_NB)
# print(Test_Y)
accuracy = metrics.accuracy_score(y_pred=predictions_NB, y_true=Test_Y) * 100
precision = metrics.precision_score(y_pred=predictions_NB, y_true=Test_Y, average='macro') * 100
recall = metrics.recall_score(y_pred=predictions_NB, y_true=Test_Y, average='macro') * 100
print("Naive Bayes Accuracy Score -> ", f'{accuracy:.2f}%')
print("Recall -> ", f'{recall:.2f}%')
print("Precision -> ", f'{precision:.2f}%')
c = mnb_model.predict(unlabeled)
d = Encoder.transform(c)
print(c)
print(d)
if fitTrainModel:
    joblib.dump(mnb_model, MNB_FILENAME, compress=9)
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier

print('Classify dataset using SVM...')
try:
    svm_model = joblib.load(SVM_FILENAME)
except FileNotFoundError:
    svm_model = None
if svm_model is None:
    svm_model = svm_generator.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    fitTrainModel = True
if fitTrainModel:
    svm_model.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = svm_model.predict(Test_X_Tfidf)
# print(predictions_SVM)
# print(Test_Y)
accuracy2 = metrics.accuracy_score(y_pred=predictions_SVM, y_true=Test_Y) * 100
precision2 = metrics.precision_score(y_pred=predictions_SVM, y_true=Test_Y, average='macro') * 100
recall2 = metrics.recall_score(y_pred=predictions_SVM, y_true=Test_Y, average='macro') * 100
print("SVM Accuracy Score -> ", f'{accuracy2:.2f}%')
print("Recall -> ", f'{recall2:.2f}%')
print("Precision -> ", f'{precision2:.2f}%')
c = svm_model.predict(unlabeled)
d = Encoder.transform(c)
print(c)
print(d)
if fitTrainModel:
    joblib.dump(svm_model, SVM_FILENAME, compress=9)

print('Classify dataset using MLP...')
try:
    mlp_model = joblib.load(MLP_FILENAME)
except FileNotFoundError:
    mlp_model = None
if mlp_model is None:
    mlp_model = multilayer_perceptron.MLPClassifier()
    fitTrainModel = True
mlp_model.set_params(max_iter=400, solver='adam')
if fitTrainModel:
    mlp_model.fit(Train_X_Tfidf, Train_Y)
predictions_MLP = mlp_model.predict(Test_X_Tfidf)
accuracy3 = metrics.accuracy_score(y_pred=predictions_MLP, y_true=Test_Y) * 100
precision3 = metrics.precision_score(y_pred=predictions_MLP, y_true=Test_Y, average='macro') * 100
recall3 = metrics.recall_score(y_pred=predictions_MLP, y_true=Test_Y, average='macro') * 100
print("MLP Accuracy Score -> ", f'{accuracy3:.2f}%')
print("Recall -> ", f'{recall3:.2f}%')
print("Precision -> ", f'{precision3:.2f}%')
c = mlp_model.predict(unlabeled)
d = Encoder.transform(c)
print(c)
print(d)
if fitTrainModel:
    joblib.dump(mlp_model, MLP_FILENAME, compress=9)

if multilabel:
    Train_Y_Multi = []
    Test_Y_Multi = []
    for category in Train_Y_Encoded:
        Train_Y_Multi.append([category])
    for category in Test_Y_Encoded:
        Test_Y_Multi.append([category])
    mlb = MultiLabelBinarizer()
    Train_Y_Set = mlb.fit_transform(Train_Y_Multi)
    Test_Y_Set = mlb.fit_transform(Test_Y_Multi)
