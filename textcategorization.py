import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes as nb_generator, svm as svm_generator, neighbors, tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import contractions

MNB_FILENAME = 'mnb_classifier.pkl'
SVM_FILENAME = 'svm_classifier.pkl'
KNN_FILENAME = 'knn_classifier.pkl'
RNC_FILENAME = 'rnc_classifier.pkl'
DT_FILENAME = 'dt_classifier.pkl'
fixContractions = False

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
np.random.seed(420)
try:
    corpus = pd.read_csv("dataset_final.csv")
except FileNotFoundError:
    corpus = None
if corpus is None:
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
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'], corpus['category'],
                                                                    test_size=0.3)
print('Encoding labels...')
Encoder = LabelEncoder()
Train_Y_Encoded = Encoder.fit_transform(Train_Y)
Test_Y_Encoded = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
unlabeled = Tfidf_vect.transform(
    ["Valencia played host to the final races of a truncated MotoE World Cup this month, "
     "with two battles worthy of the drama that has characterized the series' debut "
     "season both on and off the track. MotoE's literal rise from the ashes was the talk "
     "of motor racing in July, when riders finally assembled on the grid at Germany's "
     "Sachsenring for the debut race."])
# fit the training dataset on the NB classifier
print('Classify dataset using Naive Bayes...')
try:
    mnb_model = joblib.load(MNB_FILENAME)
except FileNotFoundError:
    mnb_model = None
if mnb_model is None:
    mnb_model = nb_generator.MultinomialNB()
mnb_model.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset
predictions_NB = mnb_model.predict(Test_X_Tfidf)  # Use accuracy_score function to get the accuracy
# print(predictions_NB)
# print(Test_Y)
accuracy = accuracy_score(predictions_NB, Test_Y) * 100
print("Naive Bayes Accuracy Score -> ", f'{accuracy:.2f}%')
c = mnb_model.predict(unlabeled)
d = Encoder.transform(c)
print(c)
print(d)
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
svm_model.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset
predictions_SVM = svm_model.predict(Test_X_Tfidf)  # Use accuracy_score function to get the accuracy
# print(predictions_SVM)
# print(Test_Y)
accuracy2 = accuracy_score(predictions_SVM, Test_Y) * 100
print("SVM Accuracy Score -> ", f'{accuracy2:.2f}%')
c = svm_model.predict(unlabeled)
d = Encoder.transform(c)
print(c)
print(d)
joblib.dump(svm_model, SVM_FILENAME, compress=9)

Train_Y_Multi = []
Test_Y_Multi = []
for category in Train_Y_Encoded:
    Train_Y_Multi.append([category])
for category in Test_Y_Encoded:
    Test_Y_Multi.append([category])
mlb = MultiLabelBinarizer()
Train_Y_Set = mlb.fit_transform(Train_Y_Multi)
Test_Y_Set = mlb.fit_transform(Test_Y_Multi)

print('Classify dataset using KNN...')
try:
    knn_model = joblib.load(KNN_FILENAME)
except FileNotFoundError:
    knn_model = None
if knn_model is None:
    knn_model = neighbors.KNeighborsClassifier()
knn_model.fit(Train_X_Tfidf, Train_Y_Set)
predictions_KNN = knn_model.predict(Test_X_Tfidf)
# for a in predictions_KNN:
#     print(a)
# print(Test_Y_Set)
c = knn_model.predict(unlabeled)
print(c)
# d = mlb.inverse_transform(c)
# print(d)
# f = []
# for e in d:
#     try:
#         f.append(e[0])
#     except IndexError:
#         pass
# g = Encoder.inverse_transform(f)
# print(g)
accuracy3 = accuracy_score(predictions_KNN, Test_Y_Set) * 100
print("KNN Accuracy Score -> ", f'{accuracy3:.2f}%')
joblib.dump(knn_model, KNN_FILENAME, compress=9)

print('Classify dataset using RNC...')
try:
    rnc_model = joblib.load(RNC_FILENAME)
except FileNotFoundError:
    rnc_model = None
if rnc_model is None:
    rnc_model = neighbors.RadiusNeighborsClassifier()
rnc_model.fit(Train_X_Tfidf, Train_Y_Set)
predictions_RNC = knn_model.predict(Test_X_Tfidf)
# for a in predictions_RNC:
#     print(a)
# print(Test_Y_Set)
c = rnc_model.predict(unlabeled)
print(c)
# d = mlb.inverse_transform(c)
# print(d)
# f = []
# for e in d:
#     try:
#         f.append(e[0])
#     except IndexError:
#         pass
# g = Encoder.inverse_transform(f)
# print(g)
accuracy4 = accuracy_score(predictions_RNC, Test_Y_Set) * 100
print("RNC Accuracy Score -> ", f'{accuracy4:.2f}%')
joblib.dump(rnc_model, RNC_FILENAME, compress=9)

print('Classify dataset using Decision Tree...')
try:
    dt_model = joblib.load(DT_FILENAME)
except FileNotFoundError:
    dt_model = None
if dt_model is None:
    dt_model = tree.DecisionTreeClassifier(class_weight="balanced")
dt_model.fit(Train_X_Tfidf, Train_Y_Set)
predictions_DT = dt_model.predict(Test_X_Tfidf)
c = rnc_model.predict(unlabeled)
print(c)
accuracy5 = accuracy_score(predictions_DT, Test_Y_Set) * 100
print("DT Accuracy Score -> ", f'{accuracy5:.2f}%')
joblib.dump(dt_model, DT_FILENAME, compress=9)
