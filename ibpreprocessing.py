from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

fixContractions = False
isVerbose = False
np.random.seed()


def write_corpus(corpus_raw=None, fix_contractions=False, verbose=False):
    global isVerbose
    isVerbose = verbose
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    if corpus_raw is None:
        corpus = pd.read_csv("dataset_final_ib_raw.csv")
    else:
        corpus = corpus_raw
    print("Test baca corpus")
    corpus['text'] = corpus['f_pin'] + " " + corpus["title"]+" "+corpus["description"]
    print("Test baca corpus 2 ",corpus["text"])
    corpus['text'].dropna(inplace=True)
    corpus['text'] = [entry.lower() for entry in corpus['text']]
    vprint('Tokenize words...')
    corpus['text'] = [word_tokenize(entry) for entry in corpus['text']]
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    vprint('Lemmatize words...')
    for index, entry in enumerate(corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word not in stopwords.words('indonesia') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        corpus.loc[index, 'text_final'] = str(Final_words)

    corpus = corpus.loc[:, ['post_id', 'text_final']]
    corpus.to_csv('dataset_final_ib-preprocessing.csv')
    return corpus


def vprint(*data):
    if isVerbose:
        print(*data)
