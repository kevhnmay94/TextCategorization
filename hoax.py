#! /home/easysoft/anaconda3/envs/hoax/bin/python

import sys
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import gensim
import joblib

overwrite = False
isVerbose = False

data = '''
tenaga medis sebarkan corona dengan suntik
'''

def vprint(*data):
    if isVerbose:
        print(*data)

for i, s in enumerate(sys.argv[1:]):
    if s[:2] == '--':
        arg = s[2:]
        if arg == 'text':
            data = sys.argv[i + 2]


    elif s[0] == '-':
        for arg in s[1:]:
            if 'v' == arg:
                verbose = True
            elif 'q' == arg:
                verbose = False
            elif 'o' == arg:
                overwrite = False
            elif 'O' == arg:
                overwrite = False

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')



def load_mdl(model_name):
    hoax_data = None
    if not overwrite:
        try:
            hoax_data = joblib.load(model_name)
        except FileNotFoundError:
            pass
    return hoax_data

def main(query,model=None):
    if 'hoax' in query.lower():
        return '{}'.format({'score': 0.00})
    if model is None:
        model = 8*[None]
        corpus = pd.read_csv("hoax.csv",sep='\t')
        title = corpus['title'].tolist()
        model[0] = title
        explanation = corpus['explanation'].tolist()
        model[4] = explanation
        link = corpus['link'].tolist()
        model[5] = link
        model[7] = corpus['statement'].tolist()

        gen_docs = [[w.lower() for w in word_tokenize(text) if w not in stopwords.words('indonesian') and w.isalnum()] for text in title]
        # print(gen_docs)
        dictionary = gensim.corpora.Dictionary(gen_docs)
        model[1] = dictionary
        bow = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
        model[6] = bow
        tf_idf = gensim.models.TfidfModel(bow)
        model[2] = tf_idf
        sims = gensim.similarities.Similarity('workdir/',tf_idf[bow],
                                                num_features=len(dictionary))
        model[3] = sims
        joblib.dump(model,"hoax_data.pkl",compress=9)

    query = [w.lower() for w in word_tokenize(query) if w not in stopwords.words('indonesian') and w.isalnum()]
    query_bow = model[1].doc2bow(query)

    query_tf_idf = model[2][query_bow]
    result = model[3][query_tf_idf]
    arr = np.array(result)
    sortd = arr.argsort()[::-1][:3]
    article = [model[0][i] for i in sortd]
    explanation = [model[4][i] for i in sortd]
    link = [model[5][i] for i in sortd]
    score = [result[i] for i in sortd]
    truth = [model[7][i] for i in sortd]
    vprint(article)
    vprint(score)
    if(score[0] >= 0.5) and truth[0] is False:
        vprint("hoax ini")
        doc = model[2][model[6]][sortd[0]]
        vprint([[model[1][id], np.around(freq, decimals=2)] for id, freq in doc])
        vprint(article[0])
        vprint(explanation[0])
        return '{}'.format({'score': score[0], 'article': article[0], 'explanation': explanation[0], 'link': link[0] })
    else:
        vprint("aman bos")
        vprint(article[0])
        if truth[0]:
            s = 1 - score[0]
        else:
            s = score[0]
        return '{}'.format({'score': s})


if __name__ == '__main__':
    model = load_mdl("hoax_data.pkl")
    print(main(data,model))