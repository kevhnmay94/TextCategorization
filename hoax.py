import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import gensim
import joblib

overwrite = False
isVerbose = False

def vprint(*data):
    if isVerbose:
        print(*data)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

data = "Penjaga Perlintasan Kereta Dibacok, Polisi: Pelaku Masih Kita Kejar "
# print(word_tokenize(data))
hoax_data = None
if not overwrite:
    try:
        hoax_data = joblib.load("hoax_data.pkl")
    except FileNotFoundError:
        hoax_data = None
if hoax_data is None:
    hoax_data = 5*[None]
    corpus = pd.read_csv("hoax.csv",sep='\t')
    hoaxes = corpus['hoax'].tolist()
    hoax_data[0] = hoaxes
    explanation = corpus['explanation'].tolist()
    hoax_data[4] = explanation

    gen_docs = [[w.lower() for w in word_tokenize(text) if w not in stopwords.words('indonesian') and w.isalnum()] for text in hoaxes]
    # print(gen_docs)
    dictionary = gensim.corpora.Dictionary(gen_docs)
    hoax_data[1] = dictionary
    bow = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(bow)
    hoax_data[2] = tf_idf
    sims = gensim.similarities.Similarity('workdir/',tf_idf[bow],
                                            num_features=len(dictionary))
    hoax_data[3] = sims
    joblib.dump(hoax_data,"hoax_data.pkl",compress=9)

query = [w.lower() for w in word_tokenize(data) if w not in stopwords.words('indonesian') and w.isalnum()]
query_bow = hoax_data[1].doc2bow(query)

query_tf_idf = hoax_data[2][query_bow]
result = hoax_data[3][query_tf_idf]
arr = np.array(result)
sortd = arr.argsort()[::-1][:3]
article = [hoax_data[0][i] for i in sortd]
explanation = [hoax_data[4][i] for i in sortd]
score = [result[i] for i in sortd]
vprint(article)
vprint(score)
if(score[0] >= 0.5):
    vprint("hoax ini")
    vprint(article[0])
    vprint(explanation[0])
    print('{}'.format({'score': score[0], 'article': article[0], 'explanation': explanation[0] }))
else:
    vprint("aman bos")
    print('{}'.format({'score': score[0]}))