import pandas as pd
from nltk.corpus import stopwords
import re
import pymysql.cursors
import urllib.parse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import warnings
import os
warnings.simplefilter("ignore", DeprecationWarning)
sns.set_style('whitegrid')
from sklearn.decomposition import LatentDirichletAllocation as LDA
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
# nltk.download('stopwords')
def unqoute(text):
    res = urllib.parse.unquote(text)
    res = urllib.parse.unquote_plus(res)
    res = res.replace("\n","")
    return res

def getStory():
    with open("database_ib_test.txt") as f:
        props = [line.rstrip() for line in f]

    # Connect to the database
    connection = pymysql.connect(host=props[0],
                                 user=props[1],
                                 password=props[2],
                                 db=props[3])
    try:
        sql = "SELECT `POST_ID`,`F_PIN`, `TITLE`, `DESCRIPTION` FROM `POST`"
        data = pd.read_sql(sql, connection)
        data.rename(columns={'POST_ID': 'post_id', 'F_PIN':'f_pin', 'TITLE': 'title', 'DESCRIPTION': 'description'},
                inplace=True)
        databaru = pd.DataFrame(columns=['post_id','f_pin', 'title', 'description'])
        for val1, val2, val3, val4 in zip(data['post_id'], data['title'], data['description'],data['f_pin']):
            databaru = databaru.append(pd.DataFrame([[val1,val4,unqoute(val2), unqoute(val3)]], columns=['post_id', 'f_pin', 'title', 'description']))
        databaru_preprocess = pd.DataFrame(columns=['post_id','text'])
        for val1, val2, val3 in zip(databaru['post_id'], databaru['title'], databaru['description']):
            textgabung = "{} {}".format(val2,val3)
            databaru_preprocess = databaru_preprocess.append(pd.DataFrame([[val1, textgabung]],columns=['post_id', 'text']))
        if len(databaru) > 0 and len (databaru_preprocess) > 0:
            print("Success Getting Data")
        databaru_preprocess.to_csv('dataset_final_ib_pre.csv', columns=['post_id', 'text'],index=False)
    except Exception as e:
        print(e)
    finally:
        connection.close()
def topic_modelling(corpus_raw = None):
    try:
        listStopword =  set(stopwords.words('indonesian'))
        listStopwordEn = set(stopwords.words('english'))
        if corpus_raw is None:
            corpus_raw = pd.read_csv("dataset_final_ib_pre.csv")
        # print(corpus_raw)
        dataset_clean = pd.DataFrame(columns=['post_id','text'])
        for x,y in zip(corpus_raw['text'],corpus_raw['post_id']) :
            x = str(x).replace("<i>","").replace("<b>","").replace("</i>","").replace("</b>","").split(" ")
            baru = ""
            for word in x:
                if word.lower() in listStopword or word.lower() in listStopwordEn:
                    x.remove(word)
                else:
                    baru = baru + word + " "
            dataset_clean = dataset_clean.append(pd.DataFrame([[y,baru]],columns=['post_id','text']))
        dataset_clean = dataset_clean.drop(columns=['post_id'],axis=1)
        dataset_clean.head()
        # print(corpus_raw)

        dataset_clean['text_preprocessed'] = dataset_clean['text'].map(lambda x:re.sub('[,\."!?]', '', x))
        dataset_clean['text_preprocessed'] = dataset_clean['text_preprocessed'].map(lambda x: x.lower())



        # print(dataset_clean['text_preprocessed'])
        dataset_clean['text_preprocessed'].head()
        # print(corpus_raw)

        long_string = ','.join(list(dataset_clean['text_preprocessed'].values))
        # print(long_string)

        # wordcloud = WordCloud(background_color='black',max_words=5000,contour_width=3,contour_color='steelblue')
        # wordcloud.generate(long_string)
        # wordcloud.to_file(filename="text.jpg")
        count_vectorizer = CountVectorizer()
        # Fit and transform the processed titles
        count_data = count_vectorizer.fit_transform(dataset_clean['text_preprocessed'])
        # Visualise the 10 most common words
        # plot_10_most_common_words(count_data, count_vectorizer)
        # Tweak the two parameters below
        number_topics = 10
        number_words = 2
        # Create and fit the LDA model
        lda = LDA(n_components=number_topics, n_jobs=-1)
        lda.fit(count_data)
        # Print the topics found by the LDA model
        print("Topics found via LDA:")
        print_topics(lda, count_vectorizer, number_words)
        ldavis_data_filepath = os.path.join('./ldavis_prepared_' + str(number_topics))
        ldavis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        # if 1 == 1:
        #
        # with open(ldavis_data_filepath, 'w') as f:
        #     pickle.dump(ldavis_prepared, f)
        #
        # # load the pre-prepared pyLDAvis data from disk
        # with open(ldavis_data_filepath) as f:
        #     ldavis_prepared = pickle.load(f)
        pyLDAvis.save_html(ldavis_prepared, './ldavis_prepared_' + str(number_topics) + '.html')

    except Exception as e:
        print(e)
        pass


# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# Initialise the count vectorizer with the English stop words

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))




# getStory()
topic_modelling(corpus_raw=None)