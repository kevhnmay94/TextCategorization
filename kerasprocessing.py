import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
from pandas import DataFrame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

CORPUS_VECTOR = 'tfidf_keras_vector.pkl'
KERAS_FILENAME = 'keras_classifier.pkl'
Encoder = LabelBinarizer()
unlabeled = None
fitTrainModel = True
isVerbose = False

def exec(corpus: DataFrame, path, write_corpus=True, fit_corpus=True, fit_train_model=True, verbose=False, new_data=None):
    global isVerbose, CORPUS_VECTOR, KERAS_FILENAME
    isVerbose = verbose
    CORPUS_VECTOR = path + CORPUS_VECTOR
    KERAS_FILENAME = path + KERAS_FILENAME
    np.random.seed(420)
    global fitTrainModel
    fitTrainModel = fit_train_model
    word_size = 5000
    labels = corpus['category'].unique()
    num_labels = len(labels)
    vprint('Preparing train and test data sets...')
    test_size = 0.25 if fitTrainModel else corpus['text_final'].size - 1
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'], corpus['category'],
                                                                        test_size=test_size)
    vprint('Encoding labels...')
    Train_Y_Encoded = Encoder.fit_transform(Train_Y)
    Test_Y_Encoded = Encoder.fit_transform(Test_Y)
    vprint(Train_Y_Encoded)

    if write_corpus is True:
        tokenizer = None
    else:
        try:
            tokenizer = joblib.load(CORPUS_VECTOR)
        except FileNotFoundError:
            tokenizer = None
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=word_size)
        fit_corpus = True
    if fit_corpus:
        tokenizer.fit_on_texts(corpus['text_final'])
        joblib.dump(tokenizer, CORPUS_VECTOR, compress=9)

    Train_X_Tfidf = tokenizer.texts_to_matrix(Train_X, mode='tfidf')
    Test_X_Tfidf = tokenizer.texts_to_matrix(Test_X, mode='tfidf')
    if new_data is not None:
        unlabeled = tokenizer.texts_to_matrix([new_data])
    else:
        unlabeled = tokenizer.texts_to_matrix(
            [
                'As August says good-bye to the dog days of summer and turns its weary head toward Labor Day, cable company '
             'call centers light up with callers requesting the RedZone channel. By now, this is hardly surprising. Each '
             'year that the NFL preseason approaches its zenith, a staggering 50 million players get ready to do their '
             'fantasy football drafts. The internet is flooded by fantasy football fever, with draft guides and strategy '
             'blogs fighting for their share of the click deluge. If you work at a company with more than a dozen people, '
             'it is virtually certain that someone in your office is playing fantasy football. During the NFL season, '
             'more people play fantasy than go to the gym, the movies, or music concerts. So even if you’re not a player '
             'or fan, there is no denying that fantasy football has become part of the cultural narrative. The '
             'interesting bit is that this game, which takes a staggering share of post-Labor Day leisure time, '
             'offers many surprising parallels with real-life business management. If you clicked on this article and '
             'read this far, it’s likely that you already know how fantasy football works. On the off chance that your '
             'knowledge is rusty, here is a short description: in fantasy football, you join a league, typically with '
             'friends or work colleagues. The league is set up and managed using a fantasy football app (CBS Sports and '
             'Yahoo are two of the most popular). At some point before the NFL regular season starts, you and the other '
             '“coaches” in the league will agree on a date to conduct a draft. The draft is done online using the fantasy '
             'football app, meaning you don’t have to be in the same room with everyone else in your league. The purpose '
             'of the draft is to select real-life NFL players to play on your fantasy team. Once you have drafted a '
             'player, that player is not available for any other team. There are two draft types. With a snake draft, '
             'all the coaches in your league draft based on a pre-set order. The person who drafts last in the first '
             'round gets to draft first in the second round, and so forth. With an auction draft, all the coaches in your '
             'league get a pre-set budget and can bid on players. Whoever bids the highest gets the player, but then has '
             'correspondingly less budget left for future bids. Once everyone has a full team the draft closes. '
             'Typically, a team is comprised of ten starters that include a quarterback, two running backs, three wide '
             'receivers, a tight end, a flex position, a kicker, and a defense. In addition, there are six bench spots '
             'for backup players that can include any of the above positions. Some leagues use either more or fewer '
             'starting positions, but they generally hew close to this general guideline. During the first 13 weeks of '
             'the NFL regular season, each team will go head-to-head against another team in your league. The fantasy '
             'football app you use will automatically calculate the real-life performance of each player on your team. '
             'The scores put up by each player are added together to give you an overall team score. If your team puts up '
             'more points in any given week than the team you are playing against then you get a win. Otherwise, '
             'you get a loss. At the end of the period, the teams with the best win/loss records go to the fantasy '
             'playoffs and, when all is said and done, the top three teams generally split a pot of winnings based on '
             'league fees.'],
            mode='tfidf')

    try:
        model = models.load_model(KERAS_FILENAME)
    except OSError:
        model = None
    except:
        import sys
        e = sys.exc_info()
        vprint(e)
        exit(1)
    if model is None:
        model = tf.keras.Sequential()
        model.add(layers.Dense(500, activation=tf.keras.activations.relu, input_shape=(word_size,)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(500, activation=tf.keras.activations.relu))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_labels, activation=tf.keras.activations.softmax))
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    if fitTrainModel:
        model.fit(Train_X_Tfidf, Train_Y_Encoded, epochs=50, batch_size=10, validation_split=0.1)
        models.save_model(model, KERAS_FILENAME)
    text_labels = Encoder.classes_
    score = model.evaluate(Test_X_Tfidf, Test_Y_Encoded, batch_size=10)
    prediction1 = model.predict(Test_X_Tfidf)
    for i in range(0, 10):
        vprint(text_labels[np.argmax(prediction1[i])], Test_Y.iloc[i])
    for u in unlabeled:
        prediction = model.predict(np.array([u]))
        vprint(text_labels)
        vprint(prediction)
        arr = np.argsort(prediction[0])[::-1][:3]
        lab = []
        for a in arr:
            lab.append(text_labels[a])
        vprint(lab)

def vprint(*data):
    if isVerbose:
        print(*data)