import joblib
import numpy as np
from pandas import DataFrame
from sklearn import metrics
from sklearn import model_selection, naive_bayes as nb_generator, svm as svm_generator
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import silhouette_score
import ast
from nltk.corpus import stopwords,verbnet

isVerbose = False

def vprint(*data):
    if isVerbose:
        print(*data)

CORPUS_VECTOR = 'tfidf_vector.pkl'
MNB_FILENAME = 'mnb_classifier.pkl'
SVM_FILENAME = 'svm_classifier.pkl'
MLP_FILENAME = 'mlp_classifier.pkl'
CLUSTER_FILENAME = 'cluster.pkl'

Train_X, Test_X, Train_Y, Test_Y = None, None, None, None
Encoder = LabelEncoder()
Train_Y_Encoded = None
Test_Y_Encoded = None
Train_X_Tfidf = None
Test_X_Tfidf = None
Cluster_X_Tfidf = None
unlabeled = None
fitTrainModel = True
probaPredict = False
corpus = None
partialTrain = True
corpusRaw = None
Tfidf_vect = None
modelToFile = True
nClusters = 20
top3Clusters = False

clustering_label = []

def prepare_cluster(corpus_input: DataFrame, corpus_raw: DataFrame, path, write_corpus=True,
                    fit_train_model=True, fit_corpus=True, verbose=False, new_data=None, cluster_label=True,
                    model_to_file=True, n_clusters=20, top_3_clusters=False):
    global isVerbose, corpus, CLUSTER_FILENAME, probaPredict, Cluster_X_Tfidf, unlabeled, fitTrainModel, corpusRaw, Tfidf_vect, modelToFile, nClusters, top3Clusters
    CLUSTER_FILENAME = path + CLUSTER_FILENAME
    isVerbose = verbose
    corpus = corpus_input
    fitTrainModel = fit_train_model
    corpusRaw = corpus_raw
    modelToFile = model_to_file
    nClusters = n_clusters
    top3Clusters = top_3_clusters
    vprint(CORPUS_VECTOR)
    if write_corpus is True:
        Tfidf_vect = None
    else:
        try:
            Tfidf_vect = joblib.load(CORPUS_VECTOR)
        except FileNotFoundError:
            Tfidf_vect = None
    if Tfidf_vect is None:
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        fit_corpus = True
    if fit_corpus:
        Tfidf_vect.fit(corpus['text_final'])
        joblib.dump(Tfidf_vect, CORPUS_VECTOR, compress=9)
    Cluster_X_Tfidf = Tfidf_vect.transform(corpus['text_final'])
    if new_data is not None:
        unlabeled = Tfidf_vect.transform([new_data])
    elif cluster_label:
        unlabeled = Cluster_X_Tfidf
    else:
        unlabeled = Tfidf_vect.transform(
            [
            # 'As August says good-bye to the dog days of summer and turns its weary head toward Labor Day, cable company '
            #  'call centers light up with callers requesting the RedZone channel. By now, this is hardly surprising. Each '
            #  'year that the NFL preseason approaches its zenith, a staggering 50 million players get ready to do their '
            #  'fantasy football drafts. The internet is flooded by fantasy football fever, with draft guides and strategy '
            #  'blogs fighting for their share of the click deluge. If you work at a company with more than a dozen people, '
            #  'it is virtually certain that someone in your office is playing fantasy football. During the NFL season, '
            #  'more people play fantasy than go to the gym, the movies, or music concerts. So even if you’re not a player '
            #  'or fan, there is no denying that fantasy football has become part of the cultural narrative. The '
            #  'interesting bit is that this game, which takes a staggering share of post-Labor Day leisure time, '
            #  'offers many surprising parallels with real-life business management. If you clicked on this article and '
            #  'read this far, it’s likely that you already know how fantasy football works. On the off chance that your '
            #  'knowledge is rusty, here is a short description: in fantasy football, you join a league, typically with '
            #  'friends or work colleagues. The league is set up and managed using a fantasy football app (CBS Sports and '
            #  'Yahoo are two of the most popular). At some point before the NFL regular season starts, you and the other '
            #  '“coaches” in the league will agree on a date to conduct a draft. The draft is done online using the fantasy '
            #  'football app, meaning you don’t have to be in the same room with everyone else in your league. The purpose '
            #  'of the draft is to select real-life NFL players to play on your fantasy team. Once you have drafted a '
            #  'player, that player is not available for any other team. There are two draft types. With a snake draft, '
            #  'all the coaches in your league draft based on a pre-set order. The person who drafts last in the first '
            #  'round gets to draft first in the second round, and so forth. With an auction draft, all the coaches in your '
            #  'league get a pre-set budget and can bid on players. Whoever bids the highest gets the player, but then has '
            #  'correspondingly less budget left for future bids. Once everyone has a full team the draft closes. '
            #  'Typically, a team is comprised of ten starters that include a quarterback, two running backs, three wide '
            #  'receivers, a tight end, a flex position, a kicker, and a defense. In addition, there are six bench spots '
            #  'for backup players that can include any of the above positions. Some leagues use either more or fewer '
            #  'starting positions, but they generally hew close to this general guideline. During the first 13 weeks of '
            #  'the NFL regular season, each team will go head-to-head against another team in your league. The fantasy '
            #  'football app you use will automatically calculate the real-life performance of each player on your team. '
            #  'The scores put up by each player are added together to give you an overall team score. If your team puts up '
            #  'more points in any given week than the team you are playing against then you get a win. Otherwise, '
            #  'you get a loss. At the end of the period, the teams with the best win/loss records go to the fantasy '
            #  'playoffs and, when all is said and done, the top three teams generally split a pot of winnings based on '
            #  'league fees.',
            #  'The alert follows last week’s warning from a UN-appointed independent rights expert that the country – once '
            #  'seen as the breadbasket of Africa - is in the grip of “man-made starvation”. In Geneva, WFP spokesperson '
            #  'Bettina Luescher said that almost $300 million was needed urgently to supply some 240,000 tonnes of aid. “A '
            #  'climate disaster” and “economic meltdown” were to blame for the ongoing crisis, she explained, with normal '
            #  'rainfall recorded in just one of the last five growing seasons. The increasingly unreliable rainy season '
            #  'affects subsistence farmers in particular as they grow maize - a very water-intensive crop, and many of '
            #  'these farmers are still recovering from the major 2014-16 El Nino-induced drought. In addition, “the crisis '
            #  'is being exacerbated by a dire shortage of currency, runaway inflation, mounting unemployment, '
            #  'lack of fuel, prolonged power outages and large-scale livestock losses, and they inflict the urban '
            #  'population just as well as rural villagers,” Ms. Luescher said. In total, however, 5.5 million people in '
            #  'the countryside and 2.2 million in urban areas need help, and acute malnutrition has risen to 3.6 per cent, '
            #  'up from 2.5 per cent last year. ',
            #  'Angelina Jolie tells daughters most attractive quality for women is '
            #  'having their own opinions Angelina Jolie has revealed she’s bringing up '
            #  'her daughters to recognise the importance of prioritising mental '
            #  'development over their looks. Writing for Elle UK  in the publication’s '
            #  'September issue, the actor argued that “there is nothing more '
            #  'attractive—you might even say enchanting—than a woman with an '
            #  'independent will and her own opinions”. Jolie continued: “I often tell '
            #  'my daughters that the most important thing they can do is to develop '
            #  'their minds. You can always put on a pretty dress, but it doesn’t '
            #  'matter what you wear on the outside if your mind isn’t strong.” '
            #  'Elsewhere in the essay, Jolie pontificates about the need for more '
            #  '“wicked women” in the world, proceeding to challenge gender stereotypes '
            #  'surrounding strong women. ',
             'The Google founders’ decision to step down ends a multiyear effort to turn their company into the Berkshire '
             'Hathaway of technology by embracing Warren Buffett’s hands-off management style. Larry Page and Sergey Brin '
             'created the Alphabet Inc. holding company in 2015 to give themselves more time to invest in new tech '
             'businesses and handed responsibility for Google to Sundar Pichai. The model was inspired by Buffett’s '
             'approach of allocating capital to disparate businesses and letting independent CEOs decide how to run the '
             'operations. On Tuesday, the Google founders effectively unwound this structure by making Pichai chief '
             'executive officer of both Google and Alphabet. Pichai was already busy running Google’s gargantuan digital '
             'advertising businesses and responding to antitrust probes, political assaults and protesting workers. Now '
             'the self-driving cars, health-care projects, digital cities, delivery drones and internet-beaming balloons '
             'are his problem, too. For many at the company, Alphabet’s purpose and structure was never really clear. '
             'Placing the head of Google, which contributes more than 99% of Alphabet’s sales, at the helm of it could '
             'call into question the entire purpose of Alphabet, one former Google senior employee said. Another former '
             'Google executive said the change will mean Pichai is stretched even more thinly. They asked not to be '
             'identified discussing private matters. Financially, Alphabet was a win because it showed investors that the '
             'company wasn’t spending too much on ambitious “moonshot” projects, while highlighting the huge '
             'profitability of the main Google business. But operationally, the structure has been in near-constant '
             'tumult and has struggled to produce a new business remotely close to Google in size and scope. '
             'Collectively, the company’s “Other Bets,” which include Waymo driverless cars and Verily health-care tech, '
             'lost $3.4 billion in 2018 and almost $1 billion in the latest quarter. Nest, a smart-device maker, '
             'started out as a standalone Alphabet company but moved into Google’s hardware division last year. Pichai '
             'also brought many of DeepMind’s ambitious artificial intelligence projects into Google’s fold. Chronicle, '
             'a cybersecurity project, debuted with considerable fanfare as an independent Alphabet business last year -- '
             'only to be subsumed into Google’s cloud division in June. Fiber, once a high-profile Other Bet, '
             'is no longer expanding.',
             ])

def cluster(model=None):
    global modelToFile
    if type(nClusters) is list:
        modelToFile = False
        for n in nClusters:
            cluster_predict(n,False)
        return None
    elif type(nClusters) is int:
        c = cluster_predict(nClusters,isVerbose,model)
        return c

def cluster_predict(n_classes,print_detail,model=None):
    global fitTrainModel
    cluster_model = None
    if modelToFile:
        try:
            cluster_model = model
        except FileNotFoundError:
            pass
    if cluster_model is None:
        cluster_model = KMeans(
            n_clusters=n_classes, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        fitTrainModel = True
    if fitTrainModel:
        cat_num = cluster_model.n_clusters
        X = Cluster_X_Tfidf
        if top3Clusters:
            y_km = cluster_model.fit_transform(X)
            cntr = Counter()
            rankd = []
            for c in y_km:
                arr = np.array(c)
                sortd = arr.argsort()[:3].tolist()
                rankd.append(sortd)
                if isVerbose:
                    cntr += Counter(sortd)
            if isVerbose:
                for r in rankd:
                    vprint(r)
                vprint(cntr)
            y_km = rankd
        else:
            y_km = cluster_model.fit_predict(X)
            if isVerbose:
                labels = cluster_model.labels_
                cluster_members = cat_num*[None]
                clustering_label = generate_label(cat_num,y_km)
                for i in range(cat_num):
                    cnt = sum(x == i for x in y_km)
                    if print_detail:
                        vprint("Cluster {} | Members: {} | Value {}".format(i,cnt,clustering_label[i]))
                    cluster_members[i] = cnt
                vprint("Clusters: {} | Standard deviation: {} | Silhouette score: {}".format(cat_num,np.std(cluster_members),silhouette_score(X,labels)))
                vprint(y_km)
        if modelToFile:
            joblib.dump(cluster_model, CLUSTER_FILENAME, compress=9)
        return y_km
    else:
        c = cluster_model.transform(unlabeled)
        arr = np.array(c[0])
        sortd = arr.argsort()[:3]
        return sortd.tolist()

def generate_label(n_clusters,cluster_result):
    cat = n_clusters*[None]
    for i in range(n_clusters):
        d = [idx for idx,val in enumerate(cluster_result) if val == i]
        counter = Counter()
        sl = corpus.iloc[d,:]['text_final']
        tfidf = Tfidf_vect.transform(sl)
        feature_names = Tfidf_vect.get_feature_names()
        for doc in range(len(d)):
            dic = {}
            feature_index = tfidf[doc, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf[doc, x] for x in feature_index])
            for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                dic[w] = s
            c = Counter(dic)
            counter += c
        cat[i] = [(x,y/len(d)) for (x,y) in counter.most_common(3)]
    return cat

def prepare(corpus_input: DataFrame, path, write_corpus=False, fit_corpus=False, fit_train_model=False,
            proba=True, partial=False, verbose=False, new_data = None):
    global Train_X, Test_X, Train_Y, Test_Y, Train_Y_Encoded, Test_Y_Encoded, Train_X_Tfidf, Test_X_Tfidf, unlabeled, \
        fitTrainModel, probaPredict, isVerbose, CORPUS_VECTOR, MNB_FILENAME, MLP_FILENAME, SVM_FILENAME, corpus, partialTrain
    corpus = corpus_input
    CORPUS_VECTOR = path + CORPUS_VECTOR
    MNB_FILENAME = path + MNB_FILENAME
    SVM_FILENAME = path + SVM_FILENAME
    MLP_FILENAME = path + MLP_FILENAME
    fitTrainModel = fit_train_model
    probaPredict = proba
    isVerbose = verbose
    partialTrain = partial
    test_size = 0.25 if fitTrainModel else corpus['text_final'].size - 1
    Train_X, Test_X, Train_Y, Test_Y = None, None, None, None

    vprint('Encoding labels...')

    Encoder.fit_transform(corpus['category'])
    if write_corpus is True:
        Tfidf_vect = None
    else:
        try:
            Tfidf_vect = joblib.load(CORPUS_VECTOR)
        except FileNotFoundError:
            Tfidf_vect = None
    if Tfidf_vect is None:
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        fit_corpus = True
    if fit_corpus:
        Tfidf_vect.fit(corpus['text_final'])
        joblib.dump(Tfidf_vect, CORPUS_VECTOR, compress=9)
        vprint('Preparing train and test data sets...')
    if fitTrainModel:
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'], corpus['category'],
                                                                            test_size=test_size)
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    if new_data is not None:
        unlabeled = Tfidf_vect.transform(new_data)
    else:
        unlabeled = Tfidf_vect.transform(
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
             'league fees.',
            #  """
            #
            #  """
             # 'The alert follows last week’s warning from a UN-appointed independent rights expert that the country – once '
             # 'seen as the breadbasket of Africa - is in the grip of “man-made starvation”. In Geneva, WFP spokesperson '
             # 'Bettina Luescher said that almost $300 million was needed urgently to supply some 240,000 tonnes of aid. “A '
             # 'climate disaster” and “economic meltdown” were to blame for the ongoing crisis, she explained, with normal '
             # 'rainfall recorded in just one of the last five growing seasons. The increasingly unreliable rainy season '
             # 'affects subsistence farmers in particular as they grow maize - a very water-intensive crop, and many of '
             # 'these farmers are still recovering from the major 2014-16 El Nino-induced drought. In addition, “the crisis '
             # 'is being exacerbated by a dire shortage of currency, runaway inflation, mounting unemployment, '
             # 'lack of fuel, prolonged power outages and large-scale livestock losses, and they inflict the urban '
             # 'population just as well as rural villagers,” Ms. Luescher said. In total, however, 5.5 million people in '
             # 'the countryside and 2.2 million in urban areas need help, and acute malnutrition has risen to 3.6 per cent, '
             # 'up from 2.5 per cent last year. ',
             # 'Angelina Jolie tells daughters most attractive quality for women is '
             # 'having their own opinions Angelina Jolie has revealed she’s bringing up '
             # 'her daughters to recognise the importance of prioritising mental '
             # 'development over their looks. Writing for Elle UK  in the publication’s '
             # 'September issue, the actor argued that “there is nothing more '
             # 'attractive—you might even say enchanting—than a woman with an '
             # 'independent will and her own opinions”. Jolie continued: “I often tell '
             # 'my daughters that the most important thing they can do is to develop '
             # 'their minds. You can always put on a pretty dress, but it doesn’t '
             # 'matter what you wear on the outside if your mind isn’t strong.” '
             # 'Elsewhere in the essay, Jolie pontificates about the need for more '
             # '“wicked women” in the world, proceeding to challenge gender stereotypes '
             # 'surrounding strong women. ',
             # 'The Google founders’ decision to step down ends a multiyear effort to turn their company into the Berkshire '
             # 'Hathaway of technology by embracing Warren Buffett’s hands-off management style. Larry Page and Sergey Brin '
             # 'created the Alphabet Inc. holding company in 2015 to give themselves more time to invest in new tech '
             # 'businesses and handed responsibility for Google to Sundar Pichai. The model was inspired by Buffett’s '
             # 'approach of allocating capital to disparate businesses and letting independent CEOs decide how to run the '
             # 'operations. On Tuesday, the Google founders effectively unwound this structure by making Pichai chief '
             # 'executive officer of both Google and Alphabet. Pichai was already busy running Google’s gargantuan digital '
             # 'advertising businesses and responding to antitrust probes, political assaults and protesting workers. Now '
             # 'the self-driving cars, health-care projects, digital cities, delivery drones and internet-beaming balloons '
             # 'are his problem, too. For many at the company, Alphabet’s purpose and structure was never really clear. '
             # 'Placing the head of Google, which contributes more than 99% of Alphabet’s sales, at the helm of it could '
             # 'call into question the entire purpose of Alphabet, one former Google senior employee said. Another former '
             # 'Google executive said the change will mean Pichai is stretched even more thinly. They asked not to be '
             # 'identified discussing private matters. Financially, Alphabet was a win because it showed investors that the '
             # 'company wasn’t spending too much on ambitious “moonshot” projects, while highlighting the huge '
             # 'profitability of the main Google business. But operationally, the structure has been in near-constant '
             # 'tumult and has struggled to produce a new business remotely close to Google in size and scope. '
             # 'Collectively, the company’s “Other Bets,” which include Waymo driverless cars and Verily health-care tech, '
             # 'lost $3.4 billion in 2018 and almost $1 billion in the latest quarter. Nest, a smart-device maker, '
             # 'started out as a standalone Alphabet company but moved into Google’s hardware division last year. Pichai '
             # 'also brought many of DeepMind’s ambitious artificial intelligence projects into Google’s fold. Chronicle, '
             # 'a cybersecurity project, debuted with considerable fanfare as an independent Alphabet business last year -- '
             # 'only to be subsumed into Google’s cloud division in June. Fiber, once a high-profile Other Bet, '
             # 'is no longer expanding.',
             ])


def test_mnb(mnb_model):
    global fitTrainModel
    vprint('Classify dataset using Naive Bayes...')
    if mnb_model is None or fitTrainModel:
        mnb_model = nb_generator.MultinomialNB()
        fitTrainModel = True
    if fitTrainModel:
        mnb_model.fit(Train_X_Tfidf, Train_Y)
        joblib.dump(mnb_model, MNB_FILENAME, compress=9)
        predictions_NB = mnb_model.predict(Test_X_Tfidf)
        accuracy = metrics.accuracy_score(y_pred=predictions_NB, y_true=Test_Y) * 100
        precision = metrics.precision_score(y_pred=predictions_NB, y_true=Test_Y, average='macro') * 100
        recall = metrics.recall_score(y_pred=predictions_NB, y_true=Test_Y, average='macro') * 100
        vprint("Naive Bayes Accuracy Score -> ", f'{accuracy:.2f}%')
        vprint("Recall -> ", f'{recall:.2f}%')
        vprint("Precision -> ", f'{precision:.2f}%')
    if probaPredict:
        c = mnb_model.predict_proba(unlabeled)
        vprint(c)
        arr = np.array(c[3])
        d = Encoder.inverse_transform(arr.argsort()[::-1][:3])
        vprint(d)
        return d
    else:
        c = mnb_model.predict(unlabeled)
        vprint(c[0])
        return c[0]


def test_svm(svm_model):
    global fitTrainModel
    vprint('Classify dataset using SVM...')
    if svm_model is None or fitTrainModel:
        svm_model = svm_generator.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
        fitTrainModel = True
    if fitTrainModel:
        svm_model.fit(Train_X_Tfidf, Train_Y)
        joblib.dump(svm_model, SVM_FILENAME, compress=9)
        predictions_SVM = svm_model.predict_proba(Test_X_Tfidf)
        accuracy2 = metrics.accuracy_score(y_pred=predictions_SVM, y_true=Test_Y) * 100
        precision2 = metrics.precision_score(y_pred=predictions_SVM, y_true=Test_Y, average='macro') * 100
        recall2 = metrics.recall_score(y_pred=predictions_SVM, y_true=Test_Y, average='macro') * 100
        vprint("SVM Accuracy Score -> ", f'{accuracy2:.2f}%')
        vprint("Recall -> ", f'{recall2:.2f}%')
        vprint("Precision -> ", f'{precision2:.2f}%')
    if probaPredict:
        c = svm_model.predict_proba(unlabeled)
        vprint(c)
        arr = np.array(c[3])
        d = Encoder.inverse_transform(arr.argsort()[::-1][:3])
        vprint(d)
        return d
    else:
        c = svm_model.predict(unlabeled)
        vprint(c[0])
        return c[0]

def load_model(model_name):
    model = None
    try:
        model = joblib.load(model_name)
    except FileNotFoundError as e:
        vprint(e)
    return model


def test_mlp(mlp_model):
    global fitTrainModel, partialTrain
    if mlp_model is None or fitTrainModel:
        mlp_model = MLPClassifier()
        fitTrainModel = True
        partialTrain = False
    mlp_model.set_params(max_iter=400, solver='adam', hidden_layer_sizes=(500,))
    if fitTrainModel:
        if partialTrain:
            mlp_model.partial_fit(Train_X_Tfidf)
        else:
            mlp_model.fit(Train_X_Tfidf, Train_Y)
        joblib.dump(mlp_model, MLP_FILENAME, compress=9)
        predictions_MLP = mlp_model.predict(Test_X_Tfidf)
        accuracy3 = metrics.accuracy_score(y_pred=predictions_MLP, y_true=Test_Y) * 100
        precision3 = metrics.precision_score(y_pred=predictions_MLP, y_true=Test_Y, average='macro') * 100
        recall3 = metrics.recall_score(y_pred=predictions_MLP, y_true=Test_Y, average='macro') * 100
        vprint("MLP Accuracy Score -> ", f'{accuracy3:.2f}%')
        vprint("Recall -> ", f'{recall3:.2f}%')
        vprint("Precision -> ", f'{precision3:.2f}%')
        return None
    if probaPredict:
        labels = Encoder.classes_
        if isVerbose:
            for idx,val in enumerate(labels):
                vprint("{}. {}".format(idx,val))
        c = mlp_model.predict_proba(unlabeled)
        prob_value = np.sort(c[0])[::-1][:3].tolist()
        arr = np.array(c[0])
        sortd = arr.argsort()[::-1][:3]
        d = Encoder.inverse_transform(sortd)
        x = [d.tolist(),d.tolist()[:1],prob_value]
        return x
    else:
        c = mlp_model.predict(unlabeled)
        vprint(c[0])
        mlp_model.partial_fit(unlabeled,c)
        joblib.dump(mlp_model,MLP_FILENAME, compress=9)
        return c[0]

