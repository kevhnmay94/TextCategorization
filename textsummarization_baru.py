import math
import operator
import re

import nltk
import translators as translation
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('indonesian'))
wordlemmatizer = WordNetLemmatizer()
translator = Translator(service_urls=['translate.google.com'])

def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words
def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text
def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq
def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb
def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf
def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf
def tf_idf_score(tf,idf):
    return tf*idf
def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf
def sentence_importance(sentence,dict_freq,sentences):
     sentence_score = 0
     sentence = remove_special_characters(str(sentence))
     sentence = re.sub(r'\d+', '', sentence)
     pos_tagged_sentence = []
     no_of_sentences = len(sentences)
     pos_tagged_sentence = pos_tagging(sentence)
     for word in pos_tagged_sentence:
          if word.lower() not in Stopwords and word not in Stopwords and len(word)>1:
                word = word.lower()
                word = wordlemmatizer.lemmatize(word)
                sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
     return sentence_score


def summarize_text(input_text:str, max_length_ratio=0.0, max_length_character=0, language='auto'):
    target_ratio = 0.0
    if(max_length_ratio==0.0 and max_length_character==0):
        return "[Error] Error in summarizing article."
    elif max_length_ratio == 0.0:
        target_ratio = (100.0 * max_length_character) / len(input_text)
    elif max_length_character == 0:
        target_ratio = max_length_ratio
    else:
        target_ratio = min(max_length_ratio, max_length_character / len(input_text))

    language_true = 'en'
    if language == 'auto':
        confidence = translator.detect(input_text)
        if confidence.lang == 'id':
            language_true = 'id'
        else:
            language_true = 'en'
    else:
        language_true = language

    if language_true == 'id':
        Stopwords = set(stopwords.words('indonesian'))
    else:
        Stopwords = set(stopwords.words('english'))

    tokenized_sentence = sent_tokenize(input_text)
    input_text = remove_special_characters(str(input_text))
    input_text = re.sub(r'\d+', '', input_text)
    tokenized_words_with_stopwords = word_tokenize(input_text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)
    no_of_sentences = int(target_ratio * len(tokenized_sentence))
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent, word_freq, tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c = c + 1
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1), reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt + 1
        else:
            break
    sentence_no.sort()
    cnt = 1
    n = 1
    for sentence in tokenized_sentence:
        if cnt in sentence_no:
            if len(sentence_no) <= 3:
                if n == 1:
                    sentence = "{}||".format(sentence)
            elif len(sentence_no) >= 4:
                if n == 2:
                    sentence = "{}||".format(sentence)
            summary.append(sentence)
            n = n + 1
        cnt = cnt + 1
    summary = " ".join(summary)
    if(len(summary)<=0):
        summary = "[Cannot summarize the article]"


    if language_true == 'id':
        sentence_split = summary.split("||")
        summary_stylized = []
        x = 1
        for sentence in sentence_split:
            if "\n\n" in sentence:
                sentence = sentence.replace("\n\n", ", ")

            if "\n" in sentence:
                sentence = sentence.replace("\n", ", ")
            if x == 1:
                sentence = "<b>{}</b>\n\n".format(sentence)
            else:
                sentence = "<i>{}</i>".format(sentence)
            summary_stylized.append(sentence)
            x = x + 1
        summary = "".join(summary_stylized)
        return summary
    else:
        translated = translation.google(summary, from_language='en', to_language='id')
        sentence_split = translated.split("||")
        summary_stylized = []
        x = 1
        for sentence in sentence_split:
            if "\n\n" in sentence:
                sentence = sentence.replace("\n\n", ", ")

            if "\n" in sentence:
                sentence = sentence.replace("\n", ", ")

            if x == 1:
                sentence = "<b>{}</b>\n\n".format(sentence)
            else:
                sentence = "<i>{}</i>".format(sentence)
            summary_stylized.append(sentence)
            x = x + 1
        translated = "".join(summary_stylized)
        return translated
        # translate_summary = translator.translate(summary,src='en', dest='id')
        # return translate_summary.text

def translate(input:str):
    translated = translation.google(input,from_language='en',to_language='id')
    return translated
def detect_language(input:str):
    detection = translator.detect(input)
    return detection
    # translate_summary = translator.translate(input,src='en', dest='id')
    # return translate_summary.text


# input_text = ""
# title = ""
#
# content = None
# news_link = None
# for i,s in enumerate(sys.argv[1:]):
#     if s[:2] == '--':
#         arg = s[2:]
#         if arg == 'content':
#             content = sys.argv[i+2]
#         if arg == 'link':
#             news_link = sys.argv[i+2]
#
# if news_link is not None:
#     url = news_link
#     article = Article(url)
#     article.download()
#     article.parse()
#     title = article.title
#     input_text = article.title + " " + article.text
# else:
#     if content is not None:
#         input_text = content
#
#
# input_length_ratio = 0.5
# input_length_character = 1000
#
# output = summarize_text(input_text, input_length_ratio, input_length_character)
# confidence = translator.detect(output)
# if confidence.lang == 'en':
#     print(title)
#     print(output)
# else:
#     translate_text = translator.translate(output, dest='en')
#     print(title)
#     print(translate_text.text)
