#! /usr/local/bin/python3.7
import sys
import time
from datetime import datetime
from http.client import RemoteDisconnected, IncompleteRead
from multiprocessing import Process, Manager
from ssl import SSLEOFError
from urllib.error import HTTPError
import mysql.connector
from newspaper import ArticleException
import ib_textcategorization
import pandas as pd

import crawler
import news_bot_story

with open("database.txt") as f:
    props = [line.rstrip() for line in f]

mydb = mysql.connector.connect(
    host=props[0],
    user=props[1],
    passwd=props[2],
    database=props[3]
)
base_path_img = props[4]
base_path_log = props[5]
TIMEOUT = 3000
timenow = "{}-{}-{}".format(datetime.now().day,datetime.now().month,datetime.now().year)
daily_log = pd.DataFrame(columns=['category','headline','content','probabilities','top3'])

def translate_category(input:str):
    if input == "Makanan":
        return "Food"

    elif input == "Hiburan":
        return "Entertainment"

    elif input == "Edukasi":
        return "Education"

    elif input == "Bisnis":
        return "Business"

    elif input == "Travel":
        return "Travel"

    elif input == "Berita":
        return "News"

    elif input == "Lain-Lain":
        return "Others"

    elif input == "Tren":
        return "Trending"

    elif input == "Kesehatan":
        return "Health"

    elif input == "Teknologi":
        return "Technology"

    elif input == "Gaya Hidup":
        return "Lifestyle"

    elif input == "Olahraga":
        return "Sports"

    elif input == "Selebriti":
        return "Celebrity"

    elif input == "Sains":
        return "Science"

    elif input == "Lowongan Pekerjaan":
        return "Jobs"

    elif input == "COVID-19":
        return "COVID-19"


def retrieve_post_tuple(url: str, post_list: list):
    try:
        global daily_log
        text_block, title, image_src = crawler.crawl_article(url)
        data = [title + " " + text_block]
        category = ib_textcategorization.classify(data)

        dataset = pd.DataFrame(data={'category': [category[1]], 'headline': [title], 'content': [text_block],
                                     'probabilities': [category[2]],
                                     'top3': [category[0]]},columns=['category','headline','content','probabilities','top3'])
        daily_log = daily_log.append(dataset)
        if category:
            print("Category TC : {}".format(category[0]))
            post_values = (category[0])
            post_list.append(post_values)
        print("Success in fetching " + url)
    except (HTTPError, RemoteDisconnected,ArticleException,IncompleteRead,SSLEOFError):
        print("Error in fetching " + url + " :\n" + str(sys.exc_info()[0]))
        pass


def get_post_news(row: list):
    post_id = row[0]
    f_pin = row[1]
    link = row[2]


    last_update_ins = int(round(datetime.now().timestamp()))
    if len(link) > 0:
        with Manager() as manager:
            post_tuple_list = manager.list()
            news_processes = []
            print("Checking url " + link)
            p = Process(target=retrieve_post_tuple, args=(link, post_tuple_list))
            p.start()
            news_processes.append(p)
            #timeout
            start = time.time()
            while time.time() - start <= TIMEOUT:
                if not any(p.is_alive() for p in news_processes):
                    # All the processes are done, break now.
                    break

                time.sleep(2)  # Just to avoid hogging the CPU
            else:
                # We only enter this if we didn't 'break' above.
                print("timed out, killing all processes")
                for p in news_processes:
                    p.terminate()
                    p.join()


            post_cat_tuples = [(post_id, post) for post in post_tuple_list]
            print(post_cat_tuples)
            query_cat = "replace into CONTENT_CATEGORY(POST_ID,CATEGORY) SELECT %s,ID from CATEGORY where CODE = %s"
            for element in post_cat_tuples:
                for category in element[1]:
                    print("Category : {}".format(category))
                    translated_category = translate_category(category)
                    print("Category Translate : {}".format(translated_category))
                    select_cursor.execute(
                        query_cat,
                        (element[0],translated_category))
                    # mydb.commit()

            print(post_tuple_list)
            if len(post_tuple_list) > 0:
                select_cursor.execute(
                    "update AUTO_POST set LAST_UPDATE = from_unixtime(" + str(last_update_ins) + ") where IS_ACTIVE = 1")
                print(str(last_update_ins))
            mydb.commit()
    pass


select_cursor = mydb.cursor()

query_check = "select P.POST_ID,A.F_PIN,P.LINK from POST P, AUTO_POST_LINKS A where P.LINK = A.URL and A.ID >= 97"
select_cursor.execute(query_check)
news = select_cursor.fetchall()

row_processes = []
if(len(news) > 0):
    TIMEOUT = min(300, 3000 / 14)

for q_row in news:
    get_post_news(q_row)
news_bot_story.news_bot_story()
daily_log.to_csv(base_path_log + 'dataset-ib-{}.csv'.format(timenow),mode='w',header=True,index=False)
