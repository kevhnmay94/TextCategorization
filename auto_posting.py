#! /usr/local/bin/python3.7
import os
import sys
import time
import urllib.parse
from datetime import datetime
from http.client import RemoteDisconnected, IncompleteRead
from multiprocessing import Process, Manager
from ssl import SSLEOFError
from urllib.error import HTTPError
from urllib.request import urlretrieve

import mysql.connector
from newspaper import ArticleException

import crawler
import newsscraper
import textsummarization

mydb = mysql.connector.connect(
    host="202.158.33.27",
    user="nup",
    passwd="5m1t0l_aptR",
    database="bwm_1"
)
TIMEOUT = 3000


def retrieve_post_tuple(url: str, post_list: list, unique_id: int, f_pin: str, privacy_flag: int):
    try:
        text_block, title, image_src = crawler.crawl_article(url)
        summary = textsummarization.summarize_text(text_block.replace("\n", " "), 1.0, 1000, 'auto')
        if summary == "[Error] Error in summarizing article." or summary == "[Cannot summarize the article]":
            summary = "-"
        curtime_milli = int(round(time.time() * 1000))
        img_filename = ""
        base_path_img = "/apps/cub/server/image"
        if image_src:
            img_filename = "APST-" + f_pin + "-" + format(curtime_milli, 'X') + "-" + str(unique_id) + \
                           os.path.splitext(os.path.basename(image_src.split("?")[0]))[-1]
            full_filename = os.path.join(base_path_img, img_filename)
            urlretrieve(image_src, full_filename)

        post_id = f_pin + str(curtime_milli) + str(unique_id)

        post_values = (
            post_id, f_pin, urllib.parse.quote_plus(title), urllib.parse.quote_plus(summary), curtime_milli,
            privacy_flag,
            img_filename,
            img_filename, curtime_milli, url, 1, curtime_milli)
        post_list.append(post_values)
        print("Success in fetching " + url)
    except (HTTPError, RemoteDisconnected,ArticleException,IncompleteRead,SSLEOFError):
        print("Error in fetching " + url + " :\n" + str(sys.exc_info()[0]))
        pass


def get_post_news(row: list):
    is_active = row[6]
    if is_active is not 1:
        return

    auto_post_id = row[0]
    f_pin = row[1]
    domain = row[2]
    category_id = row[3]
    last_update = row[4]
    privacy = row[5]

    category = ""
    if category_id == 4:
        category = "news"
    elif category_id == 10:
        category = "sport"
    elif category_id == 11:
        category = "technology"

    latest = datetime.now()
    earliest = last_update

    last_update_ins = int(round(datetime.now().timestamp()))
    news = newsscraper.fetch_news_list(domain, category, latest, earliest)
    print("Fetching " + str(len(news)) + " link(s)...")
    if len(news) > 0:
        with Manager() as manager:
            post_tuple_list = manager.list()
            news_processes = []
            uid = 1
            for link in reversed(news):
                print("Checking url " + link)
                query_check = "SELECT * from AUTO_POST_LINKS where F_PIN = '" + f_pin + "' and URL = '" + link + "' limit 1"
                select_cursor.execute(query_check)
                check_post_result = select_cursor.fetchall()
                if len(check_post_result) == 0:
                    p = Process(target=retrieve_post_tuple, args=(link, post_tuple_list, uid, f_pin, privacy))
                    uid = uid + 1
                    p.start()
                    news_processes.append(p)
            # for p in news_processes:
            #     p.join()
            #timeout
            start = time.time()
            while time.time() - start <= TIMEOUT:
                if not any(p.is_alive() for p in news_processes):
                    # All the processes are done, break now.
                    break

                time.sleep(30)  # Just to avoid hogging the CPU
            else:
                # We only enter this if we didn't 'break' above.
                print("timed out, killing all processes")
                for p in news_processes:
                    p.terminate()
                    p.join()

            print("Posting links: " + str(len(post_tuple_list)))
            query = "insert into POST(POST_ID, F_PIN, TITLE, DESCRIPTION, CREATED_DATE, PRIVACY, THUMB_ID, FILE_ID, LAST_UPDATE, LINK, FILE_TYPE, SCORE) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            select_cursor.executemany(
                query,
                post_tuple_list)
            mydb.commit()
            print("Links posted: " + str(len(post_tuple_list)))

            post_url_tuples = [(post[1], post[9]) for post in post_tuple_list]
            query_cat = "insert into AUTO_POST_LINKS(F_PIN,URL) values (%s,%s)"
            select_cursor.executemany(
                query_cat,
                post_url_tuples)
            mydb.commit()

            post_cat_tuples = [(post[0], category_id) for post in post_tuple_list]
            # for pid in post_id_list:
            #     post_cat_tuples.append((pid, category_id))
            query_cat = "insert into CONTENT_CATEGORY(POST_ID,CATEGORY) values (%s,%s)"
            select_cursor.executemany(
                query_cat,
                post_cat_tuples)
            mydb.commit()

            if len(post_tuple_list) > 0:
                select_cursor.execute(
                    "update AUTO_POST set LAST_UPDATE = from_unixtime(" + str(last_update_ins) + ") where ID = " + str(
                        auto_post_id))
                mydb.commit()
                print(str(last_update_ins))
    pass


select_cursor = mydb.cursor()
select_cursor.execute("SELECT * FROM AUTO_POST where IS_ACTIVE = 1")
auto_post_result = select_cursor.fetchall()

row_processes = []
if(len(auto_post_result) > 0):
    TIMEOUT = min(300, 3000 / len(auto_post_result))

for q_row in auto_post_result:
    get_post_news(q_row)
#     p = Process(target=get_post_news, args=(q_row,))
#     uid = uid + 1
#     p.start()
#     row_processes.append(p)
# for p in row_processes:
#     p.join(1200)
#     if p.is_alive():
#         p.terminate()
#         p.join()
