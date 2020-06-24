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
from PIL import Image

import mysql.connector
from newspaper import ArticleException
import ib_textcategorization

import crawler
import newsscraper
import textsummarization_baru
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
TIMEOUT = 3000


def retrieve_post_tuple(url: str, post_list: list, unique_id: int, f_pin: str, privacy_flag: int):
    try:
        text_block, title, image_src = crawler.crawl_article(url)
        data = []
        data.append(title + " " + text_block)
        category = ib_textcategorization.classify(data)
        cat_str = ""
        y = 0
        for cat in category:
            if y == 0:
                cat_str = "(" + cat + ","
            elif y != 0 and y != len(category) - 1:
                cat_str = cat_str + cat + ","
            elif y == len(category) - 1:
                cat_str = cat_str + cat + ")"
            y = y + 1

        title = textsummarization_baru.translate(title)
        summary = textsummarization_baru.summarize_text(text_block.replace("\n", " "), 1.0, 512, 'auto')
        if summary == "[Error] Error in summarizing article." or summary == "[Cannot summarize the article]":
            summary = "-"
        curtime_milli = int(round(time.time() * 1000))
        img_filename = ""

        # base_path_img = "/apps/indonesiabisa/server/image"
        image_total = ""
        if image_src:
            n = 0
            if type(image_src) == list:
                for image in image_src:
                    isUnicode = False
                    isWebp = False
                    extension = os.path.splitext(os.path.basename(image.split("?")[0]))[-1]
                    if extension == ".webp".casefold():
                        extension = ".jpg"
                        isWebp = True
                    img_filename = "APST-" + f_pin + "-" + format(curtime_milli, 'X') + "-" + str(n) + "-" + str(unique_id) + \
                                   extension
                    full_filename = os.path.join(base_path_img, img_filename)
                    for x in image:
                        if ord(x) > 127:
                            isUnicode = True
                    if isUnicode:
                        image = urllib.parse.quote(image, safe=":/")
                    urlretrieve(image, full_filename)
                    if isWebp:
                        webImage = Image.open(full_filename).convert("RGB")
                        webImage.save(full_filename,"jpeg")
                    # image_total = image_total + img_filename
                    n = n + 1
            elif type(image_src) == str:
                isUnicode = False
                isWebp = False
                extension = os.path.splitext(os.path.basename(image_src.split("?")[0]))[-1]
                if extension == ".webp".casefold():
                    extension = ".jpg"
                    isWebp = True
                img_filename = "APST-" + f_pin + "-" + format(curtime_milli, 'X') + "-" + str(n) + "-" + str(
                    unique_id) + \
                               extension
                full_filename = os.path.join(base_path_img, img_filename)
                for x in image_src:
                    if ord(x) > 127:
                        isUnicode = True
                if isUnicode:
                    image_src = urllib.parse.quote(image_src, safe=":/")
                urlretrieve(image_src, full_filename)
                if isWebp:
                    webImage = Image.open(full_filename).convert("RGB")
                    webImage.save(full_filename,"jpeg")

        post_id = f_pin + str(curtime_milli) + str(unique_id)
        if cat_str:
            post_values = (
                post_id, f_pin, urllib.parse.quote_plus(title), urllib.parse.quote_plus(summary), curtime_milli,
                privacy_flag,
                img_filename,
                img_filename, curtime_milli, url, 1, curtime_milli,cat_str)
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
    privacy = "3"

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
            query = "replace into POST(POST_ID, F_PIN, TITLE, DESCRIPTION, CREATED_DATE, PRIVACY, THUMB_ID, FILE_ID, LAST_UPDATE, LINK, FILE_TYPE, SCORE) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            for element in post_tuple_list:
                select_cursor.execute(
                    query,
                (element[0],element[1],element[2],element[3],element[4],element[5],element[6],element[7],element[8],element[9],element[10],element[11],))
                mydb.commit()
                print("Links posted: " + str(len(post_tuple_list)))

            post_url_tuples = [(post[1], post[9]) for post in post_tuple_list]
            query_cat = "replace into AUTO_POST_LINKS(F_PIN,URL) values (%s,%s)"
            for element in post_url_tuples:
                select_cursor.executemany(
                    query_cat,
                    (element,))
                mydb.commit()

            post_cat_tuples = [(post[0], category_id) for post in post_tuple_list]
            # for pid in post_id_list:
            #     post_cat_tuples.append((pid, category_id))
            query_cat = "replace into CONTENT_CATEGORY(POST_ID,CATEGORY) SELECT %s,ID from CATEGORY where CODE in %s"
            for element in post_cat_tuples:
                select_cursor.execute(
                    query_cat,
                    (element[0],element[12]))
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
news_bot_story.news_bot_story()
#     p = Process(target=get_post_news, args=(q_row,))
#     uid = uid + 1
#     p.start()
#     row_processes.append(p)
# for p in row_processes:
#     p.join(1200)
#     if p.is_alive():
#         p.terminate()
#         p.join()
