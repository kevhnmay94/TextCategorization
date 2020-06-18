import pandas as pd
import pymysql.cursors
import mid_time
import datetime

def news_bot_story():
    with open("database.txt") as f:
        props = [line.rstrip() for line in f]

    # Connect to the database
    connection = pymysql.connect(host=props[0],
                                 user=props[1],
                                 password=props[2],
                                 db=props[3])

    query = "SELECT DISTINCT `F_PIN` FROM `AUTO_POST`"
    fpin = pd.read_sql(query,connection)['F_PIN'][0]
    sql = "SELECT `ID`,`CODE` FROM `CATEGORY`"
    category = pd.read_sql(sql,connection)
    id = category['ID'].tolist()
    code = category['CODE'].tolist()

    try:
        sql = "SELECT COUNT(`STORY_ID`) AS `COUNT_STORY_ID` FROM `POST_STORY` WHERE `F_PIN` = '{}'".format(fpin)
        cnt_story_ids = pd.read_sql(sql, connection)['COUNT_STORY_ID'][0]
        if cnt_story_ids == 0:
            for val1,val2 in zip(id,code):
                sql2 = "SELECT `POST_ID` FROM `CONTENT_CATEGORY` WHERE `CATEGORY` = {}".format(val1)
                post_ids = pd.read_sql(sql2, connection)['POST_ID'].to_list()
                str_pid = ",".join(post_ids)
                with connection.cursor() as cursor:
                    # Read a single record
                    sql3 = "REPLACE INTO `POST_STORY` (`STORY_ID`,`STORY_NAME`,`F_PIN`,`POST_ID`,`STORY_DATE`) VALUES (%s,%s,%s,%s,%s)"
                    story_id = fpin + mid_time.next()
                    tn = round(datetime.datetime.now().timestamp() * 1000)
                    cursor.execute(sql3, (story_id,val2,fpin,str_pid,tn))
        else:
            for val1, val2 in zip(id, code):
                sql2 = "SELECT `POST_ID` FROM `CONTENT_CATEGORY` WHERE `CATEGORY` = {}".format(val1)
                post_ids = pd.read_sql(sql2, connection)['POST_ID'].to_list()
                str_pid = ",".join(post_ids)
                with connection.cursor() as cursor:
                    # Read a single record
                    sql3 = "UPDATE `POST_STORY` SET `POST_ID` =  %s and `STORY_DATE` = %s where `STORY_NAME` = %s and `F_PIN` = %s"
                    tn = round(datetime.datetime.now().timestamp() * 1000)
                    cursor.execute(sql3, (str_pid, tn, val2, fpin))
        sql4 = "SELECT `ID`,`CODE` FROM `CATEGORY` WHERE `CODE` NOT IN (SELECT `STORY_NAME` FROM `POST_STORY` WHERE `F_PIN` = '{}')".format(fpin)
        categories = pd.read_sql(sql4, connection)
        codes = categories['CODE'].to_list()
        ids = categories['ID'].to_list()
        for val1,val2 in zip(codes,ids):
            with connection.cursor() as cursor:
                # Read a single record
                sql5 = "REPLACE INTO `POST_STORY` (`STORY_ID`,`STORY_NAME`,`F_PIN`,`POST_ID`,`STORY_DATE`) VALUES (%s,%s,%s,%s,%s)"
                story_id = fpin + mid_time.next()
                tn = round(datetime.datetime.now().timestamp() * 1000)
                cursor.execute(sql5, (story_id, val1, fpin, '', tn))
            sql6 = "SELECT `POST_ID` FROM `CONTENT_CATEGORY` WHERE `CATEGORY` = {}".format(val2)
            post_ids = pd.read_sql(sql6, connection)['POST_ID'].to_list()
            if post_ids:
                str_pid = ",".join(post_ids)
                with connection.cursor() as cursor:
                    sql7 = "UPDATE `POST_STORY` SET `POST_ID` = %s WHERE `STORY_NAME` = %s"
                    cursor.execute(sql7,(str_pid,val1))
        connection.commit()
    finally:
        connection.close()

