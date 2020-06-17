import sys
from datetime import datetime, timedelta, time
from urllib.request import Request, urlopen

import dateparser
import requests
from bs4 import BeautifulSoup
# from selenium import webdriver
# import geckodriver_autoinstaller

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}


def fetch_news_list(domain: str, category: str, date_latest: datetime, date_earliest: datetime):
    news_list = []
    if domain == "bisnis.com":
        cid_list = [""]
        if category == "news":
            cid_list = ["186"]
        elif category == "sport":
            cid_list = ["57", "392"]
        for cid in cid_list:
            date_scraped = date_latest
            got_news = True
            while got_news:
                date_string = date_scraped.strftime("%d+%B+%Y")
                # print(date_string)
                got_news_today = True
                page = 1
                while got_news_today:
                    url = "https://www.bisnis.com/index/page/?c=" + cid + "&d=" + date_string + "&per_page=" + str(page)
                    # print(url)
                    req = Request(url, headers=hdr)
                    html = urlopen(req, timeout=30)
                    soup = BeautifulSoup(html, features="lxml")
                    ul_blocks = soup.find("ul", {"class": "l-style-none"})
                    li_blocks = ul_blocks.findAll("li")
                    # print(str(li_blocks))
                    page = page + 1
                    if "Tidak ada berita" in li_blocks[0].text:
                        got_news_today = False
                    else:
                        for li in li_blocks:
                            # print(li)
                            subtitle = li.find("div", {"class": "sub-title"})
                            date_time_str = subtitle.text.split(" WIB")[0]
                            # print(date_time_str)
                            date = datetime.strptime(date_time_str, '%d %b %Y, %H:%M')
                            # print(str(datetime.timestamp(date)))
                            if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                                got_news = False
                                break
                            elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                                a = li.find("a", {"class": "c-222"})
                                link = a.get("href")
                                if link not in news_list:
                                    news_list.append(link)
                                # print(str(datetime.timestamp(date) * 1000))
                    pass
                date_scraped = date_scraped - timedelta(days=1)
                if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                    got_news = False
                pass
    elif domain == "cnbcindonesia.com" and category == "news":
        got_news = True
        date_scraped = date_latest
        while got_news:
            date_string = date_scraped.strftime("%Y/%m/%d")
            got_news_today = True
            page = 1
            while got_news_today:
                url = "https://www.cnbcindonesia.com/news/indeks/3/" + str(page) + "?date=" + date_string
                # print(url)
                req = Request(url, headers=hdr)
                html = urlopen(req, timeout=30)
                soup = BeautifulSoup(html, features="lxml")
                ul_blocks = soup.find("ul", {"class": "terbaru"})
                li_blocks = ul_blocks.findAll("li")
                page = page + 1
                if len(li_blocks) == 0:
                    got_news_today = False
                else:
                    for li in li_blocks:
                        a = li.find("a")
                        link = a.get("href")
                        req_news = Request(link, headers=hdr)
                        html_news = urlopen(req_news, timeout=30)
                        soup_news = BeautifulSoup(html_news, features="lxml")
                        date_box = soup_news.find("div", {"class": "date"})
                        date = datetime.strptime(date_box.text, '%d %B %Y %H:%M')
                        if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                            got_news = False
                            break
                        elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                            if link not in news_list:
                                news_list.append(link)
                        pass
            date_scraped = date_scraped - timedelta(days=1)
            if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                got_news = False
            pass
        pass
    elif domain == "tempo.co":
        cid_list = []
        if category == "news":
            cid_list = ["nasional", "dunia"]
        elif category == "sport":
            cid_list = ["bola", "sport"]
        for cid in cid_list:
            date_scraped = date_latest
            got_news = True
            while got_news:
                date_string = date_scraped.strftime("%Y/%m/%d")
                # print(date_string)
                url = "https://tempo.co/indeks/" + date_string + "/" + cid
                # print(url)
                req = Request(url, headers=hdr)
                html = urlopen(req, timeout=30)
                soup = BeautifulSoup(html, features="lxml")
                div = soup.find("div", {"class": "col w-70"})
                ul_block = div.find("ul")
                li_blocks = ul_block.findAll("li")
                # print(str(li_blocks))
                for li in li_blocks:
                    # print(li)
                    subtitle = li.find("span", {"class": "col"})
                    date_time_str = subtitle.text.split(" WIB")[0]
                    # print(date_time_str)
                    date = dateparser.parse(date_time_str, date_formats=['%d %b %Y %H:%M'], languages=['id'])
                    # print(str(datetime.timestamp(date)))
                    if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                        got_news = False
                        break
                    elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                        a = li.find("a", {"class": "col"})
                        link = a.get("href")
                        if link not in news_list:
                            news_list.append(link)
                        # print(str(datetime.timestamp(date) * 1000))
                # print(len(news))
                date_scraped = date_scraped - timedelta(days=1)
                if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                    got_news = False
                pass
    elif domain == "cnnindonesia.com":
        cid_list = []
        if category == "news":
            cid_list = [("3", "nasional"),("6", "internasional")]
        elif category == "sport":
            cid_list = [("7", "olahraga")]
        for cid in cid_list:
            date_scraped = date_latest
            got_news = True
            while got_news:
                date_string = date_scraped.strftime("%Y/%m/%d")
                # print(date_string)
                got_news_today = True
                page = 1
                while got_news_today:
                    url = "https://www.cnnindonesia.com/" + cid[1] + "/indeks/" + cid[
                        0] + "?date=" + date_string + "&kanal=" + cid[0] + "&p=" + str(page)
                    # print(url)
                    req = Request(url, headers=hdr)
                    html = urlopen(req, timeout=30)
                    soup = BeautifulSoup(html, features="lxml")
                    div_block = soup.find("div", {"class": "media_rows"})
                    article_blocks = div_block.findAll("article")
                    # print(str(li_blocks))
                    page = page + 1
                    if len(article_blocks) == 0:
                        got_news_today = False
                    else:
                        for li in article_blocks:
                            a = li.find("a")
                            link = a.get("href")
                            req_news = Request(link, headers=hdr)
                            html_news = urlopen(req_news)
                            soup_news = BeautifulSoup(html_news, features="lxml")
                            date_box = soup_news.find("div", {"class": "date"})
                            date_time_str = date_box.text.split(" | ")[1].strip()
                            date = dateparser.parse(date_time_str, date_formats=['%A, %d/%m/%Y %H:%M WIB'],
                                                    languages=['id'])
                            if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                                got_news = False
                                break
                            elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                                if link not in news_list:
                                    news_list.append(link)
                            pass
                    pass
                date_scraped = date_scraped - timedelta(days=1)
                if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                    got_news = False
                pass
    elif domain == "kompas.com":
        cid_list = []
        if category == "news":
            cid_list = ["news"]
        elif category == "sport":
            cid_list = ["sport","bola"]
        for cid in cid_list:
            date_scraped = date_latest
            got_news = True
            while got_news:
                date_string = date_scraped.strftime("%Y-%m-%d")
                # print(date_string)
                got_news_today = True
                page = 1
                while got_news_today:
                    url = "https://indeks.kompas.com/?site=" + cid + "&date=" + date_string + "&page=" + str(page)
                    # print(url)
                    req = Request(url, headers=hdr)
                    html = urlopen(req, timeout=30)
                    soup = BeautifulSoup(html, features="lxml")
                    div_block = soup.find("div", {"class": "latest--indeks"})
                    article__list_blocks = div_block.findAll("div", {"class": "article__list"})
                    # print(str(li_blocks))
                    page = page + 1
                    if len(article__list_blocks) == 0:
                        got_news_today = False
                    else:
                        for li in article__list_blocks:
                            # print(li)
                            subtitle = li.find("div", {"class": "article__date"})
                            date_time_str = subtitle.text.split(" WIB")[0]
                            # print(date_time_str)
                            date = dateparser.parse(date_time_str, date_formats=['%d/%m/%Y, %H:%M'], languages=['id'])
                            # print(str(datetime.timestamp(date)))
                            if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                                got_news = False
                                break
                            elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                                a = li.find("a", {"class": "article__link"})
                                link = a.get("href")
                                if link not in news_list:
                                    news_list.append(link)
                                # print(str(datetime.timestamp(date) * 1000))
                date_scraped = date_scraped - timedelta(days=1)
                if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                    got_news = False
                pass
    elif domain == "okezone.com":
        cid_list = []
        if category == "news":
            cid_list = ["news"]
        elif category == "sport":
            cid_list = ["sports","bola"]
        for cid in cid_list:
            date_scraped = date_latest
            got_news = True
            while got_news:
                date_string = date_scraped.strftime("%Y/%m/%d")
                # print(date_string)
                got_news_today = True
                page = 0
                while got_news_today:
                    url = "https://" + cid + ".okezone.com/indeks/" + date_string + "/" + str(page * 10)
                    req = Request(url, headers=hdr)
                    html = urlopen(req, timeout=30)
                    soup = BeautifulSoup(html, features="lxml")
                    ul_block = soup.find("ul", {"class": "list-berita"})
                    li_blocks = ul_block.findAll("li")
                    page = page + 1
                    if len(li_blocks) == 0:
                        got_news_today = False
                    else:
                        for li in li_blocks:
                            subtitle = li.find("time")
                            date_time_str = subtitle.text.splitlines()[3].split(" WIB")[0].split(" ", 1)[1]
                            date = dateparser.parse(date_time_str, date_formats=['%d %B %Y %H:%M'], languages=['id'])
                            if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                                got_news = False
                                break
                            elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                                a = li.find("a")
                                link = a.get("href")
                                if link not in news_list:
                                    news_list.append(link)
                date_scraped = date_scraped - timedelta(days=1)
                if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                    got_news = False
                pass
    elif domain == "technologyreview.com":
        url = "https://www.technologyreview.com"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=60)
        soup = BeautifulSoup(html, features="lxml")
        li_blocks = soup.findAll("div", {"class": "feedUnitWrapper"})
        for li in li_blocks:
            time_el = li.find("time",{"class": "timestamp"})
            try:
                date_time_str = time_el.attrs.get('datetime')
                date = dateparser.parse(date_time_str)
                if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                    break
                elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                    a = li.find("a", {"class": "headLink"})
                    link = url + a.get("href")
                    if link not in news_list:
                        news_list.append(link)
            except TypeError:
                pass
    elif domain == "arstechnica.com" and category == "technology":
        date_scraped = date_latest
        got_news = True
        page = 0
        while got_news:
            url = "https://arstechnica.com/information-technology/page/" + str(page)
            print(url)
            req = Request(url, headers=hdr)
            html = urlopen(req, timeout=60)
            soup = BeautifulSoup(html, features="lxml")
            ul_block = soup.find("div", {"class": "listing-latest"}).find("ol")
            li_blocks = ul_block.findAll("li", {"class": "article"})
            page = page + 1
            if len(li_blocks) == 0:
                got_news = False
            else:
                for li in li_blocks:
                    time_el = li.find("time",{"class": "date"})
                    date_time_str = time_el.attrs.get('datetime')
                    date = dateparser.parse(date_time_str)
                    if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                        got_news = False
                        break
                    elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                        a = li.find("a", {"class": "overlay"})
                        link = a.get("href")
                        if link not in news_list:
                            news_list.append(link)
            date_scraped = date_scraped - timedelta(days=1)
            if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                got_news = False
            pass
    elif domain == "techcrunch.com" and category == "technology":
        url = "https://techcrunch.com/"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=60)
        soup = BeautifulSoup(html, features="lxml")
        ul_block = soup.find("div", {"class": "river--homepage"})
        li_blocks = ul_block.findAll("div", {"class": "post-block"})
        print(len(li_blocks))
        for li in li_blocks:
            time_el = li.find("time",{"class": "river-byline__time"})
            try:
                date_time_str = time_el.attrs.get('datetime')
                date = dateparser.parse(date_time_str)
                if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                    break
                elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                    a = li.find("a", {"class": "post-block__title__link"})
                    link = a.get("href")
                    if link not in news_list:
                        news_list.append(link)
            except TypeError:
                pass
    elif domain == "ai-magazine.com" and category == "technology":
        url = "https://ai-magazine.com/~api/papers/d3d6f972-fd0b-4bcd-a97c-53de828ee59e/articles"
        resp = requests.get(url=url)
        resp_json = resp.json()
        idx = 0
        total = resp_json['total']
        data = resp_json['data']
        for entry in data:
            link = entry['url']
            if link not in news_list:
                news_list.append(link)
        idx = idx + 10
        while idx < total:
            url = "https://ai-magazine.com/~api/papers/d3d6f972-fd0b-4bcd-a97c-53de828ee59e/articles?index="+str(idx)
            resp = requests.get(url=url)
            resp_json = resp.json()
            idx = idx + 10
            data = resp_json['data']
            for entry in data:
                link = entry['url']
                if link not in news_list:
                    news_list.append(link)
    elif domain == "artificialintelligence-news.com" and category == "technology":
        date_scraped = date_latest
        got_news = True
        page = 1
        while got_news:
            url = "https://www.artificialintelligence-news.com/page/" + str(page)
            req = Request(url, headers=hdr)
            html = urlopen(req, timeout=60)
            soup = BeautifulSoup(html, features="lxml")
            ul_block = soup.find("ul", {"class": "infinite-content"})
            li_blocks = ul_block.findAll("li", {"class": "infinite-post"})
            page = page + 1
            if len(li_blocks) == 0:
                got_news = False
            else:
                for li in li_blocks:
                    a = li.find("a")
                    link = a.get("href")
                    req_news = Request(link, headers=hdr)
                    html_news = urlopen(req_news)
                    soup_news = BeautifulSoup(html_news, features="lxml")
                    article = soup_news.find("article", {"id":"post-area"})
                    date_box = article.find("time", {"class": "post-date"})
                    date_time_str = date_box.attrs.get("datetime")
                    date = dateparser.parse(date_time_str)
                    if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                        got_news = False
                        break
                    elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                        if link not in news_list:
                            news_list.append(link)
                    pass
            date_scraped = date_scraped - timedelta(days=1)
            if datetime.timestamp(date_scraped) < datetime.timestamp(date_earliest):
                got_news = False
            pass
        pass
    elif domain == "coindesk.com":
        base = "https://www.coindesk.com"
        url = "https://www.coindesk.com/news"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=60)
        soup = BeautifulSoup(html, features="lxml")
        head_news = soup.find("section", {"class": "featured-hub-content v3up"})
        article_cards = head_news.find_all("section",{"class": "article-card-fh"})
        for card in article_cards:
            a = card.find("a")
            link = base+a.get("href")
            req = Request(link, headers=hdr)
            html = urlopen(req, timeout=60)
            html_soup = BeautifulSoup(html, features="lxml")
            article_datetime = html_soup.find("div", {"class": "article-hero-datetime"})
            dtn = article_datetime.find("time").attrs.get("datetime")
            date = dateparser.parse(dtn)
            if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                pass
            elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                if link not in news_list:
                    news_list.append(link)
            pass
        pass
        story_stack = soup.find("div", {"class": "story-stack"})
        item_wrappers = story_stack.find_all("div", {"class": "list-item-wrapper"})
        for wrapper in item_wrappers:
            a = wrapper.find("a",{"class":None})
            link = base+a.get("href")
            req = Request(link, headers=hdr)
            html = urlopen(req, timeout=60)
            html_soup = BeautifulSoup(html, features="lxml")
            article_datetime = html_soup.find("div", {"class": "article-hero-datetime"})
            dtn = article_datetime.find("time").attrs.get("datetime")
            date = dateparser.parse(dtn)
            date = date + timedelta(hours=7)
            if datetime.timestamp(date) < datetime.timestamp(date_earliest):
                pass
            elif datetime.timestamp(date) < datetime.timestamp(date_latest):
                if link not in news_list:
                    news_list.append(link)
            pass
        pass
    elif domain == "artnews.com":
        url = "https://www.artnews.com/c/art-news/news/"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=30)
        soup = BeautifulSoup(html, features="lxml")
        article_block = soup.findAll("article")
        for x in article_block:
            apart = x.find("a")
            article_link = apart.get("href")
            not_news = x.find("a", {"class": "lrv-a-unstyle-link lrv-u-display-block"})
            time = x.find("time", {"class": "c-timestamp"})
            if not_news is None and time is not None:
                time_date = str(time.text)
                if time_date is not None:
                    date_time = dateparser.parse(time_date)
                    dt_millis = date_time.timestamp() * 1000
                    print(dt_millis)
                    dt_now = datetime.now().timestamp() * 1000
                    if dt_now - dt_millis <= 86400000:
                        news_list.append(article_link)
                pass
            pass
        pass
    elif domain == "finextra.com":
        base = "https://www.finextra.com"
        url = "https://www.finextra.com/latest-news/startups"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=60)
        soup = BeautifulSoup(html, features="lxml")
        storylist = soup.find("div",{"class": "modulegroup--latest-storylisting"})
        storylist = storylist.find_all("div", {"class":"module--story"})
        for story in storylist:
            a = story.find("a")
            link = base + a.get("href")
            article_date = story.find("span",{"class":"news-date"}).text
            date = dateparser.parse(article_date)
            dt_e = date_earliest.replace(hour=0,minute=0,second=0)
            dt_l = date_latest.replace(hour=0,minute=0,second=0)
            if datetime.timestamp(date) < datetime.timestamp(dt_e):
                pass
            elif datetime.timestamp(date) <= datetime.timestamp(dt_l):
                if link not in news_list:
                    news_list.append(link)
            pass
        pass
    elif domain == "reuters.com":
        url = "https://www.reuters.com/finance"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=30)
        soup = BeautifulSoup(html, features="lxml")
        article_block = soup.findAll("a")
        time_block = soup.findAll("time",{"class":"article-time"})
        n = 0
        for x in article_block:
            title = x.find("h3", {"class": "story-title"})
            link = x.get("href")
            if title is not None and link is not None:
                time = time_block[n].find("span", {"class": "timestamp"}).string
                n = n + 1
                time_news = dateparser.parse(time) - timedelta(days=1)
                time_stamp = time_news.timestamp() * 1000
                time_now = datetime.now().timestamp() * 1000
                if time_now - (time_stamp + 39600000) <= 86400000:
                    link = "https://reuters.com{}".format(link.strip())
                    news_list.append(link)
            pass
        pass
    elif domain == "marketwatch.com":
        url = "https://www.marketwatch.com/markets?mod=top_nav"
        req = Request(url, headers=hdr)
        html = urlopen(req, timeout=30)
        soup = BeautifulSoup(html, features="lxml")
        article_block = soup.findAll("h3", {"class": "article__headline"})
        article_detail = soup.findAll("div", {"class": "article__details"})
        for x, z in zip(article_block, article_detail):
            title = x.find("a").text.strip()
            link = x.find("a").get("href")
            date = z.find("span", {"class": "article__timestamp"}).text.strip()
            datereal = dateparser.parse(date)
            datemillis = datetime.timestamp(datereal)
            datemillis = datemillis * 1000
            datenow = datetime.now().timestamp() * 1000
            if datenow - (datemillis + 43200000) <= 86400000:
                news_list.append(link)
            pass
        pass
    # elif domain == "theblockcrypto.com":
    #     base = "https://www.theblockcrypto.com"
    #     geckodriver_autoinstaller.install()
    #     try:
    #         browser = webdriver.Firefox()
    #         browser.get(base)
    #         html = browser.page_source
    #         soup = BeautifulSoup(html, features="lxml")
    #         story_feed = soup.find("div", {'class': 'storyFeed'})
    #         articles = story_feed.find_all("article", {'class': 'border-white'})
    #         for article in articles:
    #             a = article.find("a")
    #             link = base+a.get("href")
    #             dt = article.find("h5", {'class': 'font-meta'}).text
    #             date = dateparser.parse(dt)
    #             if datetime.timestamp(date) < datetime.timestamp(date_earliest):
    #                 break
    #             elif datetime.timestamp(date) < datetime.timestamp(date_latest):
    #                 if link not in news_list:
    #                     news_list.append(link)
    #             pass
    #         pass
    #     finally:
    #         browser.close()

    # else:
    #     paper = newspaper.build('https://'+domain)
    #     for article in paper.articles:
    #         news_list.append(article)

        # mit_paper = newspaper.build('https://www.technologyreview.com/')
        # for article in mit_paper.articles:
        #     link = article.url
        #     print(link)
        #     req_news = Request(link, headers=hdr)
        #     html_news = urlopen(req_news)
        #     soup_news = BeautifulSoup(html_news, features="lxml")
        #     time_el = soup_news.find("time",{"class":"timestamp"})
        #     date_time_str = time_el.attrs.get('datetime')
        #     date = dateparser.parse(date_time_str)
        #     print(str(date))
        #     if datetime.timestamp(date) < datetime.timestamp(date_earliest):
        #         got_news = False
        #         break
        #     elif datetime.timestamp(date) < datetime.timestamp(date_latest):
        #         if link not in news_list:
        #             news_list.append(link)
        #     pass
        # pass
    return news_list


# print(len(news))
# print(str(news))

# crawl the sites

def main():
    domain = "finextra.com"
    category = "news"
    date_start = datetime.now()
    date_end = date_start - timedelta(hours=1)
    latest = 0
    earliest = 0
    latest_hour = 4
    earliest_hour = 5

    for i, s in enumerate(sys.argv[1:]):
        if s[:2] == '--':
            arg = s[2:]
            if arg == 'domain':
                domain = sys.argv[i + 2]
            if arg == 'category':
                category = sys.argv[i + 2]
            if arg == 'latest':
                try:
                    latest = float(sys.argv[i + 2])
                except ValueError:
                    latest = date_start.timestamp() * 1000.0
            if arg == 'earliest':
                try:
                    earliest = float(sys.argv[i + 2])
                except ValueError:
                    earliest = date_end.timestamp() * 1000.0
            if arg == 'latest-hour':
                try:
                    latest_hour = int(sys.argv[i + 2])
                except ValueError:
                    latest_hour = date_start.hour
            if arg == 'earliest-hour':
                try:
                    earliest_hour = int(sys.argv[i + 2])
                except ValueError:
                    earliest_hour = date_end.hour

    if latest_hour > -1:
        latest_time = datetime.combine(date_start, time(latest_hour))
        if latest_time.timestamp() > date_start.timestamp():
            latest_time = latest_time - timedelta(days=1)
        if earliest_hour < 0:
            if latest_hour == 0:
                earliest_hour = 23
            else:
                earliest_hour = latest_hour - 1
        earliest_time = datetime.combine(latest_time, time(earliest_hour))
        if earliest_time.timestamp() >= latest_time.timestamp():
            earliest_time = earliest_time - timedelta(days=1)
        date_start = latest_time
        date_end = earliest_time
    else:
        if latest > earliest:
            date_start = datetime.fromtimestamp(latest / 1000.0)
            date_end = datetime.fromtimestamp(earliest / 1000.0)
        else:
            date_end = datetime.fromtimestamp(latest / 1000.0)
            date_start = datetime.fromtimestamp(earliest / 1000.0)

    news = fetch_news_list(domain, category, date_start, date_end)
    for link in news:
        print("Fetched: ")
        print(link)
        # text_block, title, image_src = crawler.crawl_article(link)
        # summary = textsummarization.summarize_text(text_block.replace("\n", " "), 1.0, 1000)
        # print(title)
        # print(image_src)
        # print(summary)
        # print("[----]")


# for i, s in enumerate(sys.argv[1:]):
#     if s[:2] == '--':
#         arg = s[2:]
#         if arg == 'link':
#             url = sys.argv[i + 2]
#         if arg == 'name':
#             alias = sys.argv[i + 2]

if __name__ == "__main__":
    main()
