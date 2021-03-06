import os
import re
import sys
import urllib
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.request import urlretrieve

from bs4 import BeautifulSoup, PageElement
from newspaper import Article, Config

import textsummarization_baru

def findClosest(arr, n, target):
    # Corner cases
    if (target <= arr[0]['size']):
        return arr[0]['size']
    if (target >= arr[n - 1]['size']):
        return arr[n - 1]['size']

        # Doing binary search
    i = 0
    j = n
    mid = 0
    while (i < j):
        mid = round((i + j) / 2)

        if (arr[mid]['size'] == target):
            return arr[mid]['size']

            # If target is less than array
        # element, then search in left
        if (target < arr[mid]['size']):

            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1]['size']):
                return getClosest(arr[mid - 1]['size'], arr[mid]['size'], target)

                # Repeat for left half
            j = mid

            # If target is greater than mid
        else:
            if (mid < n - 1 and target < arr[mid + 1]['size']):
                return getClosest(arr[mid]['size'], arr[mid + 1]['size'], target)

                # update i
            i = mid + 1

    # Only single element left after search
    return arr[mid]['size']


# Method to compare which one is the more close.
# We find the closest by taking the difference
# between the target and both values. It assumes
# that val2 is greater than val1 and target lies
# between these two.
def getClosest(val1, val2, target):
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1

def extract_p_tags_text(p_blocks: PageElement):
    textblock = ""
    paragraphs = p_blocks.find_all("p")
    for para in paragraphs:
        textblock += para.getText().strip()
        textblock += "\n"
        pass
    return textblock


def extract_list_text(ul_block: PageElement):
    textblock = ""
    lis = ul_block.find_all("li")
    for li in lis:
        try:
            if li.id == "replacementPartsFitmentBullet":
                pass
            textblock += li.getText()
            textblock += "\n"
            pass
        except Exception:
            # print(ex)
            pass
    return textblock

def getsize(url):
    file = urllib.request.urlopen(url)
    size = file.headers.get("content-length")
    file.close()
    if size is None:
        size = len(file.read())
    return size



def download_article(url):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    # user_agent = 'Mozilla/5.0 (X11; Linux x86_64)'
    newspaper_config = Config()
    newspaper_config.browser_user_agent = user_agent
    article = Article(url,config = newspaper_config)
    article.download()
    article.parse()
    image_arr = []
    size = len(article.images)
    article_text = ""
    # if size > 1:
    #     for image in article.images:
    #         # print("images : {}".format(image))
    #         extension = os.path.splitext(os.path.basename(image.split("?")[0]))[-1]
    #         # print("extension : {}".format(extension))
    #         if (extension.casefold() == ".jpg".casefold() or extension.casefold() == ".jpeg".casefold()) and extension:
    #             img_size = getsize(image)
    #             image_arr.append({'url':image,'size':int(img_size)})
    # image_arr = sorted(image_arr, key=lambda k: k['size'])
    # # print("size image : {}".format(len(image_arr)))
    # valMean = 0
    # for x in image_arr:
    #     if x['url'] == article.top_image:
    #         valMean = x['size']
    # #         print("top image : {}".format(article.top_image))
    # #         print("url : {}".format(x['url']))
    # # print("mean : {}".format(valMean))
    # # print("image arr : {}".format(image_arr))
    # closestOne = findClosest(image_arr,len(image_arr),valMean)
    # # print("closest one : {}".format(closestOne))
    # closestOneURL = ""
    # valIndex = 0
    # for x in image_arr:
    #     # print("x : {}".format(x))
    #     # print(x['url'])
    #     # print(x['size'])
    #     # print(type(x['url']))
    #     # print(type(x['size']))
    #     # print(type(closestOne))
    #     if int(x['size']) == int(closestOne):
    #         closestOneURL = x['url']
    #         valIndex = image_arr.index(x)
    # # print("index : {} {}".format(valIndex,closestOneURL))
    #
    # image_temp = []
    # for a in image_arr:
    #     if a['url'] != closestOneURL:
    #         # print("A : {}".format(a))
    #         image_temp.append(a)
    #
    #
    # closestTwoURL = ""
    # if len(image_arr) > 2:
    #     closestTwo = findClosest(image_temp,len(image_temp),valMean)
    #     for x in image_arr:
    #         if x['size'] == closestTwo and x['url'] != closestOneURL:
    #             closestTwoURL = x['url']
    #             # print("closest two : {} {}".format(closestTwo, closestTwoURL))
    #
    image_choice = []
    # if size > 2:
    #     image_choice.append(article.top_image)
    #     image_choice.append(closestOneURL)
    #     image_choice.append(closestTwoURL)
    # elif size == 2:
    #     image_choice.append(article.top_image)
    #     image_choice.append(closestOneURL)
    # else:
    #     image_choice.append(article.top_image)
    if article.text is not None and len(article.text) > 0:
        article_text = article.text
    req = Request(url, headers=hdr)
    html = urlopen(req)
    soup = BeautifulSoup(html, features="lxml")
    tag = soup.findAll("meta")
    image_url = ""
    for og in tag:
        if og.get("property",None) == "og:image":
            image_url = og.get("content",None)
    if len(image_url) == 0:
        image_url = article.top_image

    if len(image_url) > 0:
        image_choice.append(image_url)

    return article_text, article.title, image_choice


hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
basePathTxt = ""
basePathSum = ""
basePathImg = ""
summary_max_ratio = 1.0
summary_max_char = 512

text_block = ""
image_src = []
title = "[Cannot fetch the title]"
summary = "[Cannot summarize]"


def crawl_article(crawl_url: str):
    parsed_uri = urlparse(crawl_url)
    domain = '{uri.netloc}'.format(uri=parsed_uri)
    # print(domain)
    req = Request(crawl_url, headers=hdr)
    html = urlopen(req)
    soup = BeautifulSoup(html, features="lxml")
    # print(str(soup))
    # exit()
    text_block_1 = ""
    title_1 = ""
    image_link_1 = ""
    if domain.endswith('bsn.go.id'):
        try:
            p_blocks = soup.find("div", {"class": "block detail"})
            text_block_1 = extract_p_tags_text(p_blocks)

            title_1 = p_blocks.find("div", {"class": "meta"}).find("h1").getText()

            image_p = p_blocks.findAll("p")[2]
            image = image_p.find_next("img")
            image_link_1 = str(image.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('bisnis.com'):
        try:
            col_custom_left = soup.find("div", {"class": "col-custom left"})
            title_1 = col_custom_left.find("h1").getText()
            p_blocks = col_custom_left.find("div", {"class": "col-sm-10"})
            p_blocks.find("div", {"class": "topik"}).decompose()
            text_block_1 = extract_p_tags_text(p_blocks)
            img_block = col_custom_left.find("div", {"class": "main-image"}).find("img")
            image_link_1 = str(img_block.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('gatra.com'):
        try:
            section = soup.find("section", {"id": "content"})
            inside_box = section.findAll("div", {"class": "card-body inside-box"})
            title_1 = inside_box[0].find("div", {"class": "font-weight-bold"}).getText()

            text_element = section.find("div", {"class": "row"}).findAll("div", {"class": "row mrgn-bot"})[0].find(
                "div", {"class": "col-md-9"})
            for span in text_element.findAll("span"):
                span.decompose()
            for a_block in text_element.findAll("a"):
                a_block.decompose()
            text_block_1 = extract_p_tags_text(text_element)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('cnbcindonesia.com'):
        try:
            title_1 = soup.find("h1").getText()

            text_element = soup.find("div", {"class": "detail_text"})
            for center in text_element.findAll("center"):
                center.decompose()
            for table in text_element.findAll("table"):
                table.decompose()
            text_block_1 = str(text_element.text)

            img_block = soup.find("div", {"class": "media_artikel"}).find("img")
            image_link_1 = str(img_block.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('tempo.co'):
        try:
            article = soup.find("article")
            title_1 = article.find("h1", {"itemprop": "headline"}).getText().strip()
            p_blocks = article.find("div", {"itemprop": "articleBody"})
            text_block_1 = extract_p_tags_text(p_blocks)
            img_block = article.find("img", {"itemprop": "image"})
            image_link_1 = str(img_block.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('cnnindonesia.com'):
        try:
            section = soup.find("section", {"id": "content"})
            title_1 = section.find("h1", {"class": "title"}).getText().strip()
            text_element = section.find("div", {"id": "detikdetailtext"})
            for br in text_element.findAll("br", recursive=False):
                br.replaceWith("\n")
            for span in text_element.findAll("span", recursive=False):
                span.replaceWith(span.getText())
            for a_block in text_element.findAll("a"):
                a_block.decompose()
            for center in text_element.findAll("center"):
                center.decompose()
            for table in text_element.findAll("table"):
                table.decompose()
            for style in text_element.findAll("style"):
                style.decompose()
            for script in text_element.findAll("script"):
                script.decompose()
            text_block_1 = re.sub("\s\s+", " ", str(text_element.text.strip()))
            img_block = soup.find("div", {"class": "media_artikel"}).find("img")
            image_link_1 = str(img_block.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('kompas.com'):
        try:
            title_1 = soup.find("h1", {"class": "read__title"}).getText()

            text_element = soup.find("div", {"class": "read__content"})
            for span in text_element.findAll("span"):
                span.decompose()
            for a_block in text_element.findAll("a"):
                a_block.unwrap()
            text_block_1 = extract_p_tags_text(text_element)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
            text_block_1 = "\n".join(text_block_1.split("\n")[4:])
    elif domain.endswith('kontan.co.id'):
        text_block_1, title_1, image_link_1 = download_article(crawl_url)
        text_block_1 = "\n".join(text_block_1.split("\n")[4:])
    elif domain.endswith('okezone.com'):
        try:
            title_1 = soup.find("dev", {"class": "title"}).getText()

            text_element = soup.find("div", {"id": "contentx"})
            text_block_1 = extract_p_tags_text(text_element)

            img_block = soup.find("img", {"id": "imgCheck"})
            image_link_1 = str(img_block.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('coindesk.com'):
        try:
            title_block = soup.find("div", {"class": "article-hero-title"})
            title_1 = title_block.find("h1", {"class": "heading"}).getText()

            text_element = soup.find("section", {"class": "has-media news article-body"})
            text_block_1 = extract_p_tags_text(text_element)

            img_block = soup.find("div", {"class": "article-hero-media"})
            image_link_1 = img_block.find("img")
            image_link_1 = str(image_link_1.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('finextra.com'):
        try:
            base = "https://www.finextra.com"
            title_block = soup.find("div", {"class": "article--title"})
            title_1 = title_block.find("h1").text

            text_element = soup.find("div",{"id": "ctl00_ctl00_body_mainContent_NewsActicle_pnlBody"})
            text_block_1 = text_element.text

            img_block = soup.find("div", {"class": "article--image"})
            image_link_1 = img_block.find("img")
            image_link_1 = base + str(image_link_1.get("src"))
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('amazon.com'):
        try:
            bullet_blocks = soup.find("div", {"id": "feature-bullets"})
            try:
                text_block_1 = extract_list_text(bullet_blocks)
            except Exception:
                text_block_1 = ""

            # print(text_block)
            titleSpan = soup.find("span", {"id": "productTitle"})
            # print("titleSpan "+str(titleSpan))
            title_1 = titleSpan.find("span", {"id": "productTitle"}).getText()
            # print(title)
            # result = amazonscraper.search(title,max_product_nb=1)[0]
            # image_src = result.img
            # print(image_src)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('technologyreview.com'):
        try:
            if "/s/" in crawl_url:
                title_1 = soup.find("span", {"class": "storyTitle"}).getText()

                text_element = soup.find("div", {"class": "storyContent"})
                for aside in text_element.findAll("aside"):
                    aside.decompose()
                for div in text_element.findAll("div"):
                    div.unwrap()
                text_block_1 = extract_p_tags_text(text_element)

                img_block = soup.find("div", {"class": "mediaContainer"}).find("img")
                image_link_1 = str(img_block.get("src"))
            elif "/f/" in crawl_url:
                title_1 = soup.find("span", {"class": "headLink"}).getText()

                text_element = soup.findAll("article", {"class": "storyTease--expandable"})[0].find("div",{"class": "expandableBody"})
                for aside in text_element.findAll("aside"):
                    aside.decompose()
                for div in text_element.findAll("div"):
                    div.unwrap()
                text_block_1 = extract_p_tags_text(text_element)

                img_block = soup.find("div", {"class": "mediaContainer"}).find("img")
                image_link_1 = str(img_block.get("src"))
            else:
                text_block_1, title_1, image_link_1 = download_article(crawl_url)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('weforum.org'):
        try:
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
            title_1 = soup.find("h1",{"class":"article__headline"}).string
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('artnews.com'):
        try:
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('reuters.com'):
        try:
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    elif domain.endswith('marketwatch.com'):
        try:
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
        except (Exception,HTTPError):
            text_block_1, title_1, image_link_1 = download_article(crawl_url)
    else:
        text_block_1, title_1, image_link_1 = download_article(crawl_url)
    return text_block_1, title_1, image_link_1


def main():
    url = ""
    alias = ""

    for i, s in enumerate(sys.argv[1:]):
        if s[:2] == '--':
            arg = s[2:]
            if arg == 'link':
                url = sys.argv[i + 2]
            if arg == 'name':
                alias = sys.argv[i + 2]

    if not url:
        print("[Error] Link can't be empty")
        exit()

    text_block, title, image_src = crawl_article(url)

    parsed_uri = urlparse(url)
    domain = '{uri.netloc}'.format(uri=parsed_uri)

    summary = ""
    if text_block:
        txt_filename = ""
        if alias:
            txt_filename = basePathTxt + alias + ".txt"
        else:
            txt_filename = basePathTxt + domain + " - " + title.replace("/", "_") + ".txt"
        file = open(txt_filename, "w+",encoding="utf-8")
        file.write(text_block)
        file.close()

        sum_filename = ""
        if alias:
            sum_filename = basePathSum + alias + ".txt"
        else:
            sum_filename = basePathSum + domain + " - " + title.replace("/", "_") + ".txt"
        summary = textsummarization_baru.summarize_text(text_block, summary_max_ratio, summary_max_char)
        if not summary.startswith("["):
            fileSum = open(sum_filename, "w+",encoding="utf-8")
            fileSum.write(summary)
            fileSum.close()

    img_filename = "[]"
    image_total = ""
    if image_src:
        n = 0
        # print("Type : {}".format(type(image_src)))
        if type(image_src) == str:
            isUnicode = False
            if alias:
                img_filename = alias + os.path.splitext(os.path.basename(image_src.split("?")[0]))[-1]
            else:
                img_filename = domain + " - " + title + "-" + str(n) + "-" + \
                               os.path.splitext(os.path.basename(image_src.split("?")[0]))[-1]
            img_filename = img_filename.replace("/", " ").replace(":", " ")
            img_filename = re.sub(r"\s+", "_", img_filename)
            full_filename = os.path.join(basePathImg, img_filename)
            full_filename = full_filename.replace("?", "").replace("<", "").replace(">", "")
            for x in image_src:
                if ord(x) > 127:
                    isUnicode = True
            if isUnicode:
                image_src = urllib.parse.quote(image_src, safe=":/")
            urlretrieve(image_src, full_filename)
            image_total = img_filename
        elif type(image_src) == list:
            for image in image_src:
                isUnicode = False
                # print("image : {}".format(image))
                if alias:
                    img_filename = alias + os.path.splitext(os.path.basename(image.split("?")[0]))[-1]
                else:
                    img_filename = domain + " - " + title + "-" + str(n) + "-" + os.path.splitext(os.path.basename(image.split("?")[0]))[-1]
                img_filename = img_filename.replace("/", " ").replace(":", " ")
                img_filename = re.sub(r"\s+", "_", img_filename)
                full_filename = os.path.join(basePathImg, img_filename)
                full_filename = full_filename.replace("?","").replace("<","").replace(">","")
                for x in image:
                    if ord(x) > 127:
                        isUnicode = True
                if isUnicode:
                    image = urllib.parse.quote(image,safe=":/")
                urlretrieve(image, full_filename)
                image_total = image_total + img_filename + "|"
                n = n + 1

    if not title:
        title = "[Cannot fetch the title]"
    title = textsummarization_baru.translate(title)
    print(title)
    print(image_total)
    # print(len(image_src))
    print(summary)
    # print(text_block)


if __name__ == "__main__":
    main()
