from newspaper import Article

urls = ["https://towardsdatascience.com/the-shape-of-football-games-1589dc4e652a?gi=b23d8f1a12a0",
"https://www.rd.com/culture/distance-from-earth-to-sun/",
"https://www.theweek.co.uk/football/90293/ronaldo-vs-messi-the-rivalry-statistics-goals-and-awards-ballon-dor",
"https://ymcinema.com/2019/08/09/nvidia-rtx-studio-vs-macbook-pro-the-battle-for-8k-raw-editing/",
"https://news.un.org/en/story/2019/12/1052661"]
urls_2 = ["https://news.un.org/en/story/2019/12/1052621",
"https://www.independent.co.uk/life-style/women/angelina-jolie-interview-woman-own-opinions-attractive-daughters-children-a9039891.html",
"https://www.bloomberg.com/news/articles/2019-12-04/google-founders-give-up-on-being-the-warren-buffett-of-tech",
"https://medium.com/swlh/business-lessons-from-fantasy-football-fe356f264c85",
"https://medium.com/@digitaltonto/are-we-turning-our-backs-on-science-2f97913599e5",
"https://thesefootballtimes.co/2015/05/30/ronaldinho-and-messi-a-relationship-born-out-of-genius/",
"https://www.rd.com/advice/smart-hotels-privacy-concerns/",
"https://www.vice.com/en_asia/article/bjwq5m/people-comparing-philippines-host-sea-games-fyre-festival",
"https://www.chess.com/news/view/mamedyarov-joins-strong-returning-speed-chess-championship-field"]

for url in urls_2:
  article = Article(url)
  article.download()
  article.parse()
  print(url)
  print(article.title)
  print(article.authors)
  print(article.text)
  print("--------------\n")