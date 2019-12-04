import pandas as pd

json = pd.read_json("News_Category_Dataset_v2.json", lines=True)
print(json.columns)
print(json.category.unique())

a = 0
b = 60

c = 0
d = b-a

dat = json[json.category == "FOOD & DRINK"].loc[:, ['headline', 'link']][a:b]
print("""-----------------
FOOD & DRINK
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)
    print(dat.iloc[i, :].link)

dat = json[json.category == "BUSINESS"].loc[:, ['headline', 'link']][a:b]
print("""-----------------
BUSINESS
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)
    print(dat.iloc[i, :].link)

dat = json[json.category == "EDUCATION"].loc[:, ['headline', 'link']][a:b]
print("""-----------------
EDUCATION
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)
    print(dat.iloc[i, :].link)

dat = json[json.category == "WORLD NEWS"].loc[:, ['headline', 'link']][a:b]
print("""-----------------
WORLD NEWS
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)
    print(dat.iloc[i, :].link)

