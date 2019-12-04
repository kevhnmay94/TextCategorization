import pandas as pd

csv = pd.read_csv("dataset.csv")
print(csv.columns)
print(csv.category.unique())

a = 0
b = 50

c = 0
d = b-a

dat = csv[csv.category == "FOOD"].loc[:, ['headline']][a:b]
print("""-----------------
FOOD & DRINK
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)

dat = csv[csv.category == "BUSINESS"].loc[:, ['headline']][a:b]
print("""-----------------
BUSINESS
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)

dat = csv[csv.category == "EDUCATION"].loc[:, ['headline']][a:b]
print("""-----------------
EDUCATION
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)

dat = csv[csv.category == "WORLD NEWS"].loc[:, ['headline']][a:b]
print("""-----------------
WORLD NEWS
-----------------""")
for i in range(c, d):
    print(dat.iloc[i, :].headline)