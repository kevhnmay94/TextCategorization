import pandas as pd

crp = pd.read_csv('dataset-all-27.csv')
crp.loc[crp['category'] == 'food','category'] = 'Makanan'
crp.loc[crp['category'] == 'business','category'] = 'Bisnis'
crp.loc[crp['category'] == 'education','category'] = 'Edukasi'
crp.loc[crp['category'] == 'news','category'] = 'Berita'
crp.loc[crp['category'] == 'celebrity','category'] = 'Selebriti'
crp.loc[crp['category'] == 'jobs','category'] = 'Lowongan Pekerjaan'
crp.loc[crp['category'] == 'technology','category'] = 'Teknologi'
crp.loc[crp['category'] == 'science','category'] = 'Sains'
crp.loc[crp['category'] == 'sports','category'] = 'Olahraga'
crp.loc[crp['category'] == 'travel','category'] = 'Travel'
crp.loc[crp['category'] == 'lifestyle','category'] = 'Gaya Hidup'
crp.loc[crp['category'] == 'entertainment','category'] = 'Hiburan'

crp.to_csv('dataset-all-27-translated.csv',index=False)