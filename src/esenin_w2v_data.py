import os
import spacy
import re

import pandas as pd

from esenin_w2v_utils import load_data, clean_data

nlp = spacy.load("en_core_web_sm")

directory_path = os.environ["DATA_TEXT_ESENIN"]
data = load_data(directory_path).split('\n')

df = pd.DataFrame(data, columns=["sentence"])

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row['sentence'])).lower() for _j, row in df.iterrows())
data_clean = [clean_data(doc) for doc in nlp.pipe(brief_cleaning)]

df_clean = pd.DataFrame({'sentence_clean': data_clean})
df_clean = df_clean.dropna().drop_duplicates().reset_index(drop=True)

df_clean.to_csv('df_clean.csv')

print(df_clean)
