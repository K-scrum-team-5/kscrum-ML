import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings(action='ignore')


# ---------------- 영화 데이터 불러오기 ----------------
movie=pd.read_csv("./csv/movie.csv")
# print(movie.head())


# ---------------- genres 컬럼 장르 정보만 남기기 ----------------
unique_genres = movie['genres'].unique()

# print(movie['genres'][0])

# ---------------- 문자열이 아닌 리스트 형태로 다시 바꿔주는 작업 ----------------

from ast import literal_eval

movie["genres"] = movie['genres'].apply(literal_eval)

# print(type(movie['genres'][0]))

# ---------------- 장르값만 남기는 함수 ----------------
def get_genres(x):
    try:
        genre=[i['name'] for i in x]
        return genre

    except:
        genre=[]
        return genre

movie["genres"]=movie['genres'].apply(get_genres)
# print(movie.head())

# ---------------- 문자 공백 없애기 ----------------
# print(movie['genres'][21:30])

movie["genres"]=movie["genres"].apply(lambda x: [str(i).replace(" ","") for i in x])
# print(movie['genres'][21:30])

def get_text(x):
    return ' '.join(x)

# ---------------- 리스트에서 텍스트만 추출 ----------------
movie['genres']=movie['genres'].apply(get_text)
# print(movie.head())

# ---------------- 장르 TF-IDF 계산 ----------------
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

tfidf_matrix= tfidf.fit_transform(movie['genres']).toarray()
print(tfidf_matrix)
