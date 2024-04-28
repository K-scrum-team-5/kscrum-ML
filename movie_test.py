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
# print(tfidf_matrix)

# ---------------- 코사인 유사도 계산 ----------------
from sklearn.metrics.pairwise import cosine_similarity

cos_matrix = cosine_similarity(tfidf_matrix,tfidf_matrix)
# print(cos_matrix)

df_cosine= pd.DataFrame(data=cos_matrix, index=movie['id'], columns=movie['id'])
# df_cosine.head()

# ---------------- 평점 데이터 불러오기 ----------------
rating= pd.read_csv("./csv/rating.csv")

# ---------------- 평점데이터를 8:2로 train test 데이터로 나우기 ----------------

from sklearn.model_selection import train_test_split

train_rate, test_rate =train_test_split(rating, test_size=0.2, random_state=100)
# print(train_rate.head())

#train 데이터에서 유저아이디가 5인 데이터 추출
user5_train= train_rate[train_rate['userId'] == 5][['id', 'rating']]

### 공식의 분모 유사도 합 구하기 (user5_train의 79개 영화인 j와 전체 영화 i 에 대해)
sim_sum = df_cosine.loc[user5_train['id'].values, :].sum().values    #유사도 합
sim_sum = sim_sum + 1                                                #분모가 0인경우 발생할 계산오류를 피하기 위해 +1 해줌
# print(sim_sum)
# print(df_cosine.loc[user5_train['id'].values, :].T.values.shape)

#공식의 분자 계산  각각 곱하고 더한 값이 필요하므로 내적 함수사용
sim_rating = np.matmul(df_cosine.loc[user5_train['id'].values, :].T.values, user5_train['rating'].values)
sim_rating

#최종 평점 예측
pred_rating = pd.DataFrame(np.divide(sim_rating, sim_sum), index=df_cosine.index)
pred_rating.columns=["pred"]
# print(pred_rating)


### 어떤 영화 제목을 입력하면 그와 유사도가 높은 10개 영화와 각 예상 평점 나오도록 하는 함수 만들기
indices = pd.Series(movie.index , index=movie['title'])
# print(indices)

#영화 제목은 같은데 영화 id는 다를경우 위 함수로는 에러가 남. 영화 id가 다르면 다른 영화로 보고 모두 출력하는 함수 만들기

def similar_movie(title, cos_matrix):
    try:

        idx = indices[title]                #제목을 입력하면 해당 인덱스 값(몇번째 행인지)을 가져옴

        cos_sim = list(enumerate(cos_matrix[idx]))         #몇번째 행인지와 유사도를 같이 가져옴
        cos_sim.sort(key=lambda x :x[1], reverse=True)    #유사도 값을 기준으로 내림차순 정렬

        cos_sim = cos_sim[0:11]                           #상위 11개 작품 (다음 코드에서 입력한 작품 제외할 것)
        sim_movie_idx=[x[0] for x in cos_sim]            #cos_sim에서 첫번쨰 항목만 즉, 몇행인지 값 추출
        sim_movie_idx.remove(idx)                        #해당 작품 제외
        sim_movie_idx = sim_movie_idx[0:10]              #상위 10개 작품만


        title_list=movie['title'].iloc[sim_movie_idx]  #인덱스로 유사한 영화 제목 추출
        title_list=pd.DataFrame(title_list)           #데이터프레임 만듦

        id_list=movie['id'].iloc[sim_movie_idx]     #인덱스로 영화 id 추출
        id_list=pd.DataFrame(id_list)  #데이터 프레임으로 만듦

        temp1 = pd.concat([title_list,id_list], axis=1)   #title_list, id_list 데이터프레임 병합
        temp2=pd.merge(temp1, pred_rating["pred"],left_on = 'id',right_index=True)    #temp1과 pred_rating 병합

        result=temp2.drop(["id"], axis=1)
        result.rename(columns={"title":"유사한 영화 TOP10", "pred":"예상평점"},inplace=True)

        print(result)


    except:
        idx=[]
        if len(indices[title])>1:

            for i in range(len(indices[title])):
                idx.append(indices[title][i])                          #같은 제목중 코드 하나만


                cos_sim = list(enumerate(cos_matrix[idx[i]]))         #몇번째 행인지와 유사도를 같이 가져옴
                cos_sim.sort(key=lambda x :x[1], reverse=True)    #유사도 값을 기준으로 내림차순 정렬


                cos_sim = cos_sim[0:11]                         #상위 11개 작품 (다음 코드에서 입력한 작품 제외할 것)
                sim_movie_idx=[x[0] for x in cos_sim]            #cos_sim에서 첫번쨰 항목만 즉, 몇행인지 값 추출


                if idx[i] in sim_movie_idx:
                    sim_movie_idx.remove(idx[i])                    #해당 작품 제외

                sim_movie_idx = sim_movie_idx[0:10]              #상위 10개 작품만



                title_list = movie['title'].iloc[sim_movie_idx]  #인덱스로 유사한 영화 제목 추출
                title_list=pd.DataFrame(title_list)



                id_list=movie['id'].iloc[sim_movie_idx]     #인덱스로 영화 id 추출
                id_list=pd.DataFrame(id_list)


                temp1 = pd.concat([title_list,id_list], axis=1)

                temp2=pd.merge(temp1, pred_rating["pred"],left_on = 'id',right_index=True)    #temp1과 pred_rating 병합



                result=temp2.drop(["id"], axis=1)
                result.rename(columns={"title":f"영화 id:{idx[i]} 유사한 영화 TOP10", "pred":"예상평점"},inplace=True)

                print(result)

similar_movie("Toy Story",cos_matrix)
