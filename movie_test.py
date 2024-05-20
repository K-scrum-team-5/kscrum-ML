import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

movie = pd.read_csv("./csv/movie.csv")
rating = pd.read_csv("./csv/rating.csv")

unique_genres = movie['genres'].unique()

#문자열이 아닌 리스트 형태로 다시 바꿔주는 작업
from ast import literal_eval

movie["genres"]=movie['genres'].apply(literal_eval)

type(movie['genres'][0])

#장르값만 남기는 함수
def get_genres(x):
    try:
        genre=[i['name'] for i in x]
        return genre

    except:
        genre=[]
        return genre

movie["genres"]=movie['genres'].apply(get_genres)

### 문자 공백 없애기

movie["genres"]=movie["genres"].apply(lambda x: [str(i).replace(" ","") for i in x])

def get_text(x):
    return ' '.join(x)

movie['genres']=movie['genres'].apply(get_text)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

tfidf_matrix= tfidf.fit_transform(movie['genres']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 장르 벡터화
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movie['genres'])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(count_matrix, count_matrix)

df_cosine= pd.DataFrame(data=cosine_sim, index=movie['id'], columns=movie['id'])

#평점데이터를 8:2로 train test 데이터로 나우기

from sklearn.model_selection import train_test_split


train_rate, test_rate =train_test_split(rating, test_size=0.2, random_state=100)

#train 데이터에서 유저아이디가 5인 데이터 추출
user5_train= train_rate[train_rate['userId'] == 5][['id', 'rating']]

### 공식의 분모 유사도 합 구하기 (user5_train의 79개 영화인 j와 전체 영화 i 에 대해)
sim_sum = df_cosine.loc[user5_train['id'].values, :].sum().values    #유사도 합
sim_sum = sim_sum + 1                                                #분모가 0인경우 발생할 계산오류를 피하기 위해 +1 해줌

#공식의 분자 계산  각각 곱하고 더한 값이 필요하므로 내적 함수사용
sim_rating = np.matmul(df_cosine.loc[user5_train['id'].values, :].T.values, user5_train['rating'].values)

#최종 평점 예측
pred_rating = pd.DataFrame(np.divide(sim_rating, sim_sum), index=df_cosine.index)
pred_rating.columns=["pred"]


# movie_title을 인덱스로 하는 Series 생성
indices = pd.Series(movie.index, index=movie['title']).drop_duplicates()

# 각 영화의 평균 평점 계산
average_ratings = rating.groupby('id')['rating'].mean()

def get_recommendations(movie_titles):
    sim_scores_total = np.zeros(len(cosine_sim))
    
    for title in movie_titles:
        if title in indices:
            idx = indices[title]
            # 여러 인덱스 중 첫 번째 인덱스만 사용
            if isinstance(idx, pd.Series) or isinstance(idx, np.ndarray):
                idx = idx.iloc[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            for i, score in sim_scores:
                # 유사도 점수에 평균 평점을 곱하여 최종 점수 계산
                movie_id = movie.loc[i, 'id']  # 현재 영화의 ID
                if movie_id in average_ratings:
                    # 평균 평점이 존재할 경우 유사도 점수에 평균 평점을 곱함
                    sim_scores_total[i] += score * average_ratings[movie_id]
                else:
                    # 평균 평점이 없는 경우 유사도 점수만 사용
                    sim_scores_total[i] += score

    sim_scores_total = sorted(list(enumerate(sim_scores_total)), key=lambda x: x[1], reverse=True)
    sim_scores_total = sim_scores_total[1:21]  # 자기 자신 제외하고 상위 10개 추천
    
    movie_indices = [i[0] for i in sim_scores_total]

    # 추천 영화 및 예상 평점 반환
    recommended_movies = movie['title'].iloc[movie_indices]
    estimated_ratings = [average_ratings.get(movie.loc[i, 'id'], np.nan) for i in movie_indices]

    return pd.DataFrame({'Title': recommended_movies, 'Estimated Rating': estimated_ratings})


# input_movie_titles = ['The Matrix', 'Avatar', 'Inception']
# recommended_movies = get_recommendations(input_movie_titles)
# print(recommended_movies)

