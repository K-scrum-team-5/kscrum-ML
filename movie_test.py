import pandas as pd
import numpy as np
import warnings
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore')

# 데이터 로드
movie = pd.read_csv("./csv/movie.csv")
rating = pd.read_csv("./csv/rating.csv")

# 장르를 리스트 형태로 변환
movie["genres"] = movie['genres'].apply(literal_eval)

# 장르값만 남기는 함수
def get_genres(x):
    try:
        genre = [i['name'] for i in x]
        return genre
    except:
        genre = []
        return genre

movie["genres"] = movie['genres'].apply(get_genres)

# 문자 공백 없애기
movie["genres"] = movie["genres"].apply(lambda x: [str(i).replace(" ", "") for i in x])

# 장르를 문자열로 변환
def get_text(x):
    return ' '.join(x)

movie['genres'] = movie['genres'].apply(get_text)

# 장르 벡터화
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movie['genres'])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# 코사인 유사도 DataFrame 생성
df_cosine = pd.DataFrame(data=cosine_sim, index=movie['id'], columns=movie['id'])

# 평점 데이터를 8:2로 train test 데이터로 나누기
train_rate, test_rate = train_test_split(rating, test_size=0.2, random_state=100)

# train 데이터에서 유저 아이디가 5인 데이터 추출
user5_train = train_rate[train_rate['userId'] == 5][['id', 'rating']]

# 공식의 분모 유사도 합 구하기 (user5_train의 79개 영화인 j와 전체 영화 i 에 대해)
sim_sum = df_cosine.loc[user5_train['id'].values, :].sum().values  # 유사도 합
sim_sum = sim_sum + 1  # 분모가 0인 경우 발생할 계산 오류를 피하기 위해 +1 해줌

# 공식의 분자 계산  각각 곱하고 더한 값이 필요하므로 내적 함수 사용
sim_rating = np.matmul(df_cosine.loc[user5_train['id'].values, :].T.values, user5_train['rating'].values)

# 최종 평점 예측
pred_rating = pd.DataFrame(np.divide(sim_rating, sim_sum), index=df_cosine.index)
pred_rating.columns = ["pred"]

# 각 영화의 평균 평점 계산
average_ratings = rating.groupby('id')['rating'].mean()

# 영화 ID를 기반으로 추천 영화를 반환하는 함수
def get_recommendations(movie_ids):
    sim_scores_total = np.zeros(len(cosine_sim))
    
    for movie_id in movie_ids:
        if movie_id in df_cosine.index:
            idx = df_cosine.index.get_loc(movie_id)
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            for i, score in sim_scores:
                current_movie_id = movie.loc[i, 'id']
                if current_movie_id in average_ratings:
                    sim_scores_total[i] += score * average_ratings[current_movie_id]
                else:
                    sim_scores_total[i] += score

    sim_scores_total = sorted(list(enumerate(sim_scores_total)), key=lambda x: x[1], reverse=True)
    sim_scores_total = sim_scores_total[1:21]  # 자기 자신 제외하고 상위 20개 추천
    
    movie_indices = [i[0] for i in sim_scores_total]

    # 추천 영화 및 예상 평점 반환
    recommended_movies = movie['title'].iloc[movie_indices]
    estimated_ratings = [average_ratings.get(movie.loc[i, 'id'], np.nan) for i in movie_indices]
    movie_ids = movie['id'].iloc[movie_indices]

    return pd.DataFrame({'MovieID': movie_ids, 'Title': recommended_movies, 'Estimated Rating': estimated_ratings})

# 테스트 코드 (필요 시 주석 해제)
# input_movie_ids = [1, 2, 3]
# recommended_movies = get_recommendations(input_movie_ids)
# print(recommended_movies)
