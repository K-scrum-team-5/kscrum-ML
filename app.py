from flask import Flask, request, jsonify
import movie_test  # movie_test.py 파일을 포함한다고 가정
import json

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/recommendations', methods=['GET'])
def recommendations():
    movie_id_str = request.args.get('movie_id')  # GET 요청의 쿼리 파라미터에서 데이터 추출
    movie_id = list(map(int, movie_id_str.split("|")))  # 쉼표로 구분된 문자열을 리스트로 변환, 정수형으로 변환
    recommendations = movie_test.get_recommendations(movie_id)  # movie_test.get_recommendations 함수를 적절히 구현해야 함

    recommendations_json = recommendations.to_json(orient='records')
    print(recommendations_json)
    return jsonify(json.loads(recommendations_json))

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=2222)
