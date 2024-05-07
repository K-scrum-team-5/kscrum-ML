from flask import Flask, request, render_template, jsonify
from data_serve import data_serve_bp

import movie_test, json

app = Flask(__name__)


@app.route('/recommendations', methods=['GET'])
def recommendations():
    movie_titles_str = request.args.get('movie_titles')  # GET 요청의 쿼리 파라미터에서 데이터 추출
    movie_titles = movie_titles_str.split(", ")  # 쉼표로 구분된 문자열을 리스트로 변환
    recommendations = movie_test.get_recommendations(movie_titles)  # movie_test.get_recommendations 함수를 적절히 구현해야 함

    recommendations_json = recommendations.to_json(orient='records')
    print(recommendations_json)
    return jsonify(json.loads(recommendations_json))

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=2222)


