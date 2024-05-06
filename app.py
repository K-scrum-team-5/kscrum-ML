from flask import Flask, request, render_template, jsonify
from data_serve import data_serve_bp

import movie_test, json

app = Flask(__name__)

@app.route('/modifyList', methods=['POST'])
def modify_list():
    data = request.json  # Spring Boot로부터 받은 JSON 데이터
    print("Received list:", data)
    
    # 리스트에 요소 추가
    data.append("New element from Flask")
    
    # 수정된 리스트를 JSON 형태로 Spring Boot 서버로 다시 보냄
    return jsonify(data)

@app.route('/modifyAge', methods=['POST'])
def modify_age():
    person_data = request.get_json()
    person_data['age'] += 10  # 나이에 10 더하기
    return jsonify(person_data)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_titles_str = request.form.get('movie_titles')
    movie_titles = movie_titles_str.split(", ")  # 쉼표로 구분된 문자열을 리스트로 변환
    recommendations = movie_test.get_recommendations(movie_titles)

    recommendations_json = recommendations.to_json(orient='records')
    print(recommendations_json)
    return jsonify(json.loads(recommendations_json))

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=2222)


