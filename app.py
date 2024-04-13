from flask import Flask, request, render_template
from data_serve import data_serve_bp

app = Flask(__name__)

app.register_blueprint(data_serve_bp, url_prefix='/data')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0', port=2222, debug=True)
