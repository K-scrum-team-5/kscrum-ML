from flask import Flask
app = Flask(__name__)


@app.route('/')
def home():
    return 'This is Home!'


@app.route('/build')
def mypage():
    return 'This is Build Page'


if __name__ == '__main__':
    app.run('0.0.0.0', port=2222, debug=True)
