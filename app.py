from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/')
def home():
    return 'This is Home!'


@app.route('/data')
def data_form():
    return render_template('data_form.html')
     

@app.route('/data_result', methods = ['POST', 'GET'])
def data_result():
    if request.method == 'POST':
        result = request.form
    return render_template("sending_data_result.html", result = result)


if __name__ == '__main__':
    app.run('0.0.0.0', port=2222, debug=True)
