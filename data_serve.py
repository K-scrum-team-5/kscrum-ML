from flask import Flask, Blueprint, render_template, request

data_serve_bp = Blueprint('data_serve', __name__, template_folder='templates')

app = Flask(__name__)

@data_serve_bp.route('/input')
def data_form():
    return render_template('data_form.html')


@data_serve_bp.route('/data_result', methods = ['POST', 'GET'])
def data_result():
    if request.method == 'POST':
        result = request.form
    return render_template("sending_data_result.html", result = result)