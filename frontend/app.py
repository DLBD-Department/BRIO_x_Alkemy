from flask import Flask, render_template
from flask_cors import CORS
import os
from frontend.views import bias, opacity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
UPLOAD_FOLDER = os.path.abspath("uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.register_blueprint(bias.bp)
app.register_blueprint(opacity.bp)


@app.route('/', methods=['GET'])
def home():
    return render_template('homepage.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
