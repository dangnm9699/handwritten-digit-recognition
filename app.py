from flask import Flask
from flask.json import load
from tensorflow.keras.models import load_model

model = load_model('model/model.h5')

app = Flask(__name__)


@app.route('/')
def index():
    return 'dangnm'
