from flask import Flask, render_template, request
from flask.json import load
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops
from datetime import datetime
from io import BytesIO
import base64
import json
import numpy as np


model = load_model('model/model.h5')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/playground')
def playground():
    return render_template('playground.html')


@app.route('/predict', methods=['POST'])
def predict():
    retJson = {"predict_digit": "Err", "detect_img": "",
               "centering_img": "", "prob": {}}

    # request.body
    postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
    res = predict(postImg)

    if res is not None:
        res = res[0]
        retJson["predict_digit"] = str(np.argmax(res))

        for i, item in enumerate(res):
            retJson["prob"][i] = float(item * 100)

        # save digits
        Image.open(postImg).save("./results/{}_{}.png".format(
            datetime.now().strftime('%m-%d_%H.%M.%S'), retJson["predict_digit"]))

    return json.dumps(retJson)

# predict digit


def predict(img_path):
    try:
        img = Image.open(img_path).convert('L')
    except IOError:
        print("image not found")
        return None
    # centering input digit
    img = __centering_img(img)

    img.thumbnail((28, 28))  # resize to 28x28
    img = np.array(img)
    img = 1 - np.array(img / 255)  # normalize
    img = img.reshape(1, 784)
    # predict
    res = model.predict(img)
    return res

# centering input digit


def __centering_img(img):
    width, height = img.size[:2]
    left, top, right, bottom = width, height, -1, -1
    imgpix = img.getdata()

    for y in range(height):
        yoffset = y * width
        for x in range(width):
            if imgpix[yoffset + x] < 255:

                # do not use GetPixel and SetPixel, it is so slow.
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    shiftX = (left + (right - left) // 2) - width // 2
    shiftY = (top + (bottom - top) // 2) - height // 2

    return ImageChops.offset(img, -shiftX, -shiftY)
