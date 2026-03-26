from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2, tempfile
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

model = tf.keras.models.load_model("D:\FYP\Deepfake image and video detection\imagevideoDetecton.h5")
classes = ['AI', 'Edited', 'Real']

def preprocess(frame):
    frame = cv2.resize(frame, (244, 244))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    return preprocess_input(frame)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    max_class = None

    if request.method == "POST":

        # IMAGE
        if "image" in request.files and request.files["image"].filename:
            img = Image.open(request.files["image"]).convert("RGB")
            x = preprocess(np.array(img))
            preds = model.predict(x)[0]

        # VIDEO
        elif "video" in request.files and request.files["video"].filename:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            request.files["video"].save(tmp.name)

            cap = cv2.VideoCapture(tmp.name)
            all_preds = []
            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % 30 == 0:
                    x = preprocess(frame)
                    all_preds.append(model.predict(x, verbose=0)[0])
                count += 1

            cap.release()
            preds = np.mean(all_preds, axis=0)

        result = list(zip(classes, preds))
        max_class = classes[np.argmax(preds)]

    return render_template("index.html", result=result, max_class=max_class)

app.run(debug=True)
