from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
from keras.models import load_model

import numpy as np
import os
import base64

# =========================
# FLASK APP
# =========================

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# =========================
# LOAD MODEL
# =========================

model = load_model(
    "face_rating_model.h5",
    compile=False
)

# =========================
# UPLOAD FOLDER
# =========================

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):

    os.makedirs(UPLOAD_FOLDER)

# =========================
# HOME PAGE
# =========================

@app.route("/")

def home():

    return render_template("index.html")

# =========================
# PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])

def predict():

    file = request.files.get("image")

    image_data = request.form.get("image_data")

    image_path = None

    # =========================
    # UPLOAD IMAGE
    # =========================

    if file:

        image_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(image_path)

    # =========================
    # CAMERA IMAGE
    # =========================

    elif image_data:

        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        image_path = os.path.join(
            UPLOAD_FOLDER,
            "captured.png"
        )

        with open(image_path, "wb") as f:

            f.write(image_bytes)

    else:

        return jsonify({

            "error": "No image provided"

        })

    try:

        # =========================
        # DEEPFACE EMBEDDING
        # =========================

        embedding = DeepFace.represent(

            img_path=image_path,

            model_name="Facenet",

            enforce_detection=False

        )[0]["embedding"]

        embedding = np.array(
            embedding
        ).reshape(1, -1)

        # =========================
        # PREDICTION
        # =========================

        prediction = model.predict(
            embedding
        )

        rating = round(
            float(prediction[0][0]),
            2
        )

        return jsonify({

            "rating": rating

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )