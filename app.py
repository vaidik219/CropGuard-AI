from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("plant_disease_model.keras")

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.min(), img_array.max())
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    img = preprocess_image(image_bytes)
    preds = model.predict(img)

    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])

    return jsonify({
        "prediction": class_names[idx],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
