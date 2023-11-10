from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
from flask_cors import cross_origin
from enhance_image import histogram_equalization

app = Flask(__name__)

@app.route("/enhance", methods=["POST"])
@cross_origin(origins="*")
def enhance():
    image_file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    merged_image = histogram_equalization(image)

    _, img_encoded = cv2.imencode('.png', merged_image)
    img_bytes = img_encoded.tobytes()

    return send_file(
        io.BytesIO(img_bytes),
        mimetype='image/png',
        as_attachment=True,
        download_name='enhanced_image.png'
    )

if __name__ == "__main__":
    app.run(debug=True)
