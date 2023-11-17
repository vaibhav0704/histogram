from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
from flask_cors import cross_origin
from enhance_image import histogram_equalization
from clahe import clahe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

app = Flask(__name__)

@app.route("/cdf", methods=["POST"])
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

@app.route("/clahe", methods=["POST"])
@cross_origin(origins="*")
def clahe_img():
    image_file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    print(image[:,:,0])
    clahe_img = clahe(image[:,:,0],8,0,0)

    def plot_to_image(image, clahe_img):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image[:, :, 0], cmap='gray')
        axs[1].imshow(clahe_img, cmap='gray')

        # Create a canvas to render the figure to an image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert the plot to an image and return it
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # Close the plot to release resources
        plt.close(fig)

        return image

    plot_image = plot_to_image(image, clahe_img)

    _, img_encoded = cv2.imencode('.png', plot_image)
    img_bytes = img_encoded.tobytes()

    return send_file(
        io.BytesIO(img_bytes),
        mimetype='image/png',
        as_attachment=True,
        download_name='enhanced_image.png'
    )
    
if __name__ == "__main__":
    app.run(debug=True)
