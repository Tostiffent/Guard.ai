from io import BytesIO
from scripts.CropFaces import *
from scripts.DetectDeepfake import *
from scripts.DetectArt import *
from flask import Flask, request, send_file
from flask_cors import CORS
from transformers import pipeline
import io
from PIL import Image

def convert_blob_to_image(blob):
    """Converts a blob to an image and saves it."""

    # Convert blob to bytes-like object
    byte_stream = io.BytesIO(blob)

    # Open image using Pillow
    image = Image.open(byte_stream)

    # Save the image
    image.save("output_image.jpg")


pipe = pipeline("image-classification", "umm-maybe/AI-image-detector")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def handle_root():
    return 'backend api'


@app.route('/extension.crx')
def send_report():
    return send_file('fence_ai.crx')


@app.route('/detect_image', methods=['POST'])
def handle_detect_image():
    if 'file' in request.files:
        file = request.files['file']
        file_stream = BytesIO(file.read())
        file_np_array = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
            face_array = crop_faces(file_np_array)
            if (face_array.shape[0] != 0):
                predictions = [predict(face) for face in face_array]
                real = [pred['real'] for pred in predictions]
                fake = [pred['fake'] for pred in predictions]
                real_avg = sum(real) / len(real)
                fake_avg = sum(fake) / len(fake)
                print("Face detection")
                print(real_avg, fake_avg)
                # result = predict_face(pil_images)
                return f"% Real: {real_avg * 100}, % Fake: {fake_avg * 100}"
            else:
                # face absent so perfect art based detection
                print("art/no face detection")
                result = predict_art(file_np_array, pipe)
                print(result)
                return f"% Artificial: {result['artificial'] * 100}, % Real: {result['human'] * 100}"
        else:
            return f'Unsupported file format: {file_extension}'
    else:
        return 'No image data received'


if __name__ == '__main__':
    app.run()