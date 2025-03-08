import torch
from flask import Flask, request, send_file, jsonify
import io
from PIL import Image
from ultralytics import YOLO
import os
import gdown

app = Flask(__name__)

# Define the local models directory and the Google Drive folder URL.
local_models_dir = "models"
folder_url = "https://drive.google.com/drive/folders/11nDOFw5igiVzOYhCoUCLE8injwa3TCod?usp=drive_link"

# Download the models if the models directory doesn't exist or is empty.
if not os.path.exists(local_models_dir) or len(os.listdir(local_models_dir)) == 0:
    print("Downloading models from Google Drive...")
    gdown.download_folder(url=folder_url, output=local_models_dir, quiet=False)
else:
    print("Models already downloaded.")

# Load all models once at startup.
models = {
    "CORS-ADD": YOLO(os.path.join(local_models_dir, "CORS-ADD.pt")),
    "fakeplanes": YOLO(os.path.join(local_models_dir, "fakeplanes.pt")),
    "fakeplanes_3_class": YOLO(os.path.join(local_models_dir, "fakeplanes_3_class.pt"))
}
# models = {
#     "YOLO_baseline_COCO": YOLO(os.path.join(local_models_dir, "YOLO_baseline_COCO.pt")),
#     "CORS-ADD": YOLO(os.path.join(local_models_dir, "CORS-ADD.pt")),
#     "HR-Planes": YOLO(os.path.join(local_models_dir, "HR-Planes.pt")),
#     "MAR20": YOLO(os.path.join(local_models_dir, "MAR20.pt")),
#     "fakeplanes": YOLO(os.path.join(local_models_dir, "fakeplanes.pt")),
#     "fakeplanes_3_class": YOLO(os.path.join(local_models_dir, "fakeplanes_3_class.pt"))
# }

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    # Verify that the requested model exists.
    if model_name not in models:
        return jsonify({"error": f"Model {model_name} not found."}), 404

    # Ensure an image file is provided.
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Error reading image: {str(e)}"}), 400

    # Run prediction with the selected model.
    results = models[model_name].predict(image, conf=0.5, visualize=False)
    plotted_image = results[0].plot()  # returns a NumPy array
    pil_image = Image.fromarray(plotted_image)

    # Convert the prediction image to a PNG and return it.
    img_io = io.BytesIO()
    pil_image.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

# Health-check endpoint
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    # Listen on all interfaces (required for Codespaces) on port 5000.
    app.run(host="0.0.0.0", port=5000)

