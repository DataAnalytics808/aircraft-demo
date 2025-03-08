import streamlit as st
import requests
from PIL import Image
import io
import os
import textwrap
import gdown

# Set the page layout to wide.
st.set_page_config(layout="wide")

# Define a dictionary with model keys and descriptive text.
# model_dict = {
#     'CORS-ADD':           'Prediction from CORS-ADD model [2]', 
#     'fakeplanes':         'Prediction from our model, single class (Everman) [1]', 
#     'fakeplanes_3_class': 'Prediction from our model, bomber/fighter/cargo classes (Wagner) [1]'
# }
model_dict = {
    'YOLO_baseline_COCO': 'Prediction from Baseline YOLO model without fine-tuning', 
    'CORS-ADD':           'Prediction from CORS-ADD model [2]', 
    'HR-Planes':          'Prediction from HR-Planes model [3]', 
    'MAR20':              'Prediction from MAR20 model [4]', 
    'fakeplanes':         'Prediction from our model, single class (Everman) [1]', 
    'fakeplanes_3_class': 'Prediction from our model, bomber/fighter/cargo classes (Wagner) [1]'
}

########################################
# Helper Functions
########################################

def display_predictions(images, model_keys, model_dict):
    """
    Display a list of PIL images in a 2x3 grid with captions.
    """
    predictions_container = st.container()
    with predictions_container:
        st.subheader("Prediction Results")
        for row_start in range(0, len(images), 3):
            row_images = images[row_start : row_start + 3]
            row_keys = model_keys[row_start : row_start + 3]
            cols = st.columns(len(row_images))
            for col, img, key in zip(cols, row_images, row_keys):
                with col:
                    description = model_dict.get(key, "")
                    wrapped_description = "\n".join(textwrap.wrap(description, width=30))
                    st.image(img, use_container_width=True)
                    st.markdown(
                        f"<div style='font-size:16px; font-weight:bold;'>{wrapped_description}</div>"
                        f"<div style='font-size:12px;'>model = {key}.pt</div>",
                        unsafe_allow_html=True
                    )

def list_image_files(image_dir: str):
    """
    List image files with common extensions in the given directory.
    """
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".tif"]
    files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    files.sort()
    return files

def get_prediction_from_flask(model_name, image_path):
    """
    Call the Flask service to get prediction for a given model.
    """
    url = f"http://localhost:5000/predict/{model_name}"
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        st.error(f"Prediction failed for {model_name}: {response.text}")
        return None

########################################
# Setup paths and download models/images if needed
########################################

local_models_dir = "models"
local_images_dir = "images"
progress_message = st.empty()

# Download models from Google Drive if the folder is missing or empty.
folder_url = "https://drive.google.com/drive/folders/11nDOFw5igiVzOYhCoUCLE8injwa3TCod?usp=drive_link"
if not os.path.exists(local_models_dir) or len(os.listdir(local_models_dir)) == 0:
    progress_message.write("Downloading 6 models (1.1 GB) from Google Drive, this will take a minute")
    gdown.download_folder(url=folder_url, output=local_models_dir, quiet=False)
else:
    progress_message.write("Models already downloaded.")

# Download images from Google Drive if needed.
images_folder_url = "https://drive.google.com/drive/folders/11kHnyfd8uc5WfvUYOg1525kwemuJBWpa?usp=drive_link"
if not os.path.exists(local_images_dir) or len(os.listdir(local_images_dir)) == 0:
    progress_message.write("Downloading image files from Google Drive...")
    gdown.download_folder(url=images_folder_url, output=local_images_dir, quiet=False)
else:
    progress_message.write("Image files already downloaded.")

image_files = list_image_files(local_images_dir)
if not image_files:
    st.error("No image files found in the 'images' folder.")
    st.stop()

progress_message.write("All models and images ready for use")

########################################
# Display available images
########################################

images_container = st.container()
with images_container:
    st.subheader("Available Images - select at the bottom to detect aircraft")
    cols = st.columns(3)
    for i, file in enumerate(image_files):
        col = cols[i % 3]
        with col:
            st.image(file, caption=os.path.basename(file), use_container_width=True)

selected_image_name = st.selectbox(
    "Select an image to run predictions on:",
    [os.path.basename(f) for f in image_files],
)
progress_message.write(" ")

########################################
# Run predictions using Flask service
########################################

predict_button_placeholder = st.empty()
if predict_button_placeholder.button("Predict (this will take a minute)"):
    images_container.empty()
    predict_button_placeholder.empty()
    progress_prediction = st.empty()

    # Build the full path to the selected image.
    image_path = os.path.join(local_images_dir, selected_image_name)
    if not os.path.exists(image_path):
        st.error(f"Image not found at path: {image_path}")
        st.stop()

    ordered_model_keys = list(model_dict.keys())
    prediction_images = []
    total_models = len(ordered_model_keys)
    
    for i, model_key in enumerate(ordered_model_keys, start=1):
        progress_prediction.write(f"Calling Flask service for model {i} of {total_models}: {model_key} ...")
        predicted_image = get_prediction_from_flask(model_key, image_path)
        if predicted_image is not None:
            prediction_images.append(predicted_image)
    
    progress_prediction.write("All model predictions are complete, generating output")
    display_predictions(prediction_images, ordered_model_keys, model_dict)
    progress_prediction.write("All model predictions are complete, output displayed")
    
    if st.button("Predict using another image"):
        st.experimental_rerun()

##Old version that loads models into Streamlit memory - sometimes fails to run
# import streamlit as st
# import ultralytics
# import concurrent.futures
# from ultralytics import YOLO, checks, hub
# checks()  # checks

# import glob
# import pandas as pd
# from PIL import Image
# import os
# import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# import textwrap

# # 1) Set the page layout to "wide" for larger images
# st.set_page_config(layout="wide")

# ########################################
# # Helper Function
# ########################################
# def display_predictions(images, model_keys, model_dict):
#     """
#     Display a list of PIL images in a 2x3 grid using Streamlit containers/columns.
#     The first line is the descriptive placeholder text (larger font, wrapped).
#     The second line shows the model filename (smaller font).
#     """
#     predictions_container = st.container()
#     with predictions_container:
#         st.subheader("Prediction Results")

#         # We'll display up to 6 images in a 2x3 layout (2 rows, 3 columns)
#         # by iterating through images in steps of 3 for each row.
#         for row_start in range(0, len(images), 3):
#             row_images = images[row_start : row_start + 3]
#             row_keys   = model_keys[row_start : row_start + 3]

#             # Create a row of 3 columns (or fewer if at the last row)
#             cols = st.columns(len(row_images))

#             # Fill each column with one image and its text
#             for col, img, key in zip(cols, row_images, row_keys):
#                 with col:
#                     # Wrap the description so it doesn't go too wide
#                     description = model_dict.get(key, "")
#                     wrapped_description = "\n".join(textwrap.wrap(description, width=30))

#                     # Display the image
#                     st.image(img, use_container_width=True)

#                     # Display the two lines of text:
#                     #   1) the wrapped description in larger font
#                     #   2) a smaller line for the model filename
#                     st.markdown(
#                         f"<div style='font-size:16px; font-weight:bold;'>{wrapped_description}</div>"
#                         f"<div style='font-size:12px;'>model = {key}.pt</div>",
#                         unsafe_allow_html=True
#                     )

# ########################################
# # Caching functions so we don't reload
# # everything on each Streamlit rerun
# ########################################
# @st.cache_resource
# def load_models(model_dir: str):
#     """
#     Load all YOLO models from the given directory and return them as a dict.
#     Keys are the base filenames (without '.pt').
#     """
#     model_files = [
#         os.path.join(model_dir, f)
#         for f in os.listdir(model_dir)
#         if f.endswith(".pt")
#     ]
#     model_files.sort()
#     if not model_files:
#         st.error("No .pt model files found in the 'models' directory.")
#         st.stop()

#     loaded_models = {}
#     a=1
#     progress_prediction = st.empty()

#     for model_path in model_files:
#         # The key will be the filename without extension (e.g. 'CORS-ADD')
#         progress_prediction.write(f"Loading model {a} of {len(model_files)}: {model_path} ...")

#         model_name = os.path.splitext(os.path.basename(model_path))[0]
#         loaded_models[model_name] = YOLO(model_path)
#         a+=1
#     progress_prediction.write(f" ")
#     return loaded_models

# @st.cache_data
# def list_image_files(image_dir: str):
#     """
#     List valid image files in the given directory.
#     """
#     image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".tif"]
#     files = [
#         os.path.join(image_dir, f)
#         for f in os.listdir(image_dir)
#         if os.path.splitext(f)[1].lower() in image_extensions
#     ]
#     files.sort()
#     return files

# ########################################
# # MODEL DICTIONARY WITH DESCRIPTIVE TEXT
# ########################################
# model_dict = {
#     'YOLO_baseline_COCO': 'Prediction from Baseline YOLO model without fine-tuning', 
#     'CORS-ADD':           'Prediction from CORS-ADD model [2]', 
#     'HR-Planes':          'Prediction from HR-Planes model [3]', 
#     'MAR20':              'Prediction from MAR20 model [4]', 
#     'fakeplanes':         'Prediction from our model, single class (Everman) [1]', 
#     'fakeplanes_3_class': 'Prediction from our model, bomber/fighter/cargo classes (Wagner) [1]'
# }

# ########################################
# # STREAMLIT UI
# ########################################
# st.title("Aircraft Detection using Satellite Imagery")

# st.markdown(
#     """
#     <style>
#     .custom-paragraph {
#         margin: 0px;
#         padding: 0px;
#         line-height: 1.0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.text("This demonstration implements the models developed in [1], using datasets from [1-4]:")

# st.markdown('<p class="custom-paragraph">[1] <em>R. Everman, T. Wagner, N. Ranly, B. Cox, “Aircraft Detection from Satellite Imagery Using Synthetic Data,” Journal of Defense Modeling & Simulation (2025)</em></p>',
#     unsafe_allow_html=True)
# st.markdown('<p class="custom-paragraph">[2] <em>Unsal D. HRPlanesv2 Data Set - High Resolution Satellite Imagery for Aircraft Detection. [Online at github.com]; 2022.</em></p>',
#     unsafe_allow_html=True)
# st.markdown('<p class="custom-paragraph">[3] <em>Yu W, Cheng G, Wang M, Yao Y, Xie X, Yao X, et al. MAR20: A Benchmark for Military Aircraft Recognition in Remote Sensing Images. National Remote Sensing Bulletin. 2023; 27(12): 2688-2696.</em></p>',
#     unsafe_allow_html=True)
# st.markdown('<p class="custom-paragraph">[4] <em>Shi T, Gong J, Jiang S, Zhi X, Bao G, Sun Y, et al. Complex Optical Remote-Sensing Aircraft Detection Dataset and Benchmark. IEEE Transactions on Geoscience and Remote Sensing. 2023; 61.</em></p>',
#     unsafe_allow_html=True)
# st.markdown(' ',    unsafe_allow_html=True)


# # Ensure gdown is installed; if not, install it.
# try:
#     import gdown
# except ImportError:
#     import subprocess, sys
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
#     import gdown

# ########################################
# # SINGLE-LINE PROGRESS MESSAGE
# ########################################
# progress_message = st.empty()

# ########################################
# # DOWNLOAD MODELS IF NEEDED
# ########################################
# folder_url = "https://drive.google.com/drive/folders/11nDOFw5igiVzOYhCoUCLE8injwa3TCod?usp=drive_link"
# local_models_dir = "models"

# if not os.path.exists(local_models_dir) or len(os.listdir(local_models_dir)) == 0:
#     progress_message.write("Downloading 6 models (1.1 GB) from Google Drive, this will take 1-2 minutes")
#     gdown.download_folder(url=folder_url, output=local_models_dir, quiet=False)
# else:
#     progress_message.write("Model zip downloaded.")

# progress_message.write("Loading YOLO models into memory...")
# models = load_models(local_models_dir)
# progress_message.write(" ")

# ########################################
# # DOWNLOAD IMAGES IF NEEDED
# ########################################
# images_folder_url = "https://drive.google.com/drive/folders/11kHnyfd8uc5WfvUYOg1525kwemuJBWpa?usp=drive_link"
# local_images_dir = "images"

# if not os.path.exists(local_images_dir) or len(os.listdir(local_images_dir)) == 0:
#     progress_message.write("Downloading image files from Google Drive...")
#     gdown.download_folder(url=images_folder_url, output=local_images_dir, quiet=False)
# else:
#     progress_message.write("Image files downloaded.")

# image_files = list_image_files(local_images_dir)
# if not image_files:
#     st.error("No image files found in the 'images' folder.")
#     st.stop()

# progress_message.write("All models and images ready for use")

# ########################################
# # DISPLAY IMAGES AND SUBHEADER IN A CONTAINER
# ########################################
# images_container = st.container()
# with images_container:
#     st.subheader("Available Images - select at the bottom to detect aircraft")
#     cols = st.columns(3)
#     for i, file in enumerate(image_files):
#         col = cols[i % 3]
#         with col:
#             st.image(file, caption=os.path.basename(file), use_container_width=True)

# ########################################
# # SELECT IMAGE
# ########################################
# selected_image_name = st.selectbox(
#     "Select an image to run predictions on:",
#     [os.path.basename(f) for f in image_files],
# )

# progress_message.write(" ")

# # ########################################
# # # PREDICT WITH ALL MODELS
# # ########################################

# # def predict_model(model, image_path, conf=0.5):
# #     """
# #     Helper function to run prediction on a single model.
# #     Returns the plotted image from the prediction result.
# #     """
# #     results = model.predict(image_path, visualize=False, conf=conf)
# #     return results[0].plot()

# # ########################################
# # # PREDICT WITH ALL MODELS (Using ProcessPoolExecutor)
# # ########################################
# # predict_button_placeholder = st.empty()
# # if predict_button_placeholder.button("Predict (this will take a minute)"):
# #     # Clear the images container (subheader and images) and the predict button
# #     images_container.empty()
# #     predict_button_placeholder.empty()

# #     # Single-line progress text for predictions
# #     progress_prediction = st.empty()

# #     # Build the full path to the selected image
# #     image_path = os.path.join(local_images_dir, selected_image_name)
# #     if not os.path.exists(image_path):
# #         st.error(f"Image not found at path: {image_path}")
# #         st.stop()

# #     # Build a list of model keys in the EXACT order of model_dict, skipping any that aren't loaded
# #     ordered_model_keys = [k for k in model_dict if k in models]
# #     total_models = len(ordered_model_keys)
# #     prediction_images = []

# #     # Offload each model prediction to a separate process
# #     with concurrent.futures.ProcessPoolExecutor() as executor:
# #         future_to_key = {
# #             executor.submit(predict_model, models[key], image_path, 0.5): key
# #             for key in ordered_model_keys
# #         }
# #         for i, future in enumerate(concurrent.futures.as_completed(future_to_key), start=1):
# #             model_key = future_to_key[future]
# #             progress_prediction.write(f"Running model {i} of {total_models}: {model_key} ...")
# #             try:
# #                 plotted_image = future.result()
# #                 prediction_images.append(plotted_image)
# #             except Exception as exc:
# #                 progress_prediction.write(f"Model {model_key} generated an exception: {exc}")

# #     # After all models are done
# #     progress_prediction.write("All model predictions are complete, generating output")

# #     # Display the predictions side by side using the two-line title
# #     display_predictions(prediction_images, ordered_model_keys, model_dict)
# #     progress_prediction.write("All model predictions are complete, output displayed")

# #     # Button to predict using another image, which clears the screen and reruns the app
# #     if st.button("Predict using another image"):
# #         st.experimental_rerun()

# ########################################
# # PREDICT WITH ALL MODELS
# ########################################
# predict_button_placeholder = st.empty()
# if predict_button_placeholder.button("Predict (this will take a minute)"):
#     # Clear the images container (which contains the subheader and images) and the predict button
#     images_container.empty()
#     predict_button_placeholder.empty()

#     # Single-line progress text for predictions
#     progress_prediction = st.empty()

#     # Build the full path to the selected image
#     image_path = os.path.join(local_images_dir, selected_image_name)
#     if not os.path.exists(image_path):
#         st.error(f"Image not found at path: {image_path}")
#         st.stop()

#     # Build a list of model keys in the EXACT order of model_dict, skipping any that aren't loaded
#     ordered_model_keys = [k for k in model_dict if k in models]

#     # Perform predictions on the selected image with each model in that order
#     prediction_images = []
#     total_models = len(ordered_model_keys)
#     for i, model_key in enumerate(ordered_model_keys, start=1):
#         progress_prediction.write(f"Running model {i} of {total_models}: {model_key} ...")
#         results = models[model_key].predict(image_path, visualize=False, conf=0.5)
#         plotted_image = results[0].plot()
#         prediction_images.append(plotted_image)

#     # After all models are done
#     progress_prediction.write("All model predictions are complete, generating output")

#     # Display the predictions side by side using the two-line title
#     display_predictions(prediction_images, ordered_model_keys, model_dict)
#     progress_prediction.write("All model predictions are complete, output displayed")

#     # Button to predict using another image, which clears the screen and reruns the app
#     if st.button("Predict using another image"):
#         st.experimental_rerun()
