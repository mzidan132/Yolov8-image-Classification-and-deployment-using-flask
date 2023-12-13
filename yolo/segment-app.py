from flask import Flask, render_template, request
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

def kmeans_segmentation(img_path):
    img = cv2.imread(img_path)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 20
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    segmented_img = center[label.flatten()].reshape(img.shape)

    return segmented_img

def predict_disease(segmented_img):
    # Load the YOLOv5 model
    model = YOLO('./last.pt')

    # Resize the segmented image to match the YOLOv5 model input size
    img = Image.fromarray(segmented_img)
    img = img.resize((255, 255))

    # Perform prediction on the segmented image
    results = model(img, show=True)

    # Extract relevant information from the prediction
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    prediction = names_dict[probs.index(max(probs))]

    return prediction, probs, names_dict

@app.route('/')
def home():
    return render_template('index.html', prediction=None, probs=None, names_dict=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Save the file temporarily (optional)
    img_path = 'temp_image.jpg'
    file.save(img_path)

    # Perform K-means clustering on the image
    segmented_img = kmeans_segmentation(img_path)

    # Pass the segmented image to the YOLOv5 model for prediction
    prediction, probs, names_dict = predict_disease(segmented_img)

    # Render the template with the prediction result
    return render_template('index.html', prediction=prediction, probs=probs, names_dict=names_dict)

if __name__ == '__main__':
    app.run(debug=True)
