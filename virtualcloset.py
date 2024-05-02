import tensorflow
import numpy as np
import cv2
import os
import pickle
from numpy.linalg import norm
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors

# Load feature vectors and filenames
feature_list = np.array(pickle.load(open("C:\\Users\\ADMIN\\course\\Capstoneproj\\FASHION_AI_DATASET\\base\\Featurevector.pkl", "rb")))
filename = pickle.load(open("C:\\Users\\ADMIN\\course\\Capstoneproj\\FASHION_AI_DATASET\\base\\filenames.pkl", "rb"))

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a Sequential model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Define a function to extract features from an image
def extract_features(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Create an instance of NearestNeighbors class
neighbours = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
neighbours.fit(feature_list)

# Example usage: find nearest neighbors for a given image
normalized = extract_features("C:\\Users\\ADMIN\\course\\Capstoneproj\\FASHION_AI_DATASET\\base\\test_images\\test\\image\\000004.jpg", model)
distances, indices = neighbours.kneighbors([normalized])
for file_index in indices[0][1:6]:  # Exclude the first item as it is the query image itself
    img_name = filename[file_index]
    img = cv2.imread(img_name)
    cv2.imshow("Nearest Neighbor", cv2.resize(img, (640, 640)))
    cv2.waitKey(0)