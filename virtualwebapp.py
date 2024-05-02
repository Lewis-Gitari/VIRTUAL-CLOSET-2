#import the necessary libraries
from email.mime import image
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
import requests

# Set up the layout
st.set_page_config(layout="wide")

# Background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.organizeatoz.com%2Fvirtual-closet-design&psig=AOvVaw0IjKlD65_AQJ_wnAG4BBB-&ust=1714106523988000&source=images&cd=vfe&opi=89978449&ved=0CBAQjRxqFwoTCMjD2cjG3IUDFQAAAAAdAAAAABAE") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#load the pickle files containing the featurefiles and feature_names
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

#create the site appearance 
st.title('VIRTUAL CLOSET FASHION RECCOMMENDATION')
def save_uploaded_file(uploaded_file):
    try:
        # Create the uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print(f"Error saving file: {e}")
        return 0

#extract features on uploaded image
def extract_features(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend (features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=3, algorithm= "auto", metric= "euclidean")
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Upload an image")


st.subheader("Uploaded Image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
            #display the file
        display_image = Image.open(uploaded_file)
        resized_image = display_image.resize((200,200))
        st.image(resized_image)
        #feature extraction to the image
        features = extract_features(os.path.join("uploads", uploaded_file.name), model)
            #recommendations
        indices = recommend(features, feature_list)
            #show results
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filename[indices[0][0]])
            
        with col2:
            st.image(filename[indices[0][1]])
            
        with col3:
            st.image(filename[indices[0][2]])
            
        with col4:
            st.image(filename[indices[0][3]])

        with col5:
            st.image(filename[indices[0][4]])

    else:
        st.header("Please confirm your upload")
            
                        
                                    
            
            
