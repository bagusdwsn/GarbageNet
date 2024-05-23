import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from util import classify

# set title
st.title('Klasifikasi Citra Sampah')

# set header
st.header('Unggah gambar')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('model/garbage_deploy.h5')

# class names
class_names = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    