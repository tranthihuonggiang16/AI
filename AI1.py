# Import các thư viện cần thiết
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
import io
from PIL import Image

# Tải mô hình đã được huấn luyện
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x) 
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model = tf.keras.models.load_model("model01.hdf5")

# Viết hàm predict
def predict(image):
    resized = cv2.resize(image, (256, 256))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = np.expand_dims(resized, axis=0)
    return model.predict(img_reshape).argmax(axis=1)[0]

# Giao diện Streamlit
st.title("Nhận diện phương tiện")
uploaded_file = st.file_uploader("Chọn ảnh", type="jpg")
map_dict = {0: 'Ô tô. Phương tiện không vi phạm',
            1: 'Xe máy. Phương tiện vi phạm'}
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_stream = io.BytesIO(image_bytes)
    opencv_image = np.array(Image.open(image_stream))
    resized = cv2.resize(opencv_image, (224, 224))
    st.image(resized, channels="RGB")
    generate_pred = st.button("Nhận diện")
    if generate_pred:
        with st.spinner('Đang nhận diện ...'):
            prediction = predict(opencv_image)
            st.title("Đây là {}".format(map_dict[prediction]))