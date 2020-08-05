import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io
import base64
import time

st.set_option('deprecation.showfileUploaderEncoding', False)

# All the function needed 
#--------------------------------------------------------------------------------
# Define image size
IMG_SIZE = 224

# Define a batch size , 32 is a good start 
BATCH_SIZE = 32
# Import labels and create an array of 120 dog breeds
labels_csv = pd.read_csv("/home/gagan/Desktop/Ml-Sample/labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)

# Prediction label function
def get_pred_label(prediction_probabilities):
  """
  Turn an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

#---------------------------------------------------------------------------------

st.title("Welcome to Dog 🐕 Vision 👁️ AI")
st.write("")
st.write("Upload your dog's image")

file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    custom_image = Image.open(file)
    st.text("Are you excited?😀...🐶...")

if file:
	# Data preprocessing
	image = tf.io.decode_image(file.getvalue(), channels=3, dtype=tf.float32)
	image= tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
	data = tf.data.Dataset.from_tensor_slices([image])
	data_batch = data.batch(BATCH_SIZE)

	# Load pretrained model and make predictions
	loaded_full_model = tf.keras.models.load_model('/home/gagan/Desktop/Ml-Sample/20200727-18521595875929-full-image-set-mobilenetv2-Adam.h5',custom_objects={'KerasLayer':hub.KerasLayer})
	custom_preds = loaded_full_model.predict(data_batch)
	# Get predicted label
	custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
	
	# Starting a long computation...'
	latest_iteration = st.empty()
	bar = st.progress(0)

	for i in range(100):
	  # Update the progress bar with each iteration.
	  latest_iteration.text(f'Hold tight....{i+1}')
	  bar.progress(i + 1)
	  time.sleep(0.1)
	# '...and now we\'re done!'

	st.title(f'Your dog is a {custom_pred_labels[0]}')
	# st.write(custom_pred_labels[0])
	st.image(custom_image, use_column_width=True)




