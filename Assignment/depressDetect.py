import numpy as np 
import pandas as pd
import tensorflow as tf

from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the model
model = tf.keras.models.load_model("model\depression.h5")

def deep_depression_detector(activity_data_csv_file):
	#load the data
	df = pd.read_csv(activity_data_csv_file)
	#convert to numpy array
	x = np.array([df['activity'].tolist()])
	#add zero at behind so the data have same length with the longest data in data set
	x = pad_sequences(
		x, maxlen=65407, 
		padding='post', 
		truncating='post'
	)

	#Predict the result
	y_pred = model.predict(x)
	#Return the result
	if y_pred >= 0.5:
		return  {'prediction':'depressed'}
	else:
		return  {'prediction':'nondepressed'}