import numpy as np 
import pandas as pd
import tensorflow as tf

from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("depression.h5")

def deep_depression_detector(activity_data_csv_file):
	df = pd.read_csv(activity_data_csv_file)
	x = np.array([df['activity'].tolist()])
	x = pad_sequences(
		x, maxlen=65407, 
		padding='post', 
		truncating='post'
	)

	y_pred = model.predict(x)
	if y_pred >= 0.5:
		return  {'prediction':'depressed'}
	else:
		return  {'prediction':'nondepressed'}

