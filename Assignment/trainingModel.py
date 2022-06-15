import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Create two python array to contain data
X = []
y = []

#Load all the data from file and attach label to it according to their source
for i in range(1, 24):
	df = pd.read_csv("data\condition\condition_{}.csv".format(i))
	x1 = np.array(df['activity'].tolist())
	X.append(x1)
	y.append(1)

for i in range(1, 33):
	df = pd.read_csv("data\control\control_{}.csv".format(i))
	x1 = np.array(df['activity'].tolist())
	X.append(x1)
	y.append(0)

#Convert the data to numpy array
X = np.array(X)
y = np.array(y)

#Add zero behind to data so all data have same length
X = pad_sequences(
	X, 
	maxlen=max([len(x1) for x1 in X]), 
	padding='post', 
	truncating='post',
)

#Split the training set and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#Construct a nueral network model
model = models.Sequential([
	layers.Dense(128, activation='relu', input_shape=X_train.shape[1:]),
	layers.Dropout(0.3),

	layers.Dense(64, activation='relu'),
	layers.Dropout(0.3),

	layers.Dense(1, activation='sigmoid')
])

#Prepare for training the model, chose the algorithm, loss function, and metrics
model.compile(
	optimizer="adam",
	loss="binary_crossentropy",
	metrics=["binary_accuracy"],
)

#Training the model
model.fit(
	X_train,
	y_train,
	validation_data=(X_test, y_test),
	batch_size=60,
	epochs=100,
)

#Save the model
model.save("Assignment\model\depression.h5")