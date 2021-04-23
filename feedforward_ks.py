"""Simple feedforward network class in Keras

Exports:
FFNetwork (class)
"""

import keras
from keras.layers import Dense  
from keras.models import Sequential
from keras.optimizers import Adam

class FFNetwork():

	def __init__(self, dimensions):
		"""Instantiates the network """

		model = Sequential()
		model.add(Dense(dimensions[1], input_shape=(dimensions[0],), activation="relu"))
		#In Keras, one can use the Sequential class as the model, which allows the sequential adding of layers to the 
		#network with the add() method. Dense layers in Keras are equivalent to the fully-connected layers used in the
		#From Scratch model

		#Adds Dense layer for each dimension
		for i in range(len(dimensions[2:])):
			model.add(Dense(dimensions[i+2], activation="relu"))

		self.model = model 

	def run_model(self, train_x, train_y, epochs, mini_batch_size, learning_rate, val=None):
		"""Trains the model """
		
		self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
		#Compiles a model with choice of loss function and optimizer, "metrics" being the desired evaluation measurement

		self.model.fit(train_x, train_y, validation_data=val, batch_size=mini_batch_size, epochs=epochs, shuffle=True, verbose=0)
		#Trains on input data and tests on validation data

	def evaluate(self, test_x, test_y):
		"""Measures model accuracy on given data """

		return self.model.evaluate(test_x, test_y, verbose=0)