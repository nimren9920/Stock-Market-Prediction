# Importing libraries

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

def stock_model(x):

	# Reading .csv file and preparing data
	df = pd.read_csv("stock\{}_data.csv".format(x))
	# df = pd.read_csv("C:\SPP\csv\individual_stocks_5yr\individual_stocks_5yr\\UAA_data.csv")

	data = df.sort_index(ascending=True, axis=0)
	new_data = pd.DataFrame(index=range(0,len(df)),columns=['date','close'])

	for i in range(0,len(data)):
    		new_data['date'][i] = data['date'][i]
    		new_data['close'][i] = data['close'][i]

	new_data.index = new_data.date
	new_data.drop('date',axis=1,inplace=True)

	# Splitting data as training and testing

	ratio = 0.6
	elements = len(new_data)
	pivot = int(elements * ratio)

	dataset = new_data.values

	train_data = dataset[:pivot]
	test_data = dataset[pivot:]

	# Data normalization

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(dataset)

	# Creating x_train and y_train

	x_train, y_train = [], []
	for i in range(60,len(train_data)):
    		x_train.append(scaled_data[i-60:i,0])
    		y_train.append(scaled_data[i,0])
	x_train, y_train = np.array(x_train), np.array(y_train)

	x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

	model = Sequential()
	model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
	model.add(LSTM(units=50))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)

	inputs = new_data[len(new_data) - len(test_data) - 60:].values
	inputs = inputs.reshape(-1,1)
	inputs  = scaler.transform(inputs)

	X_test = []
	for i in range(60,inputs.shape[0]):
		X_test.append(inputs[i-60:i,0])
	X_test = np.array(X_test)

	X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
	closing_price = model.predict(X_test)
	closing_price = scaler.inverse_transform(closing_price)

	# RMS of model

	# rms=np.sqrt(np.mean(np.power((test_data-closing_price),2)))

	# Plotting
 
	# plt.figure(figsize = (16,8))
	train = new_data[:pivot]
	valid = new_data[pivot:]
	valid['Predictions'] = closing_price
	st.line_chart(train['close'])
	st.line_chart(valid[['Predictions']])
	st.line_chart(valid[['close','Predictions']])
	


def main():
	st.title("Stock Market Prediction")
	# x = st.text_input("Enter Stock Name : ")
	x = st.selectbox('Enter Stock Name :',('','UAA', 'AAPL', 'NKE','ADS','AMD'))
	if x == '':
		return
	else:
		stock_model(x)
	

if __name__ == "__main__":
    main()
