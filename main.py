import numpy as np
import pandas_datareader as data
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser as web
import datetime as dt
import sklearn
from tensorflow import keras
import streamlit as st



st.write(dt.describe())

st.subheader("Closing price v/s time chart")
fig=plt.figure(figsize=(12,6))
plt.plot(dt.Close)
st.pyplot(fig)



#70 per cent in train and rest in test split
train=pd.DataFrame(dt['Close'][0:int(len(dt)*0.70)])
test=pd.DataFrame(dt['Close'][int(len(dt)*0.70):int(len(dt))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_array=scaler.fit_transform(train)
x_train=[]
y_train=[]
for i in range(100,train_array.shape[0]):
    x_train.append(train_array[i-100:i])
    y_train.append(train_array[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

#ML MODEL

from tensorflow import keras
import sklearn


from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.summary()
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(x_train,y_train,epochs=50)
model.save('keras_model.h5')



# model call
'''
past_100_days = train.tail(100)
final_df = past_100_days._append(test, ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test= np.array(x_test),np.array(y_test)


#Making predictions
y_predicted=model.predict(x_test)
y_predicted/=scaler.scale_
y_test/=scaler.scale_

plt.figure(figsize=(6,3))
plt.title("Predicted closing volume")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='predicted price')
plt.legend()
plt.show()'''

