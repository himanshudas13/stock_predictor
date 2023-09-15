import numpy as np
import pandas as pd
import pandas_datareader as data
from keras.models import load_model
import matplotlib.pyplot as plt
import webbrowser as web
import datetime as dt
import sklearn
from tensorflow import keras
import streamlit as st


start_date=dt.datetime(2010,1,1)
end_date=dt.datetime(2019,12,31)

st.title('STOCK TREND PREDICTION')

user_input =  st.text_input('Enter Stock Ticker','AAPL')
dt = data.DataReader(user_input,"stooq",start=start_date,end=end_date)
#print(dt)

#describing data
st.subheader('Data from 2010 - 2019')
st.write(dt.describe())
st.subheader("Closing price v/s time chart")
fig=plt.figure(figsize=(20,10))
plt.xlabel("Days")
plt.ylabel("Close Value")
hun_avg=dt.Close.rolling(100).mean()
plt.plot(dt.Close,'b')
plt.plot(hun_avg,'r')
st.pyplot(fig)


train=pd.DataFrame(dt['Close'][0:int(len(dt)*0.70)])
test=pd.DataFrame(dt['Close'][int(len(dt)*0.70):int(len(dt))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_array=scaler.fit_transform(train)
model=load_model('keras_model.h5')
past_100_days = train.tail(100)
final_df = past_100_days._append(test, ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test= np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
y_predicted/=scaler.scale_
y_test/=scaler.scale_
st.subheader('PREDICTIONS vs ORIGINAL')
fig2=plt.figure(figsize=(6,3))
plt.title("Predicted closing volume")
plt.xlabel("Days")
plt.ylabel("Closing Value")
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='predicted price')
plt.legend()
st.pyplot(fig2)