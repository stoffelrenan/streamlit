import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
#LSTM Model
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential,Model
from tensorflow import keras
import keras


#function calling local css sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local css sheet
local_css("style.css")

#ticker search feature in sidebar
st.sidebar.subheader("""Crypto Dashboard""")

selected_stock = st.selectbox('Which coin do you want?',('BTC', 'LTC', 'ETH'))

button_clicked = st.sidebar.button("Choose coin")
if button_clicked == "Choose coin":
    main()

    #Macro function to predict the next 2 days for a certain coin.
#Returns list with the outputs for the 2 days
'''
Parameters:
name - String with the name of the coin
df - dataframe for the coin
days - how many days to use as basis for predicting (standard: 2)
epochs - how many epochs for the LSTM to run (standard: 30)

'''
#DF preparing functions for Close and High, also splitting the training/test dataset
def lstm_close(curr):
    df=curr.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df = np.array(df).reshape(-1,1)
    #df=scaler.fit_transform(np.array(df).reshape(-1,1))
    return scaler,df

#Split df using % of choice
def split_df(df):
    training_size=int(len(df)*0.95)
    test_size=len(df)-training_size
    train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]
    return train_data,test_data

#Using the provided data, create a dataset of sequential values using a timestep (number of days chosen)
def create_dataset(dataset, time_step):
    data_x, data_y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        data_x.append(a)
        data_y.append(dataset[i + time_step, 0])
    return np.array(data_x), np.array(data_y)

def predict_coin(name,df,days=2,epochs=30):
    
    scaler,df1 = lstm_close(df)
    
    #Split dataframe in train test
    train_data, test_data = split_df(df1)
    
    scaler=scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    
    #Set timestep of n days and create datasets to train
    time_step = days
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    #Reshaping the input as required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
    #Create LSTM model using layers and dropout to avoid overfitting
    model=Sequential()
    model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(days,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
   
    #Fit the model and predict for X_train and X_test. 
    #After that, present the RMSE for train and test
    model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs, batch_size=64,verbose=0)
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    #Reshaping and inverse transforming to calculate non scaled RMSE
    test_predict=scaler.inverse_transform(test_predict)
    train_predict=scaler.inverse_transform(train_predict)
    y_test=scaler.inverse_transform(y_test.reshape(-1, 1))
    testdf = pd.DataFrame()
    testdf['y_test'] = list(y_test)
    testdf['predict'] = list(test_predict)

    #Plotting
    # Train predictions plot
    look_back=days
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    
    # Test predictions plot
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:,:] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
   
    #Reshaping only the last n values for test data
    x_input=test_data[len(test_data)-days:].reshape(1,-1)
    x_input.shape
    
    #Prepare list for the loop predictions
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
#Predict the next 2 days (change value of i if want more days)
    lst_output=[]
    n_steps=days
    i=0
    while(i<1):
        if(len(temp_input)>days):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            predictions=model.predict(x_input, verbose=0)
            temp_input.extend(predictions[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(predictions.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1,n_steps,1))
            predictions = model.predict(x_input,verbose=0)
            temp_input.extend(predictions[0].tolist())
            lst_output.extend(predictions.tolist())
            i=i+1
    lst_output = scaler.inverse_transform(lst_output)
    return lst_output
    
#main function
def main():
    st.subheader("""Daily **closing price** for """ + selected_stock)
    #get data on searched ticker
    data = yf.download(tickers=selected_stock+'-USD', period = '5y', interval = '1d')
    data.name=selected_stock

    #print line chart with daily closing prices for searched ticker
    st.line_chart(data.Close)

    st.subheader("""Predicted **closing price of tomorrow** for """ + selected_stock)
    #define variable today 

    #get current date data for searched ticker
    current_price = yf.Ticker(selected_stock+'-USD')
    current_price = current_price.info['regularMarketPrice']
    
    #get current date closing price for searched ticker
    predicted_price = predict_coin(data.name,data)
    st.write('Current price: ' + str(round(current_price,2)))
    st.write('Prediction for tomorrow: ' + predicted_price,2)

    #get daily volume for searched ticker
    st.subheader("""Daily **volume** for """ + selected_stock)
    st.line_chart(data.Volume)


if __name__ == "__main__":
    main()
