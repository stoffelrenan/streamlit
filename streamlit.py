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
import ta
import math
from xgboost import XGBRegressor
import plotly.graph_objects as go


#function calling local css sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local css sheet
local_css("style.css")

#Select the coin
st.sidebar.subheader("""Crypto+Asset Dashboards""")
selected_stock = st.sidebar.text_input("Enter a valid asset name...", "BTC")
button_clicked = st.sidebar.button("Select Asset")
if button_clicked == "Select Asset":
    main()
    
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset.iloc[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def predict_coin(curr,df, date = '2021-07-01', time_step=5):
    df = df.copy()
    df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    df = df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                'Adj Close':'adj_close','Volume':'volume'})
    df['date'] = pd.to_datetime(df.date)
    
    #create dataframe with needed features
    closedf = df[['date','close',"trend_macd"]]
    
    closedf = closedf[closedf['date'] > date]
    close_stock = closedf.copy()

    #set training size
    training_size=int(len(closedf)*0.80)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf.iloc[0:training_size,:],closedf.iloc[training_size:len(closedf),:]
    
    #normalize price
    del train_data['date']
    del test_data['date']
    scaler = MinMaxScaler()
    scaler_y = MinMaxScaler().fit(np.array(train_data['close']).reshape(-1,1)) 
    scaler.fit(train_data)
    X_train_scaled = pd.DataFrame(scaler.transform(train_data)) 
    X_test_scaled = pd.DataFrame(scaler.transform(test_data))
    
    #create the looped datasets for "close"
    X_train, y_train = create_dataset(X_train_scaled, time_step)
    X_test, y_test = create_dataset(X_test_scaled, time_step)
    
    #convert to dataframes
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    #append variables
    X_train = pd.concat([X_train, X_train_scaled.iloc[time_step + 1:,1:].reset_index(drop=True)], axis = 1, ignore_index = True)
    X_test = pd.concat([X_test, X_test_scaled.iloc[time_step + 1:,1:].reset_index(drop=True)], axis = 1, ignore_index = True)

    #convert back to array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    #build model
    my_model2 = XGBRegressor(n_estimators=1000)
    my_model2.fit(X_train, y_train, verbose=True)
    
    #get predictions
    predictions = my_model2.predict(X_test)
    #scaled_rmse = math.sqrt(mean_squared_error(y_test, predictions))
    
    train_predict=my_model2.predict(X_train)
    test_predict=my_model2.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)

    # Transform back to original form
    train_predict = scaler_y.inverse_transform(train_predict)
    test_predict = scaler_y.inverse_transform(test_predict)
    original_ytrain = scaler_y.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler_y.inverse_transform(y_test.reshape(-1,1)) 
    
    #get unscaled rmse 
    #RMSE = math.sqrt(mean_squared_error(original_ytest, test_predict))
    #trainRMSE = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    
    #Predicting tomorrow
    tomorrow_scaled = my_model2.predict(X_test)[-1]
    tomorrow = scaler_y.inverse_transform(tomorrow_scaled.reshape(-1,1)) 
    
    #Predicting day after tomorrow
    appended_X_test_scaled = X_test_scaled.append([float(tomorrow_scaled)], ignore_index = True)
    X_test_append, y_test_append = create_dataset(appended_X_test_scaled, time_step)
    
    #adding tomorrows variables to dataframe 
    X_test_append = pd.DataFrame(X_test_append)
    X_test_append = pd.concat([X_test_append, X_test_scaled.iloc[time_step + 1:,1:].reset_index(drop=True)], axis = 1, ignore_index = True).fillna(method="ffill")
    
    #predicting day after tomorrow 
    tomorrow_tomorrow_scaled = my_model2.predict(X_test_append)[-1]
    tomorrow_tomorrow = scaler_y.inverse_transform(tomorrow_tomorrow_scaled.reshape(-1,1)) 
    
    #create output
    #output = [curr,round(RMSE,4),round(trainRMSE,4), round(float(tomorrow),4), round(float(tomorrow_tomorrow),4) ]
    return round(float(tomorrow),4), round(float(tomorrow_tomorrow),4)
    
#main function
def main():
    st.title("Coin Dashboard")
    #get data on searched ticker
    data = yf.download(tickers=selected_stock+'-USD', period = '5y', interval = '1d')
    data.name=selected_stock
    data.reset_index(inplace=True)

    # get current date data for searched ticker
    current_price = yf.Ticker(selected_stock + '-USD')
    current_price = current_price.info['regularMarketPrice']

    # get current date closing price for searched ticker
    predicted_price_one, predicted_price_two = predict_coin(data.name, data)

    #defining 4 cards
    row1_1, row1_2, row1_3, row1_4 = st.columns((1, 1, 1, 1))

    with row1_1:
        st.write('Current price: ' + str(round(current_price,3)))

    with row1_2:
        st.write('Prediction for tomorrow: ' + str(predicted_price_one))

    with row1_3:
        st.write('Prediction for the day after tomorrow: ' + str(predicted_price_two))

    with row1_4:
        st.write("**Something else**")

    st.subheader("""Daily **closing price** for """ + selected_stock)

    #print line chart with daily closing prices for searched ticker
    #st.line_chart(data.Close)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']) )
    st.plotly_chart(fig)

    st.subheader("""Predicted **closing price of tomorrow** for """ + selected_stock)
    #define variable today

    st.write('Current price: ' + str(round(current_price,3)))
    st.write('Prediction for tomorrow: ' + str(predicted_price_one))
    st.write('Prediction for the day after tomorrow: ' + str(predicted_price_two))


if __name__ == "__main__":
    main()
