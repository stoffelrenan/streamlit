import streamlit as st
import pandas as pd
import yfinance as yf
from stockstats import StockDataFrame
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
import matplotlib.pyplot as plt


#function calling local css sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local css sheet
local_css("style.css")

def lstm_close(curr):
    df=curr.reset_index()['close']
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

def predict_coin(name,df,days=3,epochs=5):
    
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
    model.add(LSTM(5,activation='relu',return_sequences=True,input_shape=(days,1)))
    model.add(LSTM(5,return_sequences=True))
    model.add(LSTM(5))
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
    while(i<2):
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

def fib_retrace(currency):
  currency = StockDataFrame.retype(currency)
  
  # Fibonacci constants
  max_value = currency['close'].max()
  min_value = currency['close'].min()
  difference = max_value - min_value

  # Set Fibonacci levels
  first_level = max_value - difference * 0.236
  second_level = max_value - difference * 0.382
  third_level = max_value - difference * 0.5
  fourth_level = max_value - difference * 0.618
  # Plot Fibonacci graph
  plot_title = 'Fibonacci Retracement'
  fig = plt.figure(figsize=(22.5, 12.5))
  plt.title(plot_title, fontsize=30)
  ax = fig.add_subplot(111)
  plt.plot(currency.index, currency['close'])
  plt.axhline(max_value, linestyle='--', alpha=0.5, color='purple')
  ax.fill_between(currency.index, max_value, first_level, color='purple', alpha=0.2)

  # Fill sections
  plt.axhline(first_level, linestyle='--', alpha=0.5, color='blue')
  ax.fill_between(currency.index, first_level, second_level, color='blue', alpha=0.2)
  plt.axhline(second_level, linestyle='--', alpha=0.5, color='green')
  ax.fill_between(currency.index, second_level, third_level, color='green', alpha=0.2)
  plt.axhline(third_level, linestyle='--', alpha=0.5, color='red')
  ax.fill_between(currency.index, third_level, fourth_level, color='red', alpha=0.2)
  plt.axhline(fourth_level, linestyle='--', alpha=0.5, color='orange')
  ax.fill_between(currency.index, fourth_level, min_value, color='orange', alpha=0.2)
  plt.axhline(min_value, linestyle='--', alpha=0.5, color='yellow')
  plt.xlabel('Date', fontsize=20)
  plt.ylabel('Close Price (USD)', fontsize=20)
  return fig


#rsi function
def computeRSI (data, time_window):
    diff = data.diff(1).dropna() # diff in one field(one day)
    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    up_chg_avg = up_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1, min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)    
    return rsi
def RSIgraph (currency): 
    currency = StockDataFrame.retype(currency)
    df=currency.copy()
    df['close']=computeRSI(df['close'], 14)
    #set the high and low lines (as columns)
    df['low'] = 30
    df['high'] = 70
    fig = go.Figure()
    #create lines/traces
    fig.add_trace(go.Scatter(x=df.index, y=df['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color="Blue", width=1),))
    fig.add_trace(go.Scatter(x=df.index, y=df['high'],
                         fill=None,
                         mode='lines',
                         name='Sell',
                         line=dict(width=0.5, color='rgb(222, 196, 255)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index,y=df['low'],
                             fill='tonexty', # fill area between trace0 and trace1
                             mode='lines',
                             name='Buy',
                             line=dict(width=0.5, color='rgb(222, 196, 255)', dash='dash')))
    #update axis ticks
    fig.update_yaxes(nticks=30,showgrid=True)
    fig.update_xaxes(nticks=12,showgrid=True)
    #update layout
    fig.update_layout(title="<b>Daily RSI</b>"
                     , height = 700
                     , xaxis_title='Date'
                     , yaxis_title='Relative Strength Index'
                     , template = "plotly" #['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark']
                     )
    #update legend
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    #show the figure
    return fig


def moving_average(data):
    data = StockDataFrame.retype(data)
    data['macd'] = data.get('macd') # calculate MACD

    trace1 = dict(type='scatter',
                  x=data.index,
                  y=data['close'],
                  name='Close Price'
                  )

    trace2 = dict(type='scatter',
                  x=data.index,
                  y=data['macd'],
                  name='MACD Line'
                  )

    trace3 = dict(type='scatter',
                  x=data.index,
                  y=data['macds'],
                  name='MACD Signal Line'
                  )

    # Setting color for the MACD bar
    y = np.array(data["macdh"])
    color = np.array(["rgb(255,255,255)"] * y.shape[0])
    color[y < 0] = "firebrick"
    color[y >= 0] = "green"

    trace4 = go.Bar(
        x=data.index,
        y=data["macdh"],
        marker=dict(color=color.tolist()),
        opacity=1,
        name='MACD Histogram'
    )
    # trace4 = dict(type='scatter',
    #                  x=data.index,
    #                  y=data['macdh'],
    #                 name='MACD Histogram'
    #          )

    data_macd = [trace1, trace2, trace3, trace4]

    layout_macd = dict(title=dict(text=selected_stock + ' MACD Strategy'),
                  xaxis=dict(title='Date'),
                  yaxis=dict(title=selected_stock + ' MACD')
                  )

    fig = go.Figure(data=data_macd, layout=layout_macd)
    return fig

#Select the page
st.sidebar.subheader("""Select the page""")
# Create a page dropdown 
page = st.sidebar.selectbox("Choose your page", ["Asset Dashboard", "Company Investments"])



if page == "Asset Dashboard":
    #main function
    def main():
        st.title("Asset Dashboard: "+ selected_stock)
        #get data on searched ticker
        data = yf.download(tickers=selected_stock+'-USD', period = '5y', interval = '1d')
        data.name=selected_stock
        data['macd'] = data.get('macd')  # calculate MACD
        data.reset_index(inplace=True)

        # get current date data for searched ticker
        current_price = yf.Ticker(selected_stock + '-USD')
        current_price = current_price.info['regularMarketPrice']
 

        # get current date closing price for searched ticker
        predicted_price_one, predicted_price_two = predict_coin(data.name, data)
        volume = data['Volume'][-1]


        #defining 4 cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric('Current price', str(round(current_price,3)))

        col2.metric('Prediction for tomorrow',str(predicted_price_one))

        col3.metric('Prediction for the day after tomorrow',str(predicted_price_two))

        col4.metric("Volume traded last 24h", str(volume))

        #Candlestick
        st.subheader("""Candlestick plot for """ + selected_stock)

        data_1 = [go.Candlestick(x=data.index,
                               open=data.Open,
                               high=data.High,
                               low=data.Low,
                               close=data.Close)]
        layout_1 = go.Layout(title=data.name + ' Candlestick')
        fig_1 = go.Figure(data=data_1, layout=layout_1)
        st.plotly_chart(fig_1)
        if show_macd:
            #MACD graph
            st.subheader("""MACD plot for """ + selected_stock)
            fig_2 = moving_average(data)
            st.plotly_chart(fig_2)

        if show_fibonacci:
            #Fibo Graph
            st.subheader("""Fibonacci plot for """ + selected_stock)
            fig_3 = fib_retrace(data)
            st.write(fig_3)


        if show_rsi:
            st.subheader("""RSI Analysis""")
            fig_4 = RSIgraph(data)
            st.plotly_chart(fig_4)


    #Select the coin
    st.sidebar.subheader("""Asset Dashboard""")
    selected_stock = st.sidebar.text_input("Enter a valid asset name...", "BTC")
    button_clicked = st.sidebar.button("Select Asset")
    if button_clicked == "Select Asset":
        main()
        
    show_macd = st.sidebar.checkbox('MACD Analysis')
    show_fibonacci = st.sidebar.checkbox('Fibonacci Analysis')
    show_rsi = st.sidebar.checkbox('RSI Analysis')
    

  
    
elif page == "Company Investments":
    st.title(page)
    def main():
        st.write('teste')

 
if __name__ == "__main__":
    main()
