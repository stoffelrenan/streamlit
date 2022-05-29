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


        #defining 4 cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric('Current price', str(round(current_price,3)))

        col2.metric('Prediction for tomorrow',str(predicted_price_one))

        col3.metric('Prediction for the day after tomorrow',str(predicted_price_two))

        col4.metric("Volume traded last 24h", "1234")

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
