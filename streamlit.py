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


#function calling local css sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local css sheet
local_css("style.css")

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

#main function
def page1():
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

    #MACD graph
    st.subheader("""MACD plot for """ + selected_stock)
    fig_2 = moving_average(data)
    st.plotly_chart(fig_2)


if page == "Asset Dashboard":

    #Select the coin
    st.sidebar.subheader("""Asset Dashboard""")
    selected_stock = st.sidebar.text_input("Enter a valid asset name...", "BTC")
    button_clicked = st.sidebar.button("Select Asset")
    if button_clicked == "Select Asset":
        page1()
        
    
elif page == "Company Investments":
    st.title(page)

    




if __name__ == "__main__":
    main()
