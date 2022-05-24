import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime


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
    predicted_price = '33'
    st.write('Current price: ' + str(round(current_price,2)))
    st.write('Prediction for tomorrow: ' + str(round(predicted_price,2)))

    #get daily volume for searched ticker
    st.subheader("""Daily **volume** for """ + selected_stock)
    st.line_chart(data.Volume)


if __name__ == "__main__":
    main()
