import streamlit as st
import pandas as pd
import yfinance as yf

st.write("My First Streamlit Web App")

def get_data(currency):
    data = yf.download(tickers=currency+'-USD', period = '5y', interval = '1d')
    data.name=currency
    return data
  
option = st.selectbox(
'How would you like to be contacted?',
('BTC','LTC','LUNA1'))

if st.form_submit_button(label="Choose coin", help=None, on_click=None, args=None, kwargs=None):
  df = get_data(option)
