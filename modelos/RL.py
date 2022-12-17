import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from datetime import date
import yfinance as yf 
from pandas_datareader import data as pdr
import yfinance as yfin
from sklearn.model_selection import train_test_split
from plotly import graph_objs as go
from PIL import Image

def app():
      
  st.title('Model - SVR')
  start = st.date_input('Start Train' , value=pd.to_datetime('2014-1-1'))
  end = st.date_input('End Train' , value=pd.to_datetime('2018-12-30'))
  user_input = st.text_input('Yahoo finance' , 'ETH-USD')
  df = yf.download(user_input, start, end)
  st.title('Model - K-Means')
  st.subheader('Clasificación de acciones')
 
  st.write(df)
  
  days = list()
  df_days = df.index
  for day in df_days:
        days.append([int(day.strftime('%d'))])
  
 
  
  
  #GRAFICADMOS
  
  fig3 = go.Figure()
  fig3.add_trace(go.Scatter(x=df_days, y = df['Volume'], name='Adj_Close'))
  fig3.layout.update(title_text="Volumen" )
  st.plotly_chart(fig3) 
  
  fig2 = go.Figure()
  fig2.add_trace(go.Scatter(x=df_days, y = df['Close'], name='Close'))
  fig2.layout.update(title_text="Close" )
  st.plotly_chart(fig2) 
  
  #Entrenamiento del Modelo
  st.title('Predicciones vs y_test')
  img = Image.open("RL1.png")
  st.image(img, width=700 )
 

