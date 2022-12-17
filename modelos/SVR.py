import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.svm import SVR

import yfinance as yf 
 
 
from plotly import graph_objs as go
 
 
def app():
  st.title('Model - SVR')
   

  """Cargamos los datos"""
  from pandas_datareader import data as pdr
  import yfinance as yfin
  yfin.pdr_override()

  st.subheader("Obtener datos de Yahoo finance")
  start = st.date_input('Start' , value=pd.to_datetime('2022-11-1'))
  end = st.date_input('End' , value=pd.to_datetime('2022-11-30'))
  user_input = st.text_input('Yahoo finance' , 'GLU')

  df = pdr.get_data_yahoo(user_input, start, end)
  st.write(df)
  st.subheader("Detalles de los datos")
  #mostramos las filas y columnas
  st.write(df.shape)

 

  #obtener la ultima fila 
  st.write("obtener la ultima fila")
  actual_price = df.tail(1)
  st.write(actual_price)

  #preparar datos de entrenamiento  ek modelo sCR
  #obtener todo los datos execpto la ultima fila
  st.write("obtener todo los datos execpto la ultima fila")
  df = df.head(len(df)-1)
  # print los nuevos datos
  st.write(df)

  #create una lista vacia independiente y dependiente

  days = list()
  adj_close_prices = list()

  #obtener la fecha y el precio de cieree ajustado
  df_days = df.index
  df_adj_close = df.loc[:, 'Adj Close'].to_numpy().astype(float)

  #create el conjunto de datos independiente
  for day in df_days:
    days.append([int(day.strftime('%d'))])
  # days = [df_days]
  adj_close_prices = df_adj_close

  # #create el conjunto de datos dependientes
  # for adj_close_price in df_adj_close:
  #   adj_close_prices.append(adj_close_price)

  #
  # st.write(days)
  # st.write(adj_close_prices)

  #creamos el support vector regression models
  # Create and train a SVR model using un kernel lineal
   
 
  fig3 = go.Figure()
  fig3.add_trace(go.Scatter(x=df_days, y = df['Volume'], name='Adj_Close'))
  fig3.layout.update(title_text="Volumen" )
  st.plotly_chart(fig3) 
  
  fig2 = go.Figure()
  fig2.add_trace(go.Scatter(x=df_days, y = df['Open'], name='Adj_Close'))
  fig2.layout.update(title_text="Open" )
  st.plotly_chart(fig2) 
  #FIGURA 01
  fig4 = go.Figure()
  fig4.add_trace(go.Scatter(x=df_days, y = df_adj_close, name='adj_close_prices'))
  fig4.layout.update(title_text="Ajuste de cierre" )
  st.plotly_chart(fig4) 
  

  lin_svr = SVR(kernel='linear', C=1000.0)
  lin_svr.fit(days, adj_close_prices)

  poly_svr = SVR(kernel='poly', C=1000.0, degree = 2)
  poly_svr.fit(days, adj_close_prices)

  rbf_svr = SVR(kernel='rbf', C=1000.0, gamma = 0.15)
  rbf_svr.fit(days, adj_close_prices)

  fig = plt.figure(figsize=(16,8))
  plt.scatter(days, adj_close_prices, color = 'red', label = 'Data')
  plt.plot(days, rbf_svr.predict(days), color = 'green', label='RBF Modelo')
  plt.plot(days, poly_svr.predict(days), color = 'orange', label='Polynomial Modelo')
  plt.plot(days, lin_svr.predict(days), color = 'Blue', label='Linear Modelo')
  plt.legend()
  st.pyplot(fig)

  day = [[30]]

  st.write('The RBF SVR prediction', rbf_svr.predict(day))
  st.write('The Lineal SVR prediction', lin_svr.predict(day))
  st.write('The Polynomial SVR prediction', poly_svr.predict(day))
 


 
 


'''    
def plot_data():
    #FIGURA 01
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Open'], name='stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y = data['High'], name='stock_High'))
    fig.layout.update(title_text="Open x High" )
    st.plotly_chart(fig) 
    #FIGURA 02
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y = data['Low'], name='stock_Low'))
    fig2.add_trace(go.Scatter(x=data['Date'], y = data['Close'], name='stock_Close'))
    fig2.layout.update(title_text="Low x Close" )
    st.plotly_chart(fig2) 
    #FIGURA 03
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_days, y = df_adj_close, name='Adj_Close'))
    fig3.layout.update(title_text="Ajuste de cierre" )
    st.plotly_chart(fig3) 

'''


