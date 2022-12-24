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
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import string
import re 



from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize


import warnings
warnings.filterwarnings('ignore')

import time
import datetime

def app():
    url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-con-R/master/datos/'
    tweets_elon   = pd.read_csv(url + "datos_tweets_@elonmusk.csv")
    tweets_edlee  = pd.read_csv(url + "datos_tweets_@mayoredlee.csv")
    tweets_bgates = pd.read_csv(url + "datos_tweets_@BillGates.csv")
    st.title('Lectura de datos')
    st.write('Número de tweets @BillGates: ' + str(tweets_bgates.shape[0]))
    st.write('Número de tweets @mayoredlee: ' + str(tweets_edlee.shape[0]))
    st.write('Número de tweets @elonmusk: ' + str(tweets_elon.shape[0]))

    st.title("Parseo por Fechas")    
    tweets = pd.concat([tweets_elon, tweets_edlee, tweets_bgates], ignore_index=True)

    # Se seleccionan y renombran las columnas de interés
    tweets = tweets[['screen_name', 'created_at', 'status_id', 'text']]
    tweets.columns = ['autor', 'fecha', 'id', 'texto']

    # Parseo de fechas
    tweets['fecha'] = pd.to_datetime(tweets['fecha'])
    st.write(tweets.head(3))
    
    
    st.title("Distribución temporal de los tweets")    
    
    
    fig2=plt.figure(figsize=(9, 4))
    for autor in tweets.autor.unique():
        df_temp = tweets[tweets['autor'] == autor].copy()
        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'].dt.strftime('%Y-%m'))
        df_temp = df_temp.groupby(df_temp['fecha']).size()
        plt.plot(df_temp.index, df_temp.values, label=autor)

    plt.title("Número de tuits publicados por mes")
    plt.legend()

    # Mostrar el gráfico en la aplicación de Streamlit
    st.pyplot(fig2)
    
    st.title('Limpieza y tokenizacion')
    def limpiar_tokenizar(texto):
         
    
        # Se convierte todo el texto a minúsculas
        nuevo_texto = texto.lower()
        # Eliminación de páginas web (palabras que empiezan por "http")
        nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
        # Eliminación de signos de puntuación
        regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
        nuevo_texto = re.sub(regex , ' ', nuevo_texto)
        # Eliminación de números
        nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
        # Eliminación de espacios en blanco múltiples
        nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
        # Tokenización por palabras individuales
        nuevo_texto = nuevo_texto.split(sep = ' ')
        # Eliminación de tokens con una longitud < 2
        nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
        
        return(nuevo_texto)
   
    test = "Esto es 1 ejemplo de l'limpieza de6 TEXTO  https://localhost:5505 @ejemplo #textmining"
    st.write(test)
    st.write(limpiar_tokenizar(texto=test))
    
    st.title('Se aplica la función de limpieza y tokenización a cada tweet')
    
    tweets['texto_tokenizado'] = tweets['texto'].apply(lambda x: limpiar_tokenizar(x))
    st.write(tweets[['texto', 'texto_tokenizado']].head())
    
    st.title('ANALISIS DE SENTIMIENTOS')
     
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    #importamos librerias para analisis de sentimientos
     
    analizador = SentimentIntensityAnalyzer()
     
    
    def get_sentimiento(texto):
        
        #Esta función devuelve el sentimiento de un texto.
        
        polaridad=TextBlob(texto).sentiment.polarity
        if polaridad > 0:
            sentimiento= 'positivo'
        elif polaridad == 0:
            sentimiento= 'neutro'
        else:
            sentimiento= 'negativo'
        return sentimiento
    # Se aplica la función de análisis de sentimiento a cada tweet
    # ==============================================================================
    tweets['sentimiento']= tweets['texto'].apply(lambda x: get_sentimiento(x)[0])
    st.write(tweets.head())