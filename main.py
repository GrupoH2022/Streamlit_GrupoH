import streamlit as st
from multiapp import MultiApp
from modelos import SVR,LSTM,RL,HOME
from PIL import Image
# from despliegue import modelo_lstm, modelo_arima, modelo_decision_tree, modelo_prophet,  modelo_svr

img = Image.open("Captura.PNG")
st.image(img, width=1000 )


app = MultiApp()
st.title("Equipo H - Semana 12 ")


# Add all your application here
app.add_app("HOME", HOME.app)
app.add_app("SVR", SVR.app)
app.add_app("LSTM", LSTM.app)
app.add_app("Regessor Lineal", RL.app)
 
# The main app
app.run()