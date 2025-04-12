import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np



# Cargar el modelo y el tokenizer y cachearlo para que no se recargue en cada interacción
@st.cache(allow_output_mutation=True)
def load_classifier():
    # Cargar el modelo entrenado
    model = load_model("modelos/LSTM_clasificador.h5")
    # Cargar el tokenizer guardado
    with open('modelos/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_classifier()

st.title("Clasificador Literario: Ficción vs. No Ficción")
st.write("Ingresa la descripción del libro (en inglés) y presiona el botón para clasificarlo.")

# Área de texto para ingresar la descripción
descripcion = st.text_area("Descripción del libro (en inglés)", height=200)

if st.button("Clasificar"):
    if descripcion.strip() != "":
        # Convertir la descripción a secuencia usando el tokenizer
        secuencia = tokenizer.texts_to_sequences([descripcion])
        # Aplicar padding a la secuencia
        secuencia_padded = pad_sequences(secuencia, maxlen=500)
        # Hacer la predicción
        prediccion = model.predict(secuencia_padded)[0][0]
        # Interpretar la predicción: 
        # Si la probabilidad es mayor o igual a 0.5 se clasifica como "Ficción",
        # de lo contrario, como "No ficción".
        clasificacion = "Ficción" if prediccion >= 0.5 else "No ficción"
        st.success(f"Resultado: {clasificacion} ({prediccion:.2f})")
    else:
        st.error("Por favor, ingresa una descripción.")
