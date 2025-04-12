import streamlit as st
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el dataset y construir la columna 'description'
df_hf = pd.read_csv('datasets/clean_description_english.csv')
df_hf["description"] = (
    'Title: ' + df_hf["title"] + '. ' +
    'Desc / Summary: ' + df_hf["clean_description"] + '. ' +
    'Average Rating: ' + df_hf["rating"].astype('str') + '. ' +
    'Categories: ' + df_hf["genre"] + '. ' +
    'Author: ' + df_hf["author"] + '.'
)

# Función para obtener recomendaciones basadas en la similitud coseno
def get_recommendations(query, embeddings, df, model, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Cargar los modelos y embeddings preentrenados desde archivos pickle
# Se asume que estos archivos han sido previamente guardados usando pickle.dump
with open('modelos/model_allMiniLM.pkl', 'rb') as f:
    model_allMiniLM = pickle.load(f)
with open('modelos/embeddings_allMiniLM.pkl', 'rb') as f:
    embeddings_allMiniLM = pickle.load(f)

with open('modelos/model_bertLarge.pkl', 'rb') as f:
    model_bertLarge = pickle.load(f)
with open('modelos/embeddings_bertLarge.pkl', 'rb') as f:
    embeddings_bertLarge = pickle.load(f)

with open('modelos/model_robertaLarge.pkl', 'rb') as f:
    model_robertaLarge = pickle.load(f)
with open('modelos/embeddings_robertaLarge.pkl', 'rb') as f:
    embeddings_robertaLarge = pickle.load(f)

# Construcción de la interfaz en Streamlit
st.title("Recomendador Literario - Comparativa de Modelos Preentrenados")
st.write("Introduce tu consulta (en inglés) en la siguiente caja de texto y obtén recomendaciones de libros basadas en diferentes modelos de embeddings.")

query = st.text_input("Introduce tu consulta:")

if st.button("Obtener Recomendaciones"):
    if query:
        st.header("Modelo: all-MiniLM-L6-v2")
        rec1 = get_recommendations(query, embeddings_allMiniLM, df_hf, model_allMiniLM, top_n=5)
        st.write(rec1[['title', 'description']])
        
        st.header("Modelo: bert-large-nli-mean-tokens")
        rec2 = get_recommendations(query, embeddings_bertLarge, df_hf, model_bertLarge, top_n=5)
        st.write(rec2[['title', 'description']])
        
        st.header("Modelo: all-roberta-large-v1")
        rec3 = get_recommendations(query, embeddings_robertaLarge, df_hf, model_robertaLarge, top_n=5)
        st.write(rec3[['title', 'description']])
    else:
        st.write("Por favor, introduce una consulta.")