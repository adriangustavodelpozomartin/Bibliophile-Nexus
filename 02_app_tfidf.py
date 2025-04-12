import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el DataFrame que debe incluir las columnas "title" y "description"
df_tfidf = pd.read_csv('datasets/clean_description_english.csv')

# Cargar el vectorizador, la matriz TF-IDF y la matriz de similitud previamente guardados
with open('modelos/tfidf_vectorizer.pkl', 'rb') as file:
    tf_II = pickle.load(file)

with open('modelos/tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix_II = pickle.load(file)

with open('modelos/cosine_sim_II.pkl', 'rb') as file:
    cosine_sim_II = pickle.load(file)

# Crear las series para acceder rápidamente a los títulos y sus índices
titles = df_tfidf["title"]
indices = pd.Series(df_tfidf.index, index=df_tfidf["title"])

# Función para generar recomendaciones: devuelve título y descripción de los libros recomendados
def keywords_recommendations_II(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_II[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]  # Excluir el libro consultado
    book_indices = [i[0] for i in sim_scores]
    return df_tfidf.iloc[book_indices][["title", "clean_description"]]

# Título de la app
st.title("Recomendador Literario Basado en TF-IDF")

# Instrucciones para el usuario
st.write("Selecciona un libro para ver recomendaciones similares. Usa el cuadro desplegable para buscar entre todos los títulos disponibles:")

# Dropdown interactivo y con capacidad de búsqueda
selected_title = st.selectbox("Elige un libro:", titles.sort_values().unique())

# Al pulsar el botón, se muestran las recomendaciones
if st.button("Obtener recomendaciones"):
    recommendations = keywords_recommendations_II(selected_title)
    st.write(f"Recomendaciones para **{selected_title}**:")
    for index, row in recommendations.iterrows():
        st.write(f"**{row['title']}**")
        st.write(row["clean_description"])