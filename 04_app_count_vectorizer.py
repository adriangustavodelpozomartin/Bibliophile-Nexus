import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ------------------------------------------------------------------------------
# 1. Función para cargar el dataset y cachearlo
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df.copy()

# ------------------------------------------------------------------------------
# 2. Función para preparar la matriz de conteos, la similitud coseno y el índice de títulos
# ------------------------------------------------------------------------------
def prepare_recommendations(df):
    # Asegúrate de que la columna 'genre' no contenga valores nulos
    df['genre'] = df['genre'].fillna('')
    
    # Vectorización con CountVectorizer (stopwords en inglés)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['genre'])
    
    # Calcular la matriz de similitud coseno
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Crear un objeto tipo Series que mapea cada título a su índice (se asume que la columna 'title' existe)
    indices = pd.Series(df.index, index=df['title'])
    
    return count_matrix, cosine_sim, indices

# ------------------------------------------------------------------------------
# 3. Función para obtener recomendaciones basadas en la similitud coseno
# ------------------------------------------------------------------------------
def get_recommendations(title, cosine_sim, indices, df, top_n=10):
    try:
        idx = indices[title]
    except KeyError:
        return None, None
    # Calcular la similitud entre el libro y el resto
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Ordenar en orden descendente de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Excluir el mismo libro (índice 0) y tomar los siguientes top_n
    sim_scores = sim_scores[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    recommended_titles = df['title'].iloc[recommended_indices].values
    return recommended_titles, recommended_indices

# ------------------------------------------------------------------------------
# 4. Función para obtener las coordenadas 3D a partir de la matriz de conteos
# ------------------------------------------------------------------------------
def compute_3d_coords(count_matrix, n_components=3):
    svd = TruncatedSVD(n_components=n_components)
    coords = svd.fit_transform(count_matrix)
    return coords

# ------------------------------------------------------------------------------
# 5. Función para generar el gráfico 3D mostrando solamente el libro de entrada y las recomendaciones
# ------------------------------------------------------------------------------
def plot_3d_only_recommendations(coords, df, input_idx, recommended_indices):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1) Graficar el libro de entrada en rojo
    ax.scatter(
        coords[input_idx, 0], coords[input_idx, 1], coords[input_idx, 2],
        c='red', s=120, label='Libro de Entrada'
    )
    ax.text(
        coords[input_idx, 0], coords[input_idx, 1], coords[input_idx, 2],
        df['title'].iloc[input_idx], color='red', size=10
    )
    
    # 2) Graficar los libros recomendados en verde
    for i, idx in enumerate(recommended_indices):
        label = 'Recomendado' if i == 0 else None
        ax.scatter(
            coords[idx, 0], coords[idx, 1], coords[idx, 2],
            c='green', s=80, alpha=0.8, label=label
        )
        ax.text(
            coords[idx, 0], coords[idx, 1], coords[idx, 2],
            df['title'].iloc[idx], color='green', size=9
        )
    
    ax.set_title("Visualización 3D: Solo Libro de Entrada y sus Recomendaciones")
    ax.legend()
    return fig

# ------------------------------------------------------------------------------
# 6. Función principal de la aplicación
# ------------------------------------------------------------------------------
def main():
    st.title("Recomendador Literario")
    st.write("Ingresa el título de un libro para obtener recomendaciones basadas en la similitud de géneros. Además, se muestra una visualización 3D que resalta el libro de entrada y sus recomendaciones.")
    
    # Ruta al dataset
    dataset_path = "datasets/clean_description_english.csv"  # Asegúrate de que el archivo CSV tenga al menos las columnas 'title' y 'genre'
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return
    
    # Mostrar el dataset (opcional)
    with st.expander("Mostrar vista previa del dataset"):
        st.write(df.head())
    
    # Preparar la matriz de conteos, matriz de similitud y mapeo de títulos
    count_matrix, cosine_sim, indices = prepare_recommendations(df)
    
    # Calcular las coordenadas 3D para la visualización
    coords = compute_3d_coords(count_matrix)
    
    # Entrada del usuario: título del libro (la búsqueda es sensible a mayúsculas/minúsculas)
    title_input = st.text_input("Ingresa el título del libro:")
    
    if st.button("Obtener recomendaciones"):
        if not title_input.strip():
            st.error("Por favor, ingresa un título válido.")
        else:
            recommended_titles, recommended_indices = get_recommendations(title_input, cosine_sim, indices, df)
            if recommended_titles is None:
                st.error("El título ingresado no se encontró en el dataset.")
            else:
                st.subheader("Recomendaciones:")
                for rec in recommended_titles:
                    st.write(f"- {rec}")
                
                try:
                    input_idx = indices[title_input]
                except KeyError:
                    st.error("El título ingresado no se encontró en el dataset.")
                    return
                
                # Generar y mostrar el gráfico 3D con solo el libro de entrada y las recomendaciones
                fig = plot_3d_only_recommendations(coords, df, input_idx, recommended_indices)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
