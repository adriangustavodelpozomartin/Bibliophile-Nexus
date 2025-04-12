import streamlit as st
import pandas as pd

# ------------------------------------------------------------------------------
# 1. Función para cargar el CSV y cachearlo para no recargarlo cada vez
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    # Se trabaja sobre una copia para evitar problemas de modificación
    return df.copy()

# ------------------------------------------------------------------------------
# 2. Función de recomendación (basada en tu notebook)
# ------------------------------------------------------------------------------
def recommend_by_storyline(title, df):
    recommended = []
    top10_list = []
    
    # Convertir el título a minúsculas para hacer la búsqueda insensible a mayúsculas
    title_lower = title.lower()
    
    # Asegurarse de trabajar con la columna 'title' en minúsculas
    df['title'] = df['title'].str.lower()
    
    # Extraer el topic y número de documento del título ingresado
    topic_num = df[df['title'] == title_lower].Topic.values
    doc_num = df[df['title'] == title_lower].Doc.values

    # Si no se encuentra el título, devolvemos None
    if len(topic_num) == 0 or len(doc_num) == 0:
        return None

    # Filtrar todos los documentos que tienen el mismo topic y ordenarlos por probabilidad descendente
    output_df = df[df['Topic'] == topic_num[0]].sort_values('Probability', ascending=False).reset_index(drop=True)
    
    try:
        # Obtener el índice del documento que corresponde al título ingresado
        index = output_df[output_df['Doc'] == doc_num[0]].index[0]
    except IndexError:
        return None

    # Se toman 5 documentos anteriores y 5 posteriores (sin incluir el mismo)
    indices_anteriores = list(output_df.iloc[max(0, index-5):index].index)
    indices_posteriores = list(output_df.iloc[index+1:index+6].index)
    top10_list = indices_anteriores + indices_posteriores

    # Convertir el título a formato "Title Case" para mostrar de manera más legible
    output_df['title'] = output_df['title'].str.title()

    for i in top10_list:
        recommended.append(output_df.iloc[i].title)

    return recommended

# ------------------------------------------------------------------------------
# 3. Definición de la interfaz de Streamlit (UI)
# ------------------------------------------------------------------------------
def main():
    st.title("Recomendador Literario basado en LDA")
    st.write("Ingresa el título (storyline) de la película o historia para obtener recomendaciones basadas en la asignación de temas.")

    # Cargar el dataset (se asume que 'doc_topic_matrix.csv' está en el mismo directorio)
    df = load_dataset("datasets/doc_topic_matrix.csv")
    
    # Opción para ver una previsualización del dataset (útil para debug)
    if st.checkbox("Mostrar vista previa del dataset"):
        st.write(df.head())

    # Entrada de texto para el título
    title_input = st.text_input("Ingresa el título:")

    if st.button("Obtener recomendaciones"):
        if not title_input.strip():
            st.error("Por favor, ingresa un título válido.")
        else:
            recommendations = recommend_by_storyline(title_input, df)
            if recommendations is None or len(recommendations) == 0:
                st.error("El título ingresado no se encontró en el dataset o no se pudieron obtener recomendaciones.")
            else:
                st.subheader("Recomendaciones:")
                for rec in recommendations:
                    st.write(f"- {rec}")

if __name__ == "__main__":
    main()
