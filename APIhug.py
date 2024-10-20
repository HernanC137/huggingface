import streamlit as st
from utils import cargar_modelo_sd, cargar_modelo_resnet, predecir_imagen
from PIL import Image

# Inicializar los modelos de generación y clasificación
cargar_modelo_sd()
cargar_modelo_resnet()

col1, col2 = st.columns(2)

with col1:
    st.header("Generación de imágenes")
    descripcion = st.text_area("Escribe una descripción para la imagen")

    imagen_generada = None
    if st.button("Generar Imagen"):
        try:
            pipe = st.session_state["modelo_generacion"]
            imagen_generada = pipe(descripcion).images[0]

            st.image(imagen_generada, caption="Imagen Generada", use_column_width=True)

            st.session_state["imagen_generada"] = imagen_generada

        except Exception as e:
            st.error(f"Error durante la generación: {str(e)}")

    st.subheader("Clasificación de la Imagen Generada")
    if "imagen_generada" in st.session_state:
        if st.button("Clasificar Imagen Generada"):
            try:
                imagen = st.session_state["imagen_generada"]
                resultado = predecir_imagen(imagen)
                st.success(f"Clasificación: {resultado}")
            except Exception as e:
                st.error(f"Error durante la clasificación: {str(e)}")
    else:
        st.warning("Primero genera una imagen para poder clasificarla.")

with col2:
    st.header("Clasificación de una Imagen")
    archivo = st.file_uploader("Subir una imagen", type=["png", "jpg", "jpeg"])

    if archivo is not None:
        if st.button("Clasificar Imagen"):
            try:
                imagen_subida = Image.open(archivo)
                st.image(imagen_subida, caption="Imagen Subida", use_column_width=True)
                resultado = predecir_imagen(imagen_subida)
                st.success(f"Clasificación: {resultado}")
            except Exception as e:
                st.error(f"Error durante la clasificación: {str(e)}")
    else:
        st.warning("Sube una imagen para poder clasificarla.")
