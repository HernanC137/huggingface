import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
import streamlit as st
from PIL import Image

# Cargar el modelo de generación de imágenes
def cargar_modelo_generacion():
    if "modelo_generacion" not in st.session_state:
        model_id = "CompVis/stable-diffusion-v1-4"  # Cambiar a un modelo válido
        dispositivo = "cpu"
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if dispositivo == "cuda" else torch.float32
        )
        pipeline = pipeline.to(dispositivo)
        st.session_state["modelo_generacion"] = pipeline

# Cargar el modelo de clasificación de imágenes
def cargar_modelo_clasificacion():
    if "modelo_clasificacion" not in st.session_state:
        procesador = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        modelo = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        st.session_state["modelo_clasificacion"] = (procesador, modelo)

# Función para generar una imagen basada en texto
def generar_imagen(solicitud):
    try:
        pipeline = st.session_state["modelo_generacion"]
        imagen_generada = pipeline(solicitud).images[0]
        return imagen_generada
    except Exception as e:
        st.error(f"Error al generar la imagen: {str(e)}")
        return None

# Función para clasificar una imagen
def clasificar_imagen(image):
    try:
        procesador, modelo = st.session_state["modelo_clasificacion"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = procesador(image, return_tensors="pt")
        with torch.no_grad():
            logits = modelo(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        return modelo.config.id2label[predicted_label]
    except Exception as e:
        st.error(f"Error al clasificar la imagen: {str(e)}")
        return None
