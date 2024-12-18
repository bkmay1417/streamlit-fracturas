import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from functools import lru_cache
import os
import gdown
import zipfile

# Ruta donde se descargará y descomprimirá el modelo
SAVED_MODEL_DIR = "modeloexportado/saved_model"

# Función para descargar y preparar el modelo desde Google Drive
@st.cache_resource
def download_model():
    url = "ENLACE_DE_GOOGLE_DRIVE"  # Sustituye por el enlace de Google Drive
    output = "modeloexportado.zip"

    if not os.path.exists("modeloexportado"):
        st.write("Descargando modelo...")
        gdown.download(url, output, quiet=False)
        st.write("Descomprimiendo modelo...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")
        st.write("Modelo listo para usar.")

# Función para cargar el modelo en memoria
@lru_cache(maxsize=1)
def load_model():
    print("Cargando modelo...")
    model = tf.saved_model.load(SAVED_MODEL_DIR)
    print("Modelo cargado exitosamente.")
    return model

# Función para cargar imágenes y convertirlas en arreglos de NumPy
def load_image_into_numpy_array(image):
    img = Image.open(image).convert('RGB')  # Asegurar que la imagen tiene 3 canales (RGB)
    return np.array(img, dtype=np.uint8)  # Convertir a uint8

# Interfaz de Streamlit
st.title("Detección de fracturas óseas")

# Descargar el modelo al iniciar
download_model()

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Cargar la imagen desde Streamlit
    image_np = load_image_into_numpy_array(uploaded_file)

    # Verificar las dimensiones de la imagen cargada
    st.write(f"Dimensiones de la imagen: {image_np.shape}")

    # Convertir la imagen a un tensor
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  # Agregar una dimensión para el lote

    # Cargar el modelo
    detect_fn = load_model()

    # Realizar la detección
    st.write("Realizando detección...")
    detections = detect_fn(input_tensor)

    # Procesar las detecciones
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # Filtrar resultados con alta confianza
    detection_classes = detections["detection_classes"].astype(np.int64)
    detection_boxes = detections["detection_boxes"]
    detection_scores = detections["detection_scores"]

    # Dibujar las cajas en la imagen
    image_np_with_detections = image_np.copy()
    for i in range(num_detections):
        if detection_scores[i] > 0.5:  # Umbral de confianza
            box = detection_boxes[i]
            y_min, x_min, y_max, x_max = box
            h, w, _ = image_np.shape
            start_point = (int(x_min * w), int(y_min * h))
            end_point = (int(x_max * w), int(y_max * h))

            # Dibujar rectángulo rojo para fractura
            color_box = (255, 0, 0)  # Rojo
            thickness = 2
            cv2.rectangle(image_np_with_detections, start_point, end_point, color_box, thickness)

            # Añadir texto "Fractura"
            text = "Fractura"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color_text = (255, 255, 255)  # Blanco
            thickness_text = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness_text)[0]
            text_x = start_point[0]
            text_y = start_point[1] - 10
            background_start = (text_x, text_y - text_size[1] - 4)
            background_end = (text_x + text_size[0] + 4, text_y + 4)
            cv2.rectangle(image_np_with_detections, background_start, background_end, color_box, -1)  # Fondo rojo
            cv2.putText(image_np_with_detections, text, (text_x, text_y), font, font_scale, color_text, thickness_text)

    # Mostrar la imagen con las detecciones
    st.image(image_np_with_detections, caption="Detecciones", use_column_width=True)
