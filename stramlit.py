import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Ruta del modelo exportado
SAVED_MODEL_DIR = "modeloexportado/saved_model"

# Cargar el modelo al iniciar la aplicación
@st.cache_resource
def load_model():
    print("Cargando modelo...")
    model = tf.saved_model.load(SAVED_MODEL_DIR)
    print("Modelo cargado exitosamente.")
    return model

# Convertir imagen a array numpy
def load_image_into_numpy_array(image):
    # Asegurar que la imagen tiene 3 canales (RGB)
    img = Image.open(image).convert('RGB')
    return np.array(img, dtype=np.uint8)  # Convertir a uint8

# Cargar el modelo globalmente
detect_fn = load_model()

# Interfaz de Streamlit
st.title("Detección de fracturas óseas")
uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Cargar la imagen desde Streamlit
    image_np = load_image_into_numpy_array(uploaded_file)

    # Verificar las dimensiones de la imagen cargada
    #st.write(f"Dimensiones de la imagen: {image_np.shape}")

    # Convertir la imagen a un tensor
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  # Agregar una dimensión para el lote

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

    # Determinar si hay fractura
    hay_fractura = any(detection_scores[i] > 0.5 for i in range(num_detections))
    mensaje = "Fractura detectada" if hay_fractura else "No se detectó fractura"
    color = "red" if hay_fractura else "green"

    # Dibujar las cajas en la imagen
    image_np_with_detections = image_np.copy()
    for i in range(num_detections):
        if detection_scores[i] > 0.5:  # Umbral de confianza
            box = detection_boxes[i]
            y_min, x_min, y_max, x_max = box
            h, w, _ = image_np.shape
            start_point = (int(x_min * w), int(y_min * h))
            end_point = (int(x_max * w), int(y_max * h))
            
            # Color y grosor de la caja
            color_box = (255, 0, 0)  # Rojo para las cajas
            thickness = 2
            
            # Dibujar la caja
            cv2.rectangle(image_np_with_detections, start_point, end_point, color_box, thickness)
            
            # Agregar el texto "Fractura" con fondo
            text = "Fractura"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            font_color = (255, 255, 255)  # Blanco para el texto
            bg_color = (255, 0, 0)  # Rojo para el fondo del texto

            # Calcular tamaño del texto y posición
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x, text_y = start_point[0], max(start_point[1] - 10, 10)
            text_w, text_h = text_size[0], text_size[1]
            
            # Dibujar el fondo del texto
            cv2.rectangle(image_np_with_detections, (text_x, text_y - text_h - 5), 
                        (text_x + text_w + 5, text_y + 5), bg_color, -1)
            
            # Dibujar el texto sobre el fondo
            cv2.putText(image_np_with_detections, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        # Mostrar la imagen con las detecciones
    st.image(image_np_with_detections, caption="Detecciones", use_column_width=True)

    # Mostrar el mensaje de diagnóstico
    st.markdown(f"<h3 style='color: {color};'>{mensaje}</h3>", unsafe_allow_html=True)
