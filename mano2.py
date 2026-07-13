###############################################################
# mano2.py
# Clase HandDetector: encapsula la detección de mano y gestos usando MediaPipe.
# Permite detectar la mano, dibujar los landmarks y analizar gestos como zoom, click, mano abierta/cerrada.
###############################################################

import mediapipe as mp  # Librería para visión por computadora y detección de manos
import cv2  # OpenCV para procesamiento de imágenes
import tempfile  # Manejo de archivos temporales
import os  # Operaciones del sistema operativo
import urllib.request  # Para descargar archivos desde internet
from mediapipe.tasks import python as mp_python  # Opciones base de MediaPipe
from mediapipe.tasks.python import vision as mp_vision  # Opciones de visión de MediaPipe

class HandDetector:
    # Clase para detectar manos y analizar gestos usando MediaPipe
    def is_shaka(self, hand_landmarks):
        # Detecta el gesto "shaka" (pulgar y meñique extendidos, otros dedos doblados)
        thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x  # Pulgar extendido
        pinky_extended = hand_landmarks[20].y < hand_landmarks[18].y  # Meñique extendido
        index_folded = hand_landmarks[8].y > hand_landmarks[6].y  # Índice doblado
        middle_folded = hand_landmarks[12].y > hand_landmarks[10].y  # Medio doblado
        ring_folded = hand_landmarks[16].y > hand_landmarks[14].y  # Anular doblado
        return thumb_extended and pinky_extended and index_folded and middle_folded and ring_folded  # True si es shaka

    def __init__(self):
        # Inicializa el detector de manos descargando el modelo si es necesario
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"  # URL del modelo
        MODEL_PATH = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")  # Ruta local temporal
        if not os.path.exists(MODEL_PATH):  # Si el modelo no existe localmente
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)  # Descarga el modelo
        # Configura el detector para detectar hasta 2 manos con umbrales de confianza
        #almacena la configuración necesaria para que el detector de manos sepa qué archivo de modelo cargar y usar internamente.
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)  # Opciones base
        #Este objeto almacena todos los parámetros necesarios para indicar cómo debe funcionar el detector de manos
        #opciones de mi optencion de manos.
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,  # Modelo cargado
            num_hands=2,  # Detectar hasta 2 manos
            min_hand_detection_confidence=0.7,  # Confianza mínima para detectar
            min_hand_presence_confidence=0.7,  # Confianza mínima de presencia
            min_tracking_confidence=0.7  # Confianza mínima de seguimiento
        )
        #encargado de recibir la configuración y crear el detector de manos utilizando el modelo especificado en las opciones.
        self.detector = mp_vision.HandLandmarker.create_from_options(options)  # Crea el detector
    #Se toma la diferencia en x y en y entre ambos puntos, se calcula el valor absoluto de cada diferencia y se suman.

    def is_zoom_start(self, hand_landmarks):
        # Detecta el inicio del gesto de zoom (pulgar e índice juntos)
        #usaremos el "ejemplo hand_landmarks[4]"es el valor de la posición horizontal de ese punto, normalizado entre 0 y 1 respecto al ancho de la imagen.
        #basado en el eje X y eje Y, se calcula la distancia en pulgar y el índice. Si esta distancia es menor a un umbral (0.07), se considera que el gesto de zoom está activo.
        dist = abs(hand_landmarks[4].x - hand_landmarks[8].x) + abs(hand_landmarks[4].y - hand_landmarks[8].y)  # Distancia Manhattan
        return dist < 0.07  # True si están suficientemente juntos

    def is_zoom_active(self, hand_landmarks):

        # Detecta si el gesto de zoom está activo (pulgar e índice separados)
        dist = abs(hand_landmarks[4].x - hand_landmarks[8].x) + abs(hand_landmarks[4].y - hand_landmarks[8].y)  # Distancia Manhattan
        return dist > 0.13  # True si están suficientemente separados

    def is_index_click(self, hand_landmarks):
        # Detecta el gesto de "click" con el índice (solo el índice extendido)
        return (
            hand_landmarks[8].y < hand_landmarks[6].y and  # Índice extendido
            hand_landmarks[12].y > hand_landmarks[10].y and  # Medio doblado
            hand_landmarks[16].y > hand_landmarks[14].y and  # Anular doblado
            hand_landmarks[20].y > hand_landmarks[18].y  # Meñique doblado
        )

    def detectar_mano(self, frame):
        # Procesa la imagen de la cámara y retorna una lista de landmarks de las manos detectadas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte la imagen a RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # Crea objeto imagen para MediaPipe
        results = self.detector.detect(mp_image)  # Detecta manos
        if results.hand_landmarks:
            return results.hand_landmarks  # Devuelve lista de landmarks de manos detectadas
        return []  # Si no hay manos, devuelve lista vacía

    def dibujar_mano(self, frame, hand_landmarks):
        # Dibuja los puntos (landmarks) y conexiones de la mano sobre la imagen
        #H altura,W ancho y _ para asignar colores,no esta activo.
        h, w, _ = frame.shape  # Obtiene dimensiones de la imagen
        for lm in hand_landmarks:
            x = int(lm.x * w)  # Coordenada x del landmark
            y = int(lm.y * h)  # Coordenada y del landmark
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Dibuja círculo verde en cada landmark
        # Lista de conexiones entre puntos para visualizar la estructura de la mano
        #usamos una lista para dibujar Cada tupla, como (0,1), indica que se debe dibujar una línea entre el landmark 0 y el landmark 1. Así se unen los puntos de cada dedo
        conexiones = [
            (0,1),(1,2),(2,3),(3,4),      # Pulgar
            (0,5),(5,6),(6,7),(7,8),      # Índice
            (5,9),(9,10),(10,11),(11,12), # Medio
            (9,13),(13,14),(14,15),(15,16), # Anular
            (13,17),(17,18),(18,19),(19,20), # Meñique
            (0,17)
        ]
        #ahora las dibujamos
        #Para cada conexión definida en la lista, se obtienen las coordenadas de los puntos correspondientes a los landmarks y se dibuja una línea azul entre ellos usando cv2.line. Esto crea una representación visual de la estructura de la mano en la imagen.
        for c in conexiones:
            x1 = int(hand_landmarks[c[0]].x * w)  # Punto inicial
            y1 = int(hand_landmarks[c[0]].y * h)
            x2 = int(hand_landmarks[c[1]].x * w)  # Punto final
            y2 = int(hand_landmarks[c[1]].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Dibuja línea azul entre puntos
        return frame  # Devuelve la imagen con la mano dibujada

    def is_hand_open(self, hand_landmarks):
        # Detecta si la mano está abierta (4 o más dedos extendidos)
        #con esta parte detectamos la mano a bierta y cuantos dedos tiene estirado.
        fingers = [8, 12, 16, 20]  # Índices de las puntas de los dedos
        open_fingers = 0
        #recorre dedos extendidos,compara la posicion de la punta del dedo.
        for tip in fingers:
            if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:  # Si la punta está arriba de la articulación
                open_fingers += 1
        # Verifica si el pulgar está extendido
        if hand_landmarks[4].x > hand_landmarks[3].x:
            open_fingers += 1
        return open_fingers >= 4  # True si 4 o más dedos están extendidos
    #sabemos que esta cerrada la mano por el dato de si esta por arriba o bajo de la articulacion,si esta por debajo es que esta doblada y si esta por arriba es que esta estirada,si el pulgar esta por la derecha del dedo anterior es que esta estirado y si esta por la izquierda es que esta doblado.
    def is_hand_closed(self, hand_landmarks):
        # Detecta si la mano está cerrada (3 o más dedos doblados)
        fingers = [8, 12, 16, 20]  # Índices de las puntas de los dedos
        closed_fingers = 0
        for tip in fingers:
            if hand_landmarks[tip].y > hand_landmarks[tip - 2].y:  # Si la punta está abajo de la articulación
                closed_fingers += 1
        return closed_fingers >= 3  # True si 3 o más dedos están doblados

    def distancia_pulgar_indice(self, hand_landmarks, w, h):
        # Calcula la distancia euclidiana entre la punta del pulgar y del índice
        x_thumb = int(hand_landmarks[4].x * w)  # Coordenada x del pulgar
        y_thumb = int(hand_landmarks[4].y * h)  # Coordenada y del pulgar
        x_index = int(hand_landmarks[8].x * w)  # Coordenada x del índice
        y_index = int(hand_landmarks[8].y * h)  # Coordenada y del índice
        # Retorna la distancia euclidiana entre ambos puntos
        return ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5
