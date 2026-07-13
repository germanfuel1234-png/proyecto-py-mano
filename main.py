import cv2
import numpy as np
from mano2 import HandDetector
from vispy import app
import galax
from vispy import scene
from vispy.scene import visuals

def controlar_sistema_solar():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    last_hand_pos = None
    view = galax.view
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        hand_landmarks = detector.detectar_mano(frame)
        if hand_landmarks:
            frame = detector.dibujar_mano(frame, hand_landmarks)
            # --- GESTOS ---
            if detector.is_hand_open(hand_landmarks):
                controlar_sistema_solar.last_zoom_dist = None  # Solo aquí se reinicia
                last_hand_pos = None
                cv2.imshow('Mano', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            # --- ZOOM tipo ruedita de mouse ---
            zoom_dist = detector.distancia_pulgar_indice(hand_landmarks, w, h)
            if not hasattr(controlar_sistema_solar, "last_zoom_dist") or controlar_sistema_solar.last_zoom_dist is None:
                controlar_sistema_solar.last_zoom_dist = zoom_dist
            delta = zoom_dist - controlar_sistema_solar.last_zoom_dist
            if abs(delta) > 2:
                if delta > 0:
                    view.camera.distance -= abs(delta) * 0.01
                else:
                    view.camera.distance += abs(delta) * 0.01
                view.camera.distance = max(5, min(100, view.camera.distance))
                controlar_sistema_solar.last_zoom_dist = zoom_dist  # Solo actualiza si hubo gesto
            # --- ROTACIÓN de sistema solar ---
            cx = int(hand_landmarks[0].x * w)
            cy = int(hand_landmarks[0].y * h)
            if detector.is_index_click(hand_landmarks):
                if last_hand_pos is not None:
                    dx = cx - last_hand_pos[0]
                    dy = cy - last_hand_pos[1]
                    if abs(dx) > 2 or abs(dy) > 2:
                        view.camera.azimuth -= dx * 0.3
                        view.camera.elevation -= dy * 0.3
                last_hand_pos = (cx, cy)
            else:
                last_hand_pos = (cx, cy)
        else:
            # No reiniciar last_zoom_dist: así el zoom se mantiene aunque se pierda la mano
            last_hand_pos = None
        cv2.imshow('Mano', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from threading import Thread
    t = Thread(target=app.run, daemon=True)
    t.start()
    controlar_sistema_solar()
###############################################################
# main.py
# Ventana principal: muestra la galaxia y usa la cámara para detectar la mano y controlar la escena con gestos.
###############################################################

# Importaciones principales
import subprocess

# --- Parámetros de la escena 3D ---
# Número de estrellas y radio de la galaxia
N_STARS = 2000
RADIUS = 10
# Genera posiciones aleatorias para las estrellas en 3D
np.random.seed(42)
theta = np.random.uniform(0, 2 * np.pi, N_STARS)
phi = np.random.uniform(0, np.pi, N_STARS)
r = RADIUS * np.random.uniform(0.7, 1.0, N_STARS)
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
stars = np.vstack((x, y, z)).T

# --- Ventana Vispy ---
# Crea la ventana 3D donde se dibuja la galaxia
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 600))
view = canvas.central_widget.add_view()
# Cámara 3D que se puede mover con gestos
galaxy = visuals.Markers()
galaxy.set_data(stars, edge_color=None, face_color='white', size=2)
view.add(galaxy)
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=0, elevation=30, distance=30)

# --- Inicializar cámara y detector de mano ---
# Abre la cámara web
cap = cv2.VideoCapture(0)
# Crea el detector de mano (usa MediaPipe)
detector = HandDetector()
# Variables para guardar la última posición de la mano y el zoom
last_hand_pos = None
last_zoom_dist = None

###############################################################
# --- Función de actualización ---
# Esta función se llama periódicamente para:
# - Leer la cámara
# - Detectar la mano y los gestos
# - Dibujar la mano (opcional)
# - Controlar la galaxia con los gestos
###############################################################
def update(event):
    global last_hand_pos, last_zoom_dist
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    hand_landmarks = detector.detectar_mano(frame)
    # Variables para inercia del super scroll
    if not hasattr(update, "scroll_vel"):
        update.scroll_vel = [0, 0]
    scroll_friction = 0.92  # Ajusta para más/menos inercia

    if hand_landmarks:
        frame = detector.dibujar_mano(frame, hand_landmarks)

        # --- GESTOS ---
        if detector.is_hand_open(hand_landmarks):
            print("Mano abierta")
            # Detener super scroll y mantener zoom
            update.scroll_vel = [0, 0]
            update.last_zoom_dist = None  # No modificar el zoom
            last_hand_pos = None
            return  # Salir de update, no hacer nada más
        elif detector.is_hand_closed(hand_landmarks):
            print("Mano cerrada")

        # --- ZOOM tipo ruedita de mouse ---
        zoom_dist = detector.distancia_pulgar_indice(hand_landmarks, w, h)
        if not hasattr(update, "last_zoom_dist") or update.last_zoom_dist is None:
            update.last_zoom_dist = zoom_dist
        delta = zoom_dist - update.last_zoom_dist
        if abs(delta) > 2:
            if delta > 0:
                print("Zoom + (abrir pulgar e índice)")
                view.camera.distance -= abs(delta) * 0.03
            else:
                print("Zoom - (cerrar pulgar e índice)")
                view.camera.distance += abs(delta) * 0.03
            view.camera.distance = max(2, min(100, view.camera.distance))
            update.last_zoom_dist = zoom_dist

        # --- SUPER SCROLL tipo celular ---
        cx = int(hand_landmarks[0].x * w)
        cy = int(hand_landmarks[0].y * h)
        if detector.is_index_click(hand_landmarks):
            if last_hand_pos is not None:
                dx = cx - last_hand_pos[0]
                dy = cy - last_hand_pos[1]
                if abs(dx) > 2 or abs(dy) > 2:
                    update.scroll_vel[0] = -dx * 1.5
                    update.scroll_vel[1] = -dy * 1.5
            last_hand_pos = (cx, cy)
        else:
            last_hand_pos = (cx, cy)
    else:
        print("No se detecta mano")
        last_hand_pos = None
        # No modificar update.last_zoom_dist

    # Aplica inercia del super scroll siempre
    if abs(update.scroll_vel[0]) > 0.1 or abs(update.scroll_vel[1]) > 0.1:
        view.camera.azimuth += update.scroll_vel[0]
        view.camera.elevation += update.scroll_vel[1]
        update.scroll_vel[0] *= scroll_friction
        update.scroll_vel[1] *= scroll_friction
    else:
        update.scroll_vel = [0, 0]
    # Si quieres ver la mano dibujada, descomenta la siguiente línea:
    # cv2.imshow('Mano', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        app.quit()

# Timer para refrescar la escena y leer la cámara
# Llama a update() cada 0.03 segundos (~30 FPS)
timer = app.Timer()
timer.connect(update)
timer.start(0.03)

# --- Bloque principal ---
if __name__ == '__main__':
    # Ejecuta galax.py como sistema solar principal con texturas
    subprocess.run(["python", "galax.py"])
