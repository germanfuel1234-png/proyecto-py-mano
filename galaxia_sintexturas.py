import sys
import cv2
import numpy as np
# --- Importar la clase HandDetector de mano2.py ---
from mano2 import HandDetector
from vispy import app, scene
from vispy.scene import visuals
import sys
import cv2
import numpy as np
# --- Importar la clase HandDetector de mano2.py ---
from mano2 import HandDetector
from vispy import app, scene
from vispy.scene import visuals

# --- Parámetros de la escena 3D ---
N_STARS = 2000
RADIUS = 10

# Generar posiciones aleatorias para las estrellas
np.random.seed(42)
theta = np.random.uniform(0, 2 * np.pi, N_STARS)
phi = np.random.uniform(0, np.pi, N_STARS)
r = RADIUS * np.random.uniform(0.7, 1.0, N_STARS)
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
stars = np.vstack((x, y, z)).T


# --- Inicializar el detector de mano ---
detector_mano = HandDetector()

# --- Ventana Vispy ---
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 600))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=0, elevation=30, distance=30)


# Dibujar las estrellas como un solo Mesh tipo PointCloud
from vispy.scene.visuals import Mesh
def crear_pointcloud_mesh(puntos, size=0.08):
    # Crea una esfera pequeña por cada punto
    vertices = []
    faces = []
    offset = 0
    # Esfera base de baja resolución
    phi, theta = np.mgrid[0:np.pi:3j, 0:2*np.pi:6j]
    base_x = size * np.sin(phi) * np.cos(theta)
    base_y = size * np.sin(phi) * np.sin(theta)
    base_z = size * np.cos(phi)
    base_v = np.stack([base_x.flatten(), base_y.flatten(), base_z.flatten()], axis=1)
    base_f = np.array([
        [0,1,2],[0,2,3],[0,3,4],[0,4,5],
        [0,5,1],[1,6,2],[2,7,3],[3,8,4],[4,9,5],[5,10,1],
        [6,7,2],[7,8,3],[8,9,4],[9,10,5],[10,6,1]
    ])
    for px, py, pz in puntos:
        v = base_v + np.array([px, py, pz])
        f = base_f + offset
        vertices.append(v)
        faces.append(f)
        offset += v.shape[0]
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    return vertices, faces

vertices, faces = crear_pointcloud_mesh(stars, size=0.08)
mesh_estrellas = Mesh(vertices=vertices, faces=faces, color='white', parent=view.scene)

# --- Captura de cámara ---
cap = cv2.VideoCapture(0)

# --- Buffer para estelas de rayos (en 2D, sobre la cámara) ---
NUM_DEDOS = 5
HISTORIA_ESTELAS = 15  # Cuántos frames dura la estela
estelas = [[] for _ in range(NUM_DEDOS)]  # Una lista por dedo

# Variables para control de cámara
last_hand_pos = None
last_zoom_dist = None

# --- Función de actualización ---
def update(event):
    global last_hand_pos, last_zoom_dist
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    hand_landmarks_list = detector_mano.detectar_mano(frame)
    h, w, _ = frame.shape
    # Dibuja todas las manos detectadas
    for hand_landmarks in hand_landmarks_list:
        frame = detector_mano.dibujar_mano(frame, hand_landmarks)

    # --- Rayos mágicos y estelas sobre la imagen de la cámara ---
    dedos = [4, 8, 12, 16, 20]
    colores_bgr = [
        (0,0,255),    # rojo (pulgar)
        (0,255,255),  # amarillo (índice)
        (0,255,0),    # verde (medio)
        (255,255,0),  # cyan (anular)
        (211,0,148)   # violeta (meñique)
    ]
    if len(hand_landmarks_list) == 2:
        mano1 = hand_landmarks_list[0]
        mano2 = hand_landmarks_list[1]
        for i, d in enumerate(dedos):
            x1 = int(mano1[d].x * w)
            y1 = int(mano1[d].y * h)
            x2 = int(mano2[d].x * w)
            y2 = int(mano2[d].y * h)
            # Añadir a la historia de la estela
            estelas[i].append(((x1, y1), (x2, y2)))
            if len(estelas[i]) > HISTORIA_ESTELAS:
                estelas[i].pop(0)
        # Dibujar estelas (de más viejo a más nuevo, más transparente)
        for i, tray in enumerate(estelas):
            for j, seg in enumerate(tray):
                alpha = int(255 * (j+1) / HISTORIA_ESTELAS)
                color = tuple(int(c * (j+1) / HISTORIA_ESTELAS + 30) for c in colores_bgr[i])
                overlay = frame.copy()
                cv2.line(overlay, seg[0], seg[1], color, 10)
                frame = cv2.addWeighted(overlay, alpha/255.0, frame, 1 - alpha/255.0, 0)
        # Dibujar el rayo principal encima
        for i, d in enumerate(dedos):
            x1 = int(mano1[d].x * w)
            y1 = int(mano1[d].y * h)
            x2 = int(mano2[d].x * w)
            y2 = int(mano2[d].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), colores_bgr[i], 14)
    else:
        # Si no hay dos manos, limpiar estelas
        for tray in estelas:
            tray.clear()

    # --- Mira y gestos solo para la primera mano (si existe) ---
    if hand_landmarks_list:
        hand_landmarks = hand_landmarks_list[0]
        x_mira = int(hand_landmarks[8].x * w)
        y_mira = int(hand_landmarks[8].y * h)
        overlay = frame.copy()
        cv2.circle(overlay, (x_mira, y_mira), 22, (0, 255, 255), -1)  # Amarillo
        alpha = 0.25
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        # --- Lógica de gestos para la primera mano ---
        if detector_mano.is_zoom_start(hand_landmarks):
            print("[GESTO] Pinza (zoom) detectado")
            y_centro = hand_landmarks[0].y
            if last_zoom_dist is not None:
                dy = y_centro - last_zoom_dist
                view.camera.distance += dy * 20
                view.camera.distance = max(5, min(50, view.camera.distance))
                print(f"[CAMARA] Zoom: distance={view.camera.distance:.2f}, dy={dy:.4f}")
                view.camera.view_changed()
            last_zoom_dist = y_centro
        elif detector_mano.is_hand_closed(hand_landmarks):
            print("[GESTO] Puño cerrado (pan) detectado")
            cx = int(hand_landmarks[0].x * w)
            cy = int(hand_landmarks[0].y * h)
            if last_hand_pos is not None:
                dx = cx - last_hand_pos[0]
                dy = cy - last_hand_pos[1]
                view.camera.azimuth -= dx * 2
                view.camera.elevation -= dy * 2
                print(f"[CAMARA] Pan: azimuth={view.camera.azimuth:.2f}, elevation={view.camera.elevation:.2f}, dx={dx}, dy={dy}")
                view.camera.view_changed()
            last_hand_pos = (cx, cy)
            last_zoom_dist = None
        elif detector_mano.is_hand_open(hand_landmarks):
            print("[GESTO] Mano abierta detectada (sin acción)")
            last_hand_pos = None
            last_zoom_dist = None
        else:
            last_hand_pos = None
            last_zoom_dist = None
    else:
        last_hand_pos = None
        last_zoom_dist = None
    # Mostrar la imagen con la mano dibujada y la mira translúcida
    cv2.imshow('Mano', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        app.quit()
    canvas.update()
# --- Timer para refrescar la escena y leer la cámara ---
timer = app.Timer()
timer.connect(update)
timer.start(0.03)

# Ejecutar la app principal
if __name__ == '__main__':
    app.run()

# --- Función de actualización ---
def update(event):
    global last_hand_pos, last_zoom_dist
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    hand_landmarks = detector_mano.detectar_mano(frame)
    h, w, _ = frame.shape
    if hand_landmarks:
        # Dibuja la mano sobre la imagen
        frame = detector_mano.dibujar_mano(frame, hand_landmarks)

        # --- Calcular posición de la mira (índice) ---
        x_mira = int(hand_landmarks[8].x * w)
        y_mira = int(hand_landmarks[8].y * h)

        # --- Dibuja la mira translúcida ---
        overlay = frame.copy()
        cv2.circle(overlay, (x_mira, y_mira), 22, (0, 255, 255), -1)  # Amarillo
        alpha = 0.25  # Transparencia muy suave
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # --- Lógica de gestos exclusiva ---
        # 1. Zoom (pinza)
        if detector_mano.is_zoom_start(hand_landmarks):
            print("[GESTO] Pinza (zoom) detectado")
            zoom_dist = detector_mano.distancia_pulgar_indice(hand_landmarks, w, h)
            if last_zoom_dist is not None:
                dz = zoom_dist - last_zoom_dist
                view.camera.distance -= dz * 0.05
                view.camera.distance = max(5, min(50, view.camera.distance))
                print(f"[CAMARA] Zoom: distance={view.camera.distance:.2f}")
                view.camera.view_changed()  # Forzar actualización
            last_zoom_dist = zoom_dist
            try:
                print("[DEBUG] Intentando leer frame de la cámara...")
                ret, frame = cap.read()
                print(f"[DEBUG] Frame leído: ret={ret}, frame shape={frame.shape if ret else None}")
                if not ret:
                    print("[ERROR] No se pudo leer frame de la cámara. Cerrando update.")
                    return
                frame = cv2.flip(frame, 1)
                hand_landmarks = detector_mano.detectar_mano(frame)
                h, w, _ = frame.shape
                if hand_landmarks:
                    # Dibuja la mano sobre la imagen
                    frame = detector_mano.dibujar_mano(frame, hand_landmarks)

                    # --- Calcular posición de la mira (índice) ---
                    x_mira = int(hand_landmarks[8].x * w)
                    y_mira = int(hand_landmarks[8].y * h)

                    # --- Dibuja la mira translúcida ---
                    overlay = frame.copy()
                    cv2.circle(overlay, (x_mira, y_mira), 22, (0, 255, 255), -1)  # Amarillo
                    alpha = 0.25  # Transparencia muy suave
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # --- Lógica de gestos exclusiva ---
                    if detector_mano.is_zoom_start(hand_landmarks):
                        zoom_dist = detector_mano.distancia_pulgar_indice(hand_landmarks, w, h)
                        if last_zoom_dist is not None:
                            dz = zoom_dist - last_zoom_dist
                            view.camera.distance -= dz * 0.05
                            view.camera.distance = max(5, min(50, view.camera.distance))
                            print(f"[CAMARA] Zoom: distance={view.camera.distance:.2f}")
                            view.camera.view_changed()  # Forzar actualización
                        last_zoom_dist = zoom_dist
                        last_hand_pos = None
                    elif detector_mano.is_hand_closed(hand_landmarks):
                        print("[GESTO] Puño cerrado (pan) detectado")
                        cx = int(hand_landmarks[0].x * w)
                        cy = int(hand_landmarks[0].y * h)
                        if last_hand_pos is not None:
                            dx = cx - last_hand_pos[0]
                            dy = cy - last_hand_pos[1]
                            view.camera.azimuth -= dx * 2  # Mucho más sensible
                            view.camera.elevation -= dy * 2
                            print(f"[CAMARA] Pan: azimuth={view.camera.azimuth:.2f}, elevation={view.camera.elevation:.2f}, dx={dx}, dy={dy}")
                            view.camera.view_changed()  # Forzar actualización
                        last_hand_pos = (cx, cy)
                        last_zoom_dist = None
                    elif detector_mano.is_hand_open(hand_landmarks):
                        print("[GESTO] Mano abierta detectada (sin acción)")
                        # Aquí podrías poner lógica de rotación especial si lo deseas
                        last_hand_pos = None
                        last_zoom_dist = None
                    else:
                        last_hand_pos = None
                        last_zoom_dist = None
                else:
                    last_zoom_dist = None
                # Mostrar la imagen con la mano dibujada y la mira translúcida
                cv2.imshow('Mano', frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                    app.quit()
                else:
                    last_hand_pos = None
                    last_zoom_dist = None
                canvas.update()
            except Exception as e:
                import traceback
                print("\n[ERROR EN UPDATE]:", e)
                traceback.print_exc()
