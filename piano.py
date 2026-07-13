
import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd
import threading
import os
import tempfile
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Mezclador de audio (notas simultáneas) ──────────────────────────────────
SAMPLE_RATE = 44100
_active: list = []
_lock = threading.Lock()

def _callback(outdata, frames, time, status):
    with _lock:
        result = np.zeros(frames, dtype=np.float32)
        done = []
        for s in _active:
            rem = len(s['data']) - s['pos']
            if rem <= 0:
                done.append(s)
                continue
            n = min(frames, rem)
            result[:n] += s['data'][s['pos']:s['pos'] + n]
            s['pos'] += n
            if s['pos'] >= len(s['data']):
                done.append(s)
        for s in done:
            _active.remove(s)
        outdata[:, 0] = np.clip(result, -1.0, 1.0)

_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                           dtype='float32', callback=_callback, blocksize=512)
_stream.start()

def play_note(sound_array):
    """Añade la nota al mezclador; suena una vez y para sola."""
    with _lock:
        _active.append({'data': sound_array.copy(), 'pos': 0})

# ── Generador de notas (sin archivos externos) ───────────────────────────────
def generar_nota(freq, duracion=0.5):
    t = np.linspace(0, duracion, int(SAMPLE_RATE * duracion), False)
    wave = (np.sin(2 * np.pi * freq * t) * 0.55 +
            np.sin(2 * np.pi * freq * 2 * t) * 0.25 +
            np.sin(2 * np.pi * freq * 3 * t) * 0.12 +
            np.sin(2 * np.pi * freq * 4 * t) * 0.05)
    attack  = int(0.005 * SAMPLE_RATE)
    release = int(0.35  * SAMPLE_RATE)
    env = np.ones(len(wave))
    env[:attack]   = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return (wave * env * 0.7).astype(np.float32)

# Do Re Mi Fa Sol (mano 1) | La Si Do8 Re8 Mi8 (mano 2)
NOTAS      = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Si', 'Do8', 'Re8', 'Mi8']
FRECUENCIAS = [261,  294,  330,  349,  392,   440,  494,  523,   587,   659]
SONIDOS = [generar_nota(f) for f in FRECUENCIAS]

# ── MediaPipe ────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo HandLandmarker...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
hand_detector = mp_vision.HandLandmarker.create_from_options(_options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

WIDTH, HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

dedos_presionados = {}          # clave: (mano_id, dedo_id)
TIPS = [4, 8, 12, 16, 20]       # landmarks punta de dedo

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = hand_detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Dibujar landmarks y conexiones
            for a, b in HAND_CONNECTIONS:
                x1 = int(hand_landmarks[a].x * WIDTH); y1 = int(hand_landmarks[a].y * HEIGHT)
                x2 = int(hand_landmarks[b].x * WIDTH); y2 = int(hand_landmarks[b].y * HEIGHT)
                cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * WIDTH), int(lm.y * HEIGHT)), 4, (255, 255, 255), -1)
            for i, tip_id in enumerate(TIPS):
                tip = hand_landmarks[tip_id]
                pip = hand_landmarks[tip_id - 2 if tip_id != 4 else 2]

                x = int(tip.x * WIDTH)
                y = int(tip.y * HEIGHT)

                # El pulgar se dobla en el eje X, no en Y
                if tip_id == 4:
                    # Mano derecha (landmarks.x del pulgar aumenta al doblarse hacia la palma)
                    # Usamos la muñeca (punto 0) como referencia de orientación
                    wrist = hand_landmarks[0]
                    mcp   = hand_landmarks[2]   # base del pulgar
                    # Si la punta está más cerca de la palma que la base en X → doblado
                    flexionado = abs(tip.x - wrist.x) < abs(mcp.x - wrist.x) - 0.04
                else:
                    margen = 0.04
                    flexionado = tip.y > pip.y + margen
                clave      = (hand_idx, i)
                nota_idx   = (hand_idx * 5 + i) % len(SONIDOS)
                color      = (0, 255, 0)

                if flexionado:
                    color = (0, 0, 255)
                    if not dedos_presionados.get(clave, False):
                        play_note(SONIDOS[nota_idx])   # suena UNA vez y para
                        dedos_presionados[clave] = True
                else:
                    dedos_presionados[clave] = False

                cv2.circle(frame, (x, y), 20, color, -1)
                cv2.putText(frame, NOTAS[nota_idx], (x - 15, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Piano con Manos', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
_stream.stop()
_stream.close()
