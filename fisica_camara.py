###############################################################
# fisica_camara.py
# Pelotas con física real que rebotan en CUALQUIER borde
# detectado por la cámara: cinta adhesiva, líneas dibujadas,
# tu mano, una regla, un libro… lo que sea.
#
# ESC      → salir
# ESPACIO  → añadir pelota
# R        → reiniciar pelotas
# D        → toggle debug (muestra bordes de colisión en verde)
###############################################################

import cv2
import numpy as np
import random
from collections import deque
from mano2 import HandDetector

# ── Parámetros de física ────────────────────────────────────
GRAVITY     = 0.38        # aceleración gravitatoria (px/frame²)
RESTITUTION = 0.70        # coeficiente de rebote (0=muerto, 1=perfecto)
FRICTION    = 0.96        # rozamiento al deslizar sobre superficie
BALL_RADIUS = 12

# ── Parámetros de detección de bordes ──────────────────────
CANNY_LO    = 35          # umbral bajo de Canny
CANNY_HI    = 110         # umbral alto de Canny
EDGE_DILATE = 7           # grosor en px de las superficies detectadas

# ── Interacción mano-pelota ────────────────────────────────
# Radio de cada "esfera" de dedo para colisión directa
FINGER_RADIUS   = 18              # px: tamaño físico de cada segmento de dedo
MAX_HAND_VEL    = 32.0            # límite de velocidad heredable
VELOCITY_BLEND  = 0.80            # fracción de velocidad de mano transferida al golpear

# ── Misc ────────────────────────────────────────────────────
MAX_BALLS   = 30
TRAIL_LEN   = 16
INIT_BALLS  = 10

COLORS = [
    (0, 220, 255), (0, 255, 120), (255, 80,   0), (255,   0, 180),
    (120, 255,  0), (0, 120, 255), (255, 210,  0), (200,   0, 255),
    (0, 255, 210), (255, 160,  0), (0, 200, 255), (160, 255,   0),
]


# ── Seguimiento de velocidad de la mano ─────────────────────
class HandTracker:
    """Calcula la velocidad suavizada del centro de la palma."""
    def __init__(self):
        self._prev = None
        self._vx_buf: deque = deque(maxlen=5)
        self._vy_buf: deque = deque(maxlen=5)
        self.vx = 0.0
        self.vy = 0.0

    def update(self, hand, w: int, h: int):
        # Centro de la palma = landmark 9
        cx = hand[9].x * w
        cy = hand[9].y * h
        if self._prev is not None:
            self._vx_buf.append(cx - self._prev[0])
            self._vy_buf.append(cy - self._prev[1])
            raw_vx = sum(self._vx_buf) / len(self._vx_buf)
            raw_vy = sum(self._vy_buf) / len(self._vy_buf)
            # Limitar velocidad máxima heredable
            mag = np.hypot(raw_vx, raw_vy)
            if mag > MAX_HAND_VEL:
                scale = MAX_HAND_VEL / mag
                raw_vx *= scale
                raw_vy *= scale
            self.vx, self.vy = raw_vx, raw_vy
        else:
            self.vx = self.vy = 0.0
        self._prev = (cx, cy)

    def reset(self):
        self._prev = None
        self._vx_buf.clear()
        self._vy_buf.clear()
        self.vx = self.vy = 0.0


# ── Clase Pelota ────────────────────────────────────────────
class Pelota:
    def __init__(self, x: float, y: float):
        self.x     = float(x)
        self.y     = float(y)
        self.vx    = random.uniform(-5.0,  5.0)
        self.vy    = random.uniform(-6.0, -1.0)
        self.color = random.choice(COLORS)
        self.trail: deque = deque(maxlen=TRAIL_LEN)

    # ── Física ──────────────────────────────────────────────
    def step(self, h: int, w: int, thick, gx_map, gy_map):
        self.vy += GRAVITY
        nx = self.x + self.vx
        ny = self.y + self.vy

        # Colisión con paredes de la ventana
        if nx - BALL_RADIUS < 0:
            nx = float(BALL_RADIUS)
            self.vx = abs(self.vx) * RESTITUTION
        elif nx + BALL_RADIUS > w:
            nx = float(w - BALL_RADIUS)
            self.vx = -abs(self.vx) * RESTITUTION

        if ny - BALL_RADIUS < 0:
            ny = float(BALL_RADIUS)
            self.vy = abs(self.vy) * RESTITUTION
        elif ny + BALL_RADIUS > h:
            ny = float(h - BALL_RADIUS)
            self.vy = -abs(self.vy) * RESTITUTION
            self.vx *= FRICTION          # rozamiento en el suelo

        # Colisión con la máscara de bordes
        nx, ny = self._collide(nx, ny, h, w, thick, gx_map, gy_map)

        self.trail.append((int(self.x), int(self.y)))
        self.x, self.y = nx, ny

    def _collide(self, nx, ny, h, w, thick, gx_map, gy_map):
        """
        Muestrea 12 puntos en el perímetro de la pelota.
        Si alguno toca la máscara de bordes calcula la normal
        (gradiente Sobel) y aplica la reflexión de velocidad.
        """
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            sx = int(np.clip(nx + BALL_RADIUS * np.cos(rad), 0, w - 1))
            sy = int(np.clip(ny + BALL_RADIUS * np.sin(rad), 0, h - 1))

            if thick[sy, sx] == 0:
                continue   # sin contacto en este punto

            # Normal de la superficie en el punto de contacto
            n_x = float(gx_map[sy, sx])
            n_y = float(gy_map[sy, sx])
            mag = np.hypot(n_x, n_y)

            if mag > 1e-5:
                n_x /= mag
                n_y /= mag
                # Reflexión física: v' = v - 2(v·n)n
                dot = self.vx * n_x + self.vy * n_y
                self.vx = (self.vx - 2.0 * dot * n_x) * RESTITUTION
                self.vy = (self.vy - 2.0 * dot * n_y) * RESTITUTION
            else:
                # Normal no disponible → inversión simple
                self.vx *= -RESTITUTION
                self.vy *= -RESTITUTION

            self.vx *= FRICTION
            # Retroceder al último punto seguro
            return float(int(self.x)), float(int(self.y))

        return nx, ny

    # ── Colisión con segmentos de dedo ───────────────────────
    def apply_hand_force(self, lm_px: list, hvx: float, hvy: float):
        """
        Modelo de colisión real: cada segmento de dedo es una cápsula.
        Si la pelota se superpone con alguna cápsula:
          - Se empuja fuera del overlap (depenetración).
          - Hereda la velocidad de la mano (efecto de golpe/arrastre).
        lm_px: lista de (x,y) en píxeles de los 21 landmarks.
        """
        best_overlap = 0.0
        best_nx = best_ny = 0.0

        for (i, j) in FINGER_SEGMENTS:
            ax, ay = lm_px[i]
            bx, by = lm_px[j]
            cx, cy, dist = _closest_on_segment(ax, ay, bx, by, self.x, self.y)
            combined = BALL_RADIUS + FINGER_RADIUS
            if dist < combined:
                overlap = combined - dist
                if overlap > best_overlap:
                    best_overlap = overlap
                    if dist > 0.5:
                        best_nx = (self.x - cx) / dist
                        best_ny = (self.y - cy) / dist
                    else:
                        best_nx, best_ny = 0.0, -1.0

        if best_overlap > 0.0:
            # 1. Depenetración: sacar la pelota del dedo
            self.x += best_nx * best_overlap
            self.y += best_ny * best_overlap
            # 2. Transferir velocidad de la mano (efecto de golpe)
            self.vx = self.vx * 0.3 + hvx * VELOCITY_BLEND
            self.vy = self.vy * 0.3 + hvy * VELOCITY_BLEND
            # 3. Pequeño impulso extra en la dirección normal (para que "salte")
            self.vx += best_nx * 1.5
            self.vy += best_ny * 1.5

    # ── Dibujo ──────────────────────────────────────────────
    def draw(self, img):
        trail = list(self.trail)
        n = len(trail)
        for i, (tx, ty) in enumerate(trail):
            alpha = (i + 1) / max(n, 1)
            r = max(2, int(BALL_RADIUS * alpha * 0.55))
            c = tuple(int(ch * alpha * 0.40) for ch in self.color)
            cv2.circle(img, (tx, ty), r, c, -1)

        cx, cy = int(self.x), int(self.y)
        # Glow (dos capas)
        for extra, alpha in ((BALL_RADIUS + 8, 0.18), (BALL_RADIUS + 4, 0.26)):
            ov = img.copy()
            cv2.circle(ov, (cx, cy), extra, self.color, -1)
            cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)
        cv2.circle(img, (cx, cy), BALL_RADIUS, self.color, -1)
        # Brillo especular
        bx = cx - BALL_RADIUS // 3
        by = cy - BALL_RADIUS // 3
        cv2.circle(img, (bx, by), max(2, BALL_RADIUS // 4), (255, 255, 255), -1)


# Segmentos de dedo usados para colisión (pares de índices MediaPipe)
FINGER_SEGMENTS = [
    (0, 1),(1, 2),(2, 3),(3, 4),        # pulgar
    (0, 5),(5, 6),(6, 7),(7, 8),        # índice
    (5, 9),(9,10),(10,11),(11,12),       # medio
    (9,13),(13,14),(14,15),(15,16),      # anular
    (13,17),(17,18),(18,19),(19,20),     # meñique
    (0,17),(0,5),                        # palma
]

def _closest_on_segment(ax, ay, bx, by, px, py):
    """Punto más cercano al segmento AB desde P; devuelve (cx, cy, dist)."""
    dx, dy = bx - ax, by - ay
    seg_len2 = dx*dx + dy*dy
    if seg_len2 < 1e-6:
        return ax, ay, np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / seg_len2))
    cx, cy = ax + t*dx, ay + t*dy
    return cx, cy, np.hypot(px - cx, py - cy)


# ── Construir mapa de colisión ──────────────────────────────
def build_collision(frame, hands, w, h):
    """
    SOLO Canny sobre objetos físicos del mundo (cintas, libros, etc.).
    La mano NO entra aquí; su física se gestiona con apply_hand_force.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LO, CANNY_HI)

    # Enmascarar la región de la mano para que Canny no la detecte
    # (evita que la mano cree una pared que pelee con apply_hand_force)
    for hand in hands:
        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in hand], np.int32
        )
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(edges, hull, 0)   # borrar bordes dentro de la mano

    # Dilatar para dar grosor físico
    kernel = np.ones((EDGE_DILATE, EDGE_DILATE), np.uint8)
    thick  = cv2.dilate(edges, kernel, iterations=1)

    # Normales de superficie vía gradiente
    ft = thick.astype(np.float32)
    gx = cv2.Sobel(ft, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(ft, cv2.CV_32F, 0, 1, ksize=5)

    return thick, gx, gy


# ── Bucle principal ─────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara.")
        return

    detector    = HandDetector()
    debug       = False   # mostrar bordes de colisión
    # Un tracker por mano (hasta 2)
    trackers    = [HandTracker(), HandTracker()]

    def spawn(w, initial=False):
        y = random.randint(10, 80) if initial else 20
        return Pelota(random.randint(BALL_RADIUS + 10, w - BALL_RADIUS - 10), y)

    balls = []   # se crean tras conocer w, h

    print("Física en cámara | ESC=salir  ESPACIO=+pelota  R=reset  D=debug")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Crear pelotas iniciales ahora que conocemos w
        if not balls:
            balls = [spawn(w, initial=True) for _ in range(INIT_BALLS)]

        # Detección de mano y mapa de colisión
        hands              = detector.detectar_mano(frame)
        thick, gx_m, gy_m  = build_collision(frame, hands, w, h)

        # Actualizar trackers de velocidad para cada mano detectada
        hand_forces = []   # lista de (lm_px, vx, vy) por mano
        for i, hand in enumerate(hands[:2]):
            trackers[i].update(hand, w, h)
            lm_px = [(hand[lm].x * w, hand[lm].y * h) for lm in range(21)]
            hand_forces.append((lm_px, trackers[i].vx, trackers[i].vy))
        # Resetear trackers de manos que dejaron de verse
        for i in range(len(hands), 2):
            trackers[i].reset()

        # Fondo: frame tenue
        bg = (frame.astype(np.float32) * 0.45).astype(np.uint8)

        # Esqueleto de mano
        for hand in hands:
            detector.dibujar_mano(bg, hand)

        # Overlay de bordes de colisión (modo debug)
        if debug:
            ev = np.zeros_like(bg)
            ev[thick > 0] = (30, 255, 90)
            cv2.addWeighted(bg, 1.0, ev, 0.35, 0, bg)

        # Física y dibujo
        for ball in balls:
            for (lm_px, hvx, hvy) in hand_forces:
                ball.apply_hand_force(lm_px, hvx, hvy)
            ball.step(h, w, thick, gx_m, gy_m)
            ball.draw(bg)

        # HUD mano
        if hand_forces:
            cv2.putText(bg, f"mano vel: ({hand_forces[0][1]:+.1f}, {hand_forces[0][2]:+.1f})",
                        (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (255, 180, 60), 1, cv2.LINE_AA)

        # HUD
        hints = "ESC salir | ESPACIO +pelota | R reset | D debug"
        cv2.putText(bg, f"Pelotas: {len(balls)}   {hints}",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (80, 200, 255), 1, cv2.LINE_AA)
        if debug:
            cv2.putText(bg, "DEBUG: bordes activos",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (30, 255, 90), 1, cv2.LINE_AA)

        cv2.imshow("Fisica en camara", bg)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:                          # ESC
            break
        elif key == ord(' '):                  # añadir pelota
            if len(balls) < MAX_BALLS:
                balls.append(spawn(w))
        elif key in (ord('r'), ord('R')):      # reiniciar
            balls = [spawn(w, initial=True) for _ in range(INIT_BALLS)]
        elif key in (ord('d'), ord('D')):      # toggle debug
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
