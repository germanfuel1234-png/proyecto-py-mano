###############################################################
# fuego_mano.py
# Detecta la mano con MediaPipe y genera una estela de fuego
# que sigue las puntas de los dedos en tiempo real.
# ESC para salir.
###############################################################

import cv2
import numpy as np
import random
from mano2 import HandDetector

# Índices de las puntas de los 5 dedos en MediaPipe
FINGERTIPS = [4, 8, 12, 16, 20]

# Máximo de partículas vivas al mismo tiempo
MAX_PARTICLES = 1200

# Partículas por punta por frame (más = fuego más denso)
SPAWN_RATE = 6


# ---------------------------------------------------------------
# Partícula de fuego
# ---------------------------------------------------------------
class FuegoParticula:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'decay', 'size']

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        # Velocidad: sube con ligera dispersión horizontal
        self.vx = random.gauss(0, 1.8)
        self.vy = random.uniform(-5.5, -1.8)
        self.life = 1.0
        self.decay = random.uniform(0.022, 0.055)
        self.size = random.randint(5, 13)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        # Aceleración ascendente (el fuego sube más rápido al enfriarse)
        self.vy -= 0.09
        # Fricción horizontal
        self.vx *= 0.95
        self.life -= self.decay

    @property
    def alive(self) -> bool:
        return self.life > 0.0


# ---------------------------------------------------------------
# Color basado en ciclo de vida: blanco → amarillo → naranja → rojo
# ---------------------------------------------------------------
def _color_fuego(life: float):
    """Devuelve color BGR según vida (1.0 = recién nacida, 0 = muerta)."""
    t = max(0.0, min(1.0, life))
    if t > 0.75:                          # blanco → amarillo
        frac = (t - 0.75) / 0.25
        b = int(255 * frac)
        g, r = 255, 255
    elif t > 0.45:                        # amarillo → naranja
        frac = (t - 0.45) / 0.30
        b = 0
        g = int(80 + 175 * frac)
        r = 255
    elif t > 0.18:                        # naranja → rojo
        frac = (t - 0.18) / 0.27
        b = 0
        g = int(80 * frac)
        r = 255
    else:                                 # rojo → negro
        frac = t / 0.18
        b, g = 0, 0
        r = int(255 * frac)
    return (b, g, r)


# ---------------------------------------------------------------
# Dibujar todas las partículas en una capa y aplicar glow
# ---------------------------------------------------------------
def _dibujar_capa_fuego(shape, particles) -> np.ndarray:
    capa = np.zeros(shape, dtype=np.uint8)
    for p in particles:
        if not p.alive:
            continue
        color = _color_fuego(p.life)
        sz = max(1, int(p.size * (0.3 + 0.7 * p.life)))
        cv2.circle(capa, (int(p.x), int(p.y)), sz, color, -1)

    # Glow: sumar la capa difuminada a sí misma
    blur = cv2.GaussianBlur(capa, (19, 19), 0)
    glow = cv2.addWeighted(capa, 1.0, blur, 0.8, 0)
    return glow


# ---------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la cámara.")
        return

    detector = HandDetector()
    particles: list[FuegoParticula] = []

    print("Mueve la mano frente a la cámara.  ESC = salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- Detectar manos y spawnear partículas ---
        hands = detector.detectar_mano(frame)
        if hands:
            for hand in hands:
                for tip_idx in FINGERTIPS:
                    tx = hand[tip_idx].x * w
                    ty = hand[tip_idx].y * h
                    for _ in range(SPAWN_RATE):
                        if len(particles) < MAX_PARTICLES:
                            # Pequeño offset aleatorio para que nazcan dispersas
                            ox = random.uniform(-4, 4)
                            oy = random.uniform(-4, 4)
                            particles.append(FuegoParticula(tx + ox, ty + oy))

        # --- Actualizar y limpiar partículas ---
        for p in particles:
            p.update()
        particles = [p for p in particles if p.alive]

        # --- Fondo oscuro: cámara muy tenue ---
        bg = (frame.astype(np.float32) * 0.20).astype(np.uint8)

        # --- Dibujar esqueleto de la mano encima del fondo ---
        if hands:
            for hand in hands:
                detector.dibujar_mano(bg, hand)

        # --- Capa de fuego ---
        fuego = _dibujar_capa_fuego(bg.shape, particles)

        # --- Mezcla tipo "Screen": el fuego se suma sin oscurecer el fondo ---
        bg_f    = bg.astype(np.float32) / 255.0
        fuego_f = fuego.astype(np.float32) / 255.0
        resultado = 1.0 - (1.0 - bg_f) * (1.0 - fuego_f)
        resultado = (resultado * 255.0).clip(0, 255).astype(np.uint8)

        # --- HUD mínimo ---
        cv2.putText(resultado, "Fuego Mano | ESC salir",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(resultado, f"Particulas: {len(particles)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("Fuego en la mano", resultado)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
