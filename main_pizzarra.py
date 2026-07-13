def draw_holo_face(screen, cx, cy, size=80):
    # Ojos (esferas)
    eye_offset_x = int(size * 0.38)
    eye_offset_y = int(size * -0.22)
    eye_radius = int(size * 0.22)
    for ex in [-eye_offset_x, eye_offset_x]:
        draw_holo_sphere(screen, cx + ex, cy + eye_offset_y, eye_radius)
    # Boca (estirada, tipo "boca abierta")
    mouth_w = int(size * 0.95)
    mouth_h = int(size * 0.32)
    mouth_y = int(cy + size * 0.38)
    overlay = screen.copy()
    cv2.ellipse(overlay, (cx, mouth_y), (mouth_w // 2, mouth_h // 2), 0, 0, 360, (75, 255, 220), -1)
    cv2.addWeighted(overlay, 0.13, screen, 0.87, 0, screen)
    cv2.ellipse(screen, (cx, mouth_y), (mouth_w // 2, mouth_h // 2), 0, 0, 360, (120, 255, 255), 2)
    cv2.ellipse(screen, (cx, mouth_y), (max(2, mouth_w // 12), max(2, mouth_h // 6)), 0, 0, 360, (255, 255, 255), -1)
import cv2
import numpy as np
import random
import time
from collections import deque
from mano2 import HandDetector


# -------------------------------
# Holographic sphere interaction
# -------------------------------
class HoloSphere:
    def __init__(self, x, y, radius=80):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.grabbed_by = None
        self.grab_offset = (0.0, 0.0)

    def contains(self, px, py):
        return (self.x - px) ** 2 + (self.y - py) ** 2 <= self.radius ** 2


def draw_holo_sphere(screen, sx, sy, sr):
    sr = int(max(8, sr))
    for r, alpha in [(sr + 26, 0.06), (sr + 14, 0.11), (sr + 6, 0.17)]:
        overlay = screen.copy()
        cv2.circle(overlay, (int(sx), int(sy)), r, (75, 255, 220), -1)
        cv2.addWeighted(overlay, alpha, screen, 1 - alpha, 0, screen)

    cv2.circle(screen, (int(sx), int(sy)), sr, (120, 255, 255), 2)
    cv2.circle(screen, (int(sx), int(sy)), max(2, sr // 16), (255, 255, 255), -1)


# -------------------------------
# Shape recognition
# -------------------------------
def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    rc = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, rc


def is_closed(points, threshold=30):
    return np.linalg.norm(points[0] - points[-1]) < threshold


def is_triangle(points):
    from scipy.signal import find_peaks

    if len(points) < 20:
        return False
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    curv = np.abs(np.gradient(np.arctan2(dy, dx)))
    peaks, _ = find_peaks(curv, height=0.2, distance=max(3, len(points) // 4))
    return len(peaks) == 3 and is_closed(points, threshold=45)


def get_triangle(points):
    from scipy.signal import find_peaks

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    curv = np.abs(np.gradient(np.arctan2(dy, dx)))
    peaks, _ = find_peaks(curv, height=0.2, distance=max(3, len(points) // 4))
    corners = points[peaks]
    return [tuple(map(int, p)) for p in corners]


def is_square(points):
    from scipy.signal import find_peaks

    if len(points) < 24:
        return False
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    curv = np.abs(np.gradient(np.arctan2(dy, dx)))
    peaks, _ = find_peaks(curv, height=0.18, distance=max(4, len(points) // 5))
    if len(peaks) != 4:
        return False
    corners = points[peaks]
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    ratio = np.std(sides) / max(np.mean(sides), 1e-6)
    return ratio < 0.35 and is_closed(points, threshold=45)


def get_square(points):
    from scipy.signal import find_peaks

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    curv = np.abs(np.gradient(np.arctan2(dy, dx)))
    peaks, _ = find_peaks(curv, height=0.18, distance=max(4, len(points) // 5))
    corners = points[peaks]
    return [tuple(map(int, p)) for p in corners]


def is_spiral(points):
    if len(points) < 30:
        return False
    center = np.mean(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    angles = np.unwrap(np.angle((points[:, 0] - center[0]) + 1j * (points[:, 1] - center[1])))
    return np.ptp(radius) > 60 and np.sum(np.diff(angles) > 0) > 10


def is_star(points):
    from scipy.signal import find_peaks

    if len(points) < 30:
        return False
    center = np.mean(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    peaks, _ = find_peaks(radius, height=np.mean(radius) + 10, distance=max(4, len(points) // 8))
    return len(peaks) in [5, 10] and is_closed(points, threshold=45)


def get_star(points):
    from scipy.signal import find_peaks

    center = np.mean(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    peaks, _ = find_peaks(radius, height=np.mean(radius) + 10, distance=max(4, len(points) // 8))
    return [tuple(map(int, points[i])) for i in peaks]


def recognize_shape(points):
    p = np.array(points)
    if len(p) < 10:
        return None, None

    xc, yc, rc = fit_circle(p)
    err = np.mean(np.abs(np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - rc))
    if err < 10 and is_closed(p):
        return "circle", (int(xc), int(yc), int(rc))

    dist = np.linalg.norm(p[0] - p[-1])
    length = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    if length > 0 and dist > 0.85 * length:
        return "line", (tuple(p[0]), tuple(p[-1]))

    if is_square(p):
        return "square", get_square(p)

    if is_spiral(p):
        return "spiral", None

    if is_star(p):
        return "star", get_star(p)

    return None, None


def normalize_group(group):
    out = []
    for g in group:
        arr = np.array(g)
        if arr.size == 0:
            continue
        out.append(arr.reshape(-1, 2))
    return out


# -------------------------------
# Particle system
# -------------------------------
class Spark:
    def __init__(self, x, y):
        ang = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(30, 60)
        self.vx = np.cos(ang) * dist / 8.0
        self.vy = np.sin(ang) * dist / 8.0
        self.x = float(x)
        self.y = float(y)
        self.t0 = time.time()
        self.duration = 0.18 + np.random.uniform(0, 0.12)
        self.size = np.random.randint(2, 4)
        self.color = (0, 215, 255)
        self.max_dist = dist
        self.current_dist = 0.0

    def alive(self):
        return (time.time() - self.t0) < self.duration and self.current_dist < self.max_dist

    def alpha(self):
        return max(0.0, 1.0 - (time.time() - self.t0) / self.duration)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.current_dist += np.sqrt(self.vx ** 2 + self.vy ** 2)


# -------------------------------
# App state
# -------------------------------
def clamp(val, mn, mx):
    return max(mn, min(val, mx))


def draw_glow_line(img, p1, p2):
    for r, alpha in [(10, 0.10), (7, 0.18), (4, 0.28)]:
        overlay = img.copy()
        cv2.line(overlay, p1, p2, (255, 255, 255), r)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.line(img, p1, p2, (255, 255, 255), 2)


def draw_shape_glow(img, shape, params):
    if shape == "circle":
        xc, yc, rc = params
        for r, alpha in [(18, 0.10), (12, 0.18), (8, 0.28)]:
            overlay = img.copy()
            cv2.circle(overlay, (xc, yc), rc, (255, 255, 255), r)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.circle(img, (xc, yc), rc, (255, 255, 255), 4)
    elif shape == "line":
        p1, p2 = params
        for r, alpha in [(18, 0.10), (12, 0.18), (8, 0.28)]:
            overlay = img.copy()
            cv2.line(overlay, p1, p2, (255, 255, 255), r)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.line(img, p1, p2, (255, 255, 255), 4)
    elif shape == "square":
        pts = params
        if len(pts) >= 4:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(pts)], 255)
            img[mask == 255] = 0
            for i in range(4):
                p1, p2 = pts[i], pts[(i + 1) % 4]
                for r, alpha in [(18, 0.10), (12, 0.18), (8, 0.28)]:
                    overlay = img.copy()
                    cv2.line(overlay, p1, p2, (255, 255, 255), r)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.line(img, p1, p2, (255, 255, 255), 4)
    elif shape == "star":
        pts = params
        if len(pts) >= 5:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(pts)], 255)
            img[mask == 255] = 0
            n = len(pts)
            for i in range(n):
                p1, p2 = pts[i], pts[(i + 2) % n]
                for r, alpha in [(18, 0.10), (12, 0.18), (8, 0.28)]:
                    overlay = img.copy()
                    cv2.line(overlay, p1, p2, (255, 255, 255), r)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.line(img, p1, p2, (255, 255, 255), 4)


CANVAS_WIDTH, CANVAS_HEIGHT = 2400, 1600
WIDTH, HEIGHT = 800, 400

pizarra = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)
detector = HandDetector()

zoom = 1.0
zoom_active = False
y_zoom_start = None
zoom_start_value = 1.0

pan_x, pan_y = 0, 0
pan_active = False
pan_x_start = 0
pan_y_start = 0
hand_x_start = 0
hand_y_start = 0

smooth_x = deque(maxlen=5)
smooth_y = deque(maxlen=5)

trazos = []
trazo_actual = []
ultimo_punto = None

sparks = []

spheres = []
invoke_active = False
invoke_last_time = 0.0
hand_scale_prev = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    hands = detector.detectar_mano(frame)
    hand_main = hands[0] if hands else None

    draw_mode = False
    draw_point = None

    # Escalar esfera o cara con gesto de zoom sobre el objeto
    if hand_main is not None and detector.is_zoom_start(hand_main):
        # Centro de la palma (landmark 9)
        cx_vis = int(hand_main[9].x * WIDTH)
        cy_vis = int(hand_main[9].y * HEIGHT)
        cx = int((cx_vis + pan_x) / zoom)
        cy = int((cy_vis + pan_y) / zoom)
        # Buscar esfera bajo la mano
        for sphere in spheres:
            if sphere.contains(cx, cy):
                if not hasattr(sphere, 'zoom_y_start') or sphere.zoom_y_start is None:
                    sphere.zoom_y_start = cy
                    sphere.radius_start = sphere.radius
                else:
                    delta_y = sphere.zoom_y_start - cy
                    sphere.radius = clamp(sphere.radius_start + delta_y * 0.5, 35, 220)
                break
        else:
            # Si no hay esfera, buscar si la mano está sobre una cara (boca)
            if detector.is_shaka(hand_main):
                # Boca centrada en la palma
                if not hasattr(hand_main, 'zoom_y_start') or hand_main.zoom_y_start is None:
                    hand_main.zoom_y_start = cy
                    hand_main.face_size_start = 85
                else:
                    delta_y = hand_main.zoom_y_start - cy
                    face_size = clamp(hand_main.face_size_start + delta_y * 0.5, 35, 220)
                    draw_holo_face(pizarra, cx, cy, size=face_size)
    else:
        # Reset zoom tracking para esferas
        for sphere in spheres:
            if hasattr(sphere, 'zoom_y_start'):
                sphere.zoom_y_start = None
                sphere.radius_start = None
        # Reset zoom tracking para cara
        if hand_main is not None and hasattr(hand_main, 'zoom_y_start'):
            hand_main.zoom_y_start = None
            hand_main.face_size_start = None

    # Dibuja cara si la mano principal hace el gesto shaka (con delay de 5s)
    if 'shaka_last_time' not in globals():
        global shaka_last_time
        shaka_last_time = 0.0
    if hand_main is not None and detector.is_shaka(hand_main):
        if time.time() - shaka_last_time > 5.0:
            # Centro de la palma (landmark 9)
            cx_vis = int(hand_main[9].x * WIDTH)
            cy_vis = int(hand_main[9].y * HEIGHT)
            cx = int((cx_vis + pan_x) / zoom)
            cy = int((cy_vis + pan_y) / zoom)
            cx = clamp(cx, 0, CANVAS_WIDTH - 1)
            cy = clamp(cy, 0, CANVAS_HEIGHT - 1)
            # Tamaño similar a la esfera
            draw_holo_face(pizarra, cx, cy, size=85)
            shaka_last_time = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    hands = detector.detectar_mano(frame)
    hand_main = hands[0] if hands else None

    draw_mode = False
    draw_point = None

    # Hand points in canvas coordinates (for all hands)
    hand_points_canvas = []
    for i, hand in enumerate(hands[:2]):
        px_vis = int(hand[9].x * WIDTH)
        py_vis = int(hand[9].y * HEIGHT)
        px = int((px_vis + pan_x) / zoom)
        py = int((py_vis + pan_y) / zoom)
        px = clamp(px, 0, CANVAS_WIDTH - 1)
        py = clamp(py, 0, CANVAS_HEIGHT - 1)
        hand_points_canvas.append((i, hand, px, py))

    # Draw hand landmarks for feedback
    for hand in hands:
        frame = detector.dibujar_mano(frame, hand)

    # Gesture: invoke sphere when two hands join
    if len(hand_points_canvas) >= 2:
        _, h0, x0, y0 = hand_points_canvas[0]
        _, h1, x1, y1 = hand_points_canvas[1]
        dist_hands = np.hypot(x0 - x1, y0 - y1)
        both_open = detector.is_hand_open(h0) and detector.is_hand_open(h1)
        ready_to_invoke = dist_hands < 160 and both_open

        if ready_to_invoke and not invoke_active and (time.time() - invoke_last_time) > 5.0:
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            spheres.append(HoloSphere(cx, cy, radius=85))
            invoke_active = True
            invoke_last_time = time.time()

        if not ready_to_invoke:
            invoke_active = False
    else:
        invoke_active = False

    # Grab / move / release spheres with fist inside
    for sphere in spheres:
        # release check
        if sphere.grabbed_by is not None:
            grabbed_hand = hands[sphere.grabbed_by] if sphere.grabbed_by < len(hands) else None
            if grabbed_hand is None or detector.is_hand_open(grabbed_hand):
                sphere.grabbed_by = None
                hand_scale_prev = None

        # acquire check
        if sphere.grabbed_by is None:
            for idx, hand, hx, hy in hand_points_canvas:
                if sphere.contains(hx, hy) and detector.is_hand_closed(hand):
                    sphere.grabbed_by = idx
                    sphere.grab_offset = (sphere.x - hx, sphere.y - hy)
                    break

        # move while grabbed
        if sphere.grabbed_by is not None and sphere.grabbed_by < len(hand_points_canvas):
            idx = sphere.grabbed_by
            _, hand, hx, hy = hand_points_canvas[idx]
            if detector.is_hand_closed(hand):
                sphere.x = hx + sphere.grab_offset[0]
                sphere.y = hy + sphere.grab_offset[1]

    # Scale selected sphere when both hands are near it
    if len(hand_points_canvas) >= 2:
        _, h0, x0, y0 = hand_points_canvas[0]
        _, h1, x1, y1 = hand_points_canvas[1]
        hand_dist = np.hypot(x0 - x1, y0 - y1)

        for sphere in spheres:
            near_both = sphere.contains(x0, y0) and sphere.contains(x1, y1)
            if near_both and detector.is_hand_closed(h0) and detector.is_hand_closed(h1):
                if hand_scale_prev is None:
                    hand_scale_prev = hand_dist
                else:
                    delta = hand_dist - hand_scale_prev
                    sphere.radius = clamp(sphere.radius + delta * 0.25, 35, 220)
                    hand_scale_prev = hand_dist
                break
        else:
            hand_scale_prev = None
    else:
        hand_scale_prev = None


    # Keep zoom/pan/draw behavior from first hand
    if hand_main is not None:
        cx = int(hand_main[0].x * w)
        cy = int(hand_main[0].y * h)

        # Verificar si la mano principal está agarrando una esfera
        hand_main_grabbing_sphere = any(
            sphere.grabbed_by == 0 for sphere in spheres
        )

        if detector.is_zoom_start(hand_main):
            if not zoom_active:
                y_zoom_start = cy
                zoom_start_value = zoom
                zoom_active = True
            else:
                delta_y = y_zoom_start - cy
                factor = 1 + (delta_y / 400.0)
                zoom = clamp(zoom_start_value * factor, 0.3, 3.0)
            pan_active = False
        elif zoom_active:
            zoom_active = False
            y_zoom_start = None

        if detector.is_index_click(hand_main) and not detector.is_zoom_start(hand_main):
            draw_mode = True
            ix_vis = int(hand_main[8].x * WIDTH)
            iy_vis = int(hand_main[8].y * HEIGHT)
            ix = int((ix_vis + pan_x) / zoom)
            iy = int((iy_vis + pan_y) / zoom)
            draw_point = (ix, iy)
            pan_active = False
        elif (
            detector.is_hand_closed(hand_main)
            and not detector.is_zoom_start(hand_main)
            and not detector.is_index_click(hand_main)
            and not hand_main_grabbing_sphere
        ):
            if not pan_active:
                hand_x_start = cx
                hand_y_start = cy
                pan_x_start = pan_x
                pan_y_start = pan_y
                pan_active = True
                ultimo_punto = None
            else:
                dx = cx - hand_x_start
                dy = cy - hand_y_start
                pan_x = pan_x_start - dx
                pan_y = pan_y_start - dy
                vis_w = int(CANVAS_WIDTH * zoom)
                vis_h = int(CANVAS_HEIGHT * zoom)
                pan_x = clamp(pan_x, 0, max(0, vis_w - WIDTH))
                pan_y = clamp(pan_y, 0, max(0, vis_h - HEIGHT))
        else:
            if pan_active:
                pan_active = False

    mira_punto = None
    if hand_main is not None:
        ix_vis = int(hand_main[8].x * WIDTH)
        iy_vis = int(hand_main[8].y * HEIGHT)
        ix = int((ix_vis + pan_x) / zoom)
        iy = int((iy_vis + pan_y) / zoom)
        x = clamp(ix, 0, CANVAS_WIDTH - 1)
        y = clamp(iy, 0, CANVAS_HEIGHT - 1)
        mira_punto = (x, y)

    if draw_mode and draw_point is not None and hand_main is not None:
        trazo_actual.append((x, y))

        smooth_x.append(draw_point[0])
        smooth_y.append(draw_point[1])
        x = int(np.mean(smooth_x))
        y = int(np.mean(smooth_y))
        x = clamp(x, 0, CANVAS_WIDTH - 1)
        y = clamp(y, 0, CANVAS_HEIGHT - 1)

        if 0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT:
            if ultimo_punto is not None:
                x0, y0 = ultimo_punto
                dist = np.hypot(x - x0, y - y0)
                steps = max(1, int(dist))
                for i in range(1, steps + 1):
                    xi = int(x0 + (x - x0) * i / steps)
                    yi = int(y0 + (y - y0) * i / steps)
                    draw_glow_line(pizarra, (x0, y0), (xi, yi))
                    x0, y0 = xi, yi
            else:
                for r, alpha in [(10, 0.10), (7, 0.18), (4, 0.28)]:
                    overlay = pizarra.copy()
                    cv2.circle(overlay, (x, y), r, (255, 255, 255), -1)
                    cv2.addWeighted(overlay, alpha, pizarra, 1 - alpha, 0, pizarra)
                cv2.circle(pizarra, (x, y), 4, (255, 255, 255), -1)
                cv2.circle(pizarra, (x, y), 2, (255, 255, 255), -1)

            ultimo_punto = (x, y)
            px_vis = int(hand_main[8].x * WIDTH)
            py_vis = int(hand_main[8].y * HEIGHT)
            for _ in range(random.randint(3, 7)):
                sparks.append(Spark(px_vis, py_vis))
    else:
        if trazo_actual:
            if len(trazo_actual) > 10:
                close_dist = np.linalg.norm(np.array(trazo_actual[0]) - np.array(trazo_actual[-1]))
                if close_dist < 40:
                    trazo_actual.append(trazo_actual[0])

            shape, params = recognize_shape(trazo_actual)

            # Multi-stroke square
            if len(trazos) >= 3:
                group = normalize_group(trazos[-3:] + [trazo_actual])
                if len(group) >= 2:
                    points_group = np.concatenate(group)
                    if is_square(points_group):
                        mask = np.zeros(pizarra.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [points_group], 255)
                        pizarra[mask == 255] = 0
                        draw_shape_glow(pizarra, "square", get_square(points_group))
                        trazos = trazos[:-3]
                        trazo_actual = []

            # Multi-stroke triangle
            if len(trazos) >= 2 and trazo_actual:
                group = normalize_group(trazos[-2:] + [trazo_actual])
                if len(group) >= 2:
                    points_group = np.concatenate(group)
                    if is_triangle(points_group):
                        mask = np.zeros(pizarra.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [points_group], 255)
                        pizarra[mask == 255] = 0
                        pts = get_triangle(points_group)
                        if len(pts) >= 3:
                            for i in range(3):
                                draw_shape_glow(pizarra, "line", (pts[i], pts[(i + 1) % 3]))
                        trazos = trazos[:-2]
                        trazo_actual = []

            # Multi-stroke star
            if len(trazos) >= 4 and trazo_actual:
                group = normalize_group(trazos[-4:] + [trazo_actual])
                if len(group) >= 2:
                    points_group = np.concatenate(group)
                    if is_star(points_group):
                        mask = np.zeros(pizarra.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [points_group], 255)
                        pizarra[mask == 255] = 0
                        draw_shape_glow(pizarra, "star", get_star(points_group))
                        trazos = trazos[:-4]
                        trazo_actual = []

            if shape == "circle":
                points = np.array(trazo_actual)
                if len(points) >= 3:
                    mask = np.zeros(pizarra.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)
                    pizarra[mask == 255] = 0
                draw_shape_glow(pizarra, "circle", params)
            elif shape == "line":
                draw_shape_glow(pizarra, "line", params)
            elif shape == "square":
                draw_shape_glow(pizarra, "square", params)
            elif shape == "spiral":
                pts = np.array(trazo_actual)
                for i in range(1, len(pts)):
                    draw_glow_line(pizarra, tuple(pts[i - 1]), tuple(pts[i]))
            elif shape == "star":
                draw_shape_glow(pizarra, "star", params)
            else:
                trazos.append(trazo_actual)

            trazo_actual = []

        ultimo_punto = None
        smooth_x.clear()
        smooth_y.clear()

    vis = cv2.resize(pizarra, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    h_vis, w_vis, _ = vis.shape
    fondo = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    x0 = clamp(pan_x, 0, max(0, w_vis - WIDTH))
    y0 = clamp(pan_y, 0, max(0, h_vis - HEIGHT))
    x1 = x0 + WIDTH
    y1 = y0 + HEIGHT

    fondo[: min(h_vis - y0, HEIGHT), : min(w_vis - x0, WIDTH)] = vis[y0:y1, x0:x1]

    # Draw spheres on the visible screen (not permanently on canvas)
    for sphere in spheres:
        sx = int(sphere.x * zoom - pan_x)
        sy = int(sphere.y * zoom - pan_y)
        sr = int(sphere.radius * zoom)
        if -sr < sx < WIDTH + sr and -sr < sy < HEIGHT + sr:
            draw_holo_sphere(fondo, sx, sy, sr)

    new_sparks = []
    for s in sparks:
        s.step()
        if s.alive():
            alpha = s.alpha()
            overlay = fondo.copy()
            cv2.circle(overlay, (int(s.x), int(s.y)), s.size, s.color, -1)
            cv2.addWeighted(overlay, alpha, fondo, 1 - alpha, 0, fondo)
            new_sparks.append(s)
    sparks = new_sparks

    if hand_main is not None:
        fondo = detector.dibujar_mano(fondo, hand_main)

    if mira_punto is not None:
        mx = int((mira_punto[0] * zoom) - pan_x)
        my = int((mira_punto[1] * zoom) - pan_y)
        if 0 <= mx < WIDTH and 0 <= my < HEIGHT:
            for r, alpha in [(28, 0.10), (18, 0.18), (12, 0.28)]:
                overlay = fondo.copy()
                cv2.circle(overlay, (mx, my), r, (0, 215, 255), -1)
                cv2.addWeighted(overlay, alpha, fondo, 1 - alpha, 0, fondo)
            cv2.circle(fondo, (mx, my), 10, (255, 255, 100), 2)
            cv2.circle(fondo, (mx, my), 6, (255, 255, 255), -1)

    cv2.putText(fondo, "Junta 2 manos abiertas: invocar esfera", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    cv2.putText(fondo, "Punio dentro: mover | 2 punios en esfera: escalar", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)

    cv2.imshow("Pizarra magica", fondo)
    cv2.imshow("Mano", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
