"""Primitivas de dibujo del estilo Toybox: bloques, chef, comida, fuegos, íconos.

Todas las medidas están en unidades del mockup (lienzo de 470 x 320) y se
multiplican por `k`. Las funciones que reciben un `Iso` toman `k` de él.

Uso para revisar a ojo:
    python -m viz.sprites   # guarda results/figures/sprites_preview.png
"""

import math
import random

import pygame

from viz.iso import COUNTER_HEIGHT, STATION_TILES, Iso
from viz.style import DISH_COLORS, PALETTE, SHIRTS

OUTLINE_W = 1.6
ICONS = ("confused", "yum", "check", "warning", "angry", "puke", "dots", "question",
         "clock", "coin", "minus_money", "plus_money")

_fonts: dict[int, pygame.font.Font] = {}


# --------------------------------------------------------------- básicos
def _w(k: float, ow: float = OUTLINE_W) -> int:
    return max(1, round(ow * k))


def font(size: int) -> pygame.font.Font:
    size = max(6, int(size))
    if size not in _fonts:
        if not pygame.font.get_init():
            pygame.font.init()
        if not _fonts:
            # Las fuentes dejan de servir después de pygame.quit(), que además olvida
            # los ganchos registrados: se registra de nuevo cada vez que se llena el caché
            pygame.register_quit(_fonts.clear)
        _fonts[size] =pygame.font.SysFont("nunito,quicksand,segoeuisemibold,segoeui,arial",
                                           size, bold=True)
    return _fonts[size]


def poly(surf, fill, pts, k, ow=OUTLINE_W, outline=None):
    """Polígono relleno con contorno de esquinas redondeadas."""
    pygame.draw.polygon(surf, fill, pts)
    if ow:
        w = _w(k, ow)
        color = outline or PALETTE["outline"]
        pygame.draw.lines(surf, color, True, pts, w)
        if w > 2:
            for p in pts:
                pygame.draw.circle(surf, color, p, w / 2)


def circle(surf, fill, c, r, k, ow=OUTLINE_W):
    if ow:
        pygame.draw.circle(surf, PALETTE["outline"], c, r + _w(k, ow) / 2)
    pygame.draw.circle(surf, fill, c, max(1, r - _w(k, ow) / 2 if ow else r))


def ellipse(surf, fill, c, rx, ry, k, ow=OUTLINE_W):
    rect = pygame.Rect(0, 0, 2 * rx, 2 * ry)
    rect.center = c
    if ow:
        w = _w(k, ow)
        pygame.draw.ellipse(surf, PALETTE["outline"], rect.inflate(w, w))
        rect = rect.inflate(-w, -w)
    pygame.draw.ellipse(surf, fill, rect)


def rrect(surf, fill, rect, radius, k, ow=OUTLINE_W):
    rect = pygame.Rect(rect)
    if ow:
        w = _w(k, ow)
        pygame.draw.rect(surf, PALETTE["outline"], rect.inflate(w, w), border_radius=int(radius + w / 2))
        rect = rect.inflate(-w, -w)
    pygame.draw.rect(surf, fill, rect, border_radius=int(radius))


def text(surf, s, pos, size, color, outline=None, anchor="center", alpha=255):
    img = font(size).render(s, True, color)
    if outline:
        o = max(1, size // 9)
        base = font(size).render(s, True, outline)
        full = pygame.Surface((img.get_width() + 2 * o, img.get_height() + 2 * o), pygame.SRCALPHA)
        for dx in (-o, 0, o):
            for dy in (-o, 0, o):
                full.blit(base, (o + dx, o + dy))
        full.blit(img, (o, o))
        img = full
    if alpha < 255:
        img.set_alpha(alpha)
    rect = img.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
    surf.blit(img, rect)
    return rect


def layer(size):
    return pygame.Surface((max(1, int(size[0])), max(1, int(size[1]))), pygame.SRCALPHA)


def blit_rotated(surf, img, pos, pivot, angle, alpha=255):
    """Pega `img` rotada `angle` grados alrededor de `pivot` (coordenadas locales) en `pos`."""
    offset = pygame.math.Vector2(pivot) - pygame.math.Vector2(img.get_size()) / 2
    rotated = pygame.transform.rotate(img, angle) if angle else img
    center = pygame.math.Vector2(pos) - offset.rotate(-angle)
    if alpha < 255:
        rotated.set_alpha(alpha)
    surf.blit(rotated, rotated.get_rect(center=(round(center.x), round(center.y))))


def soft_circle(surf, color, c, r, alpha):
    if alpha <= 0 or r <= 0:
        return
    img = layer((2 * r + 2, 2 * r + 2))
    pygame.draw.circle(img, color, (r + 1, r + 1), r)
    img.set_alpha(int(alpha))
    surf.blit(img, (c[0] - r - 1, c[1] - r - 1))


def soft_ellipse(surf, color, c, rx, ry, alpha):
    img = layer((2 * rx + 2, 2 * ry + 2))
    pygame.draw.ellipse(img, color, img.get_rect())
    img.set_alpha(int(alpha))
    surf.blit(img, (c[0] - rx - 1, c[1] - ry - 1))


def particles(seed: int, n: int, t: float, origin, spread: float, rise: float):
    """Partículas que salen de `origin`, suben y se desvanecen; t en [0, 1].

    Devuelve [(x, y, alpha)]. Determinista: depende solo de los argumentos.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        dx = rng.uniform(-1, 1) * spread
        h = rng.uniform(0.5, 1.0) * rise
        delay = rng.uniform(0, 0.35)
        p = (t - delay) / (1 - delay)
        if p <= 0:
            out.append((origin[0], origin[1], 0))
            continue
        p = min(p, 1.0)
        e = 1 - (1 - p) ** 2
        out.append((origin[0] + dx * e, origin[1] - h * e, int(255 * (1 - p) ** 0.8)))
    return out


# ---------------------------------------------------------------- escena
def draw_block(surf, iso: Iso, x, y, height, top=None, grey=False):
    k = iso.k
    h = height
    left = PALETTE["grey_left" if grey else "counter_left"]
    right = PALETTE["grey_right" if grey else "counter_right"]
    top = top or PALETTE["grey_top" if grey else "counter_top"]
    p = iso.to_screen
    poly(surf, left, [p(x, y + 1, h), p(x + 1, y + 1, h), p(x + 1, y + 1, 0), p(x, y + 1, 0)], k)
    poly(surf, right, [p(x + 1, y, h), p(x + 1, y + 1, h), p(x + 1, y + 1, 0), p(x + 1, y, 0)], k)
    poly(surf, top, [p(x, y, h), p(x + 1, y, h), p(x + 1, y + 1, h), p(x, y + 1, h)], k)
    return p(x + 0.5, y + 0.5, h)


def draw_floor(surf, iso: Iso, size: int):
    p = iso.to_screen
    for x in range(size):
        for y in range(size):
            color = PALETTE["floor_b" if (x + y) % 2 else "floor_a"]
            pygame.draw.polygon(surf, color, [p(x, y), p(x + 1, y), p(x + 1, y + 1), p(x, y + 1)])


def blob_points(c, r):
    x, y = c
    pts = [(x + r * math.cos(a), y - 1.1 * r * math.sin(a))
           for a in (math.pi * i / 12 for i in range(13))]
    pts += [(x + r * math.cos(a), y + 0.3 * r * -math.sin(a))
            for a in (math.pi + math.pi * i / 8 for i in range(1, 8))]
    return pts


def draw_blob(surf, c, color, r, k=1.0):
    poly(surf, color, blob_points(c, r), k)
    soft_ellipse(surf, (255, 255, 255), (c[0] - 0.35 * r, c[1] - 0.7 * r), 0.28 * r, 0.17 * r, 150)


def draw_carrier(surf, c, kind, color, k=1.0):
    """Tabla cuadrada ("board") o plato redondo ("plate") con la comida encima."""
    x, y = c
    if kind == "board":
        poly(surf, PALETTE["board"], [(x - 13 * k, y), (x, y - 6.5 * k), (x + 13 * k, y),
                                      (x, y + 6.5 * k)], k)
    else:
        ellipse(surf, PALETTE["plate"], c, 11 * k, 5 * k, k)
    draw_blob(surf, (x, y), color, 5.5 * k, k)


def draw_chef(surf, pos, shirt, *, lean=0.0, wiggle=0.0, face="normal", hop=0.0, k=1.0):
    """Chef con cuerpo de cápsula, cabeza redonda y gorro. `pos` = punto de los pies."""
    soft_ellipse(surf, (0, 0, 0), (pos[0], pos[1] + 2 * k), 14 * k, 6 * k, 38)
    W, H = 64, 84
    img = layer((W * k, H * k))
    fx, fy = W / 2, H - 4                          # pies, en unidades locales

    def P(x, y):
        return ((fx + x) * k, (fy + y) * k)

    def R(x, y, w, h):
        return pygame.Rect(round((fx + x) * k), round((fy + y) * k), round(w * k), round(h * k))

    ol = PALETTE["outline"]
    w = _w(k)
    # Cuerpo (camisa) y delantal
    rrect(img, shirt, R(-10, -30, 20, 30), 10 * k, k)
    rrect(img, PALETTE["chef_white"], R(-6.5, -16, 13, 14), 4 * k, k, ow=1.1)
    # Cabeza
    circle(img, PALETTE["skin"], P(0, -38), 9 * k, k)
    for sx in (-1, 1):
        soft_circle(img, PALETTE["blush"], P(sx * 5.5, -35), 2.2 * k, 170)
    if face == "worried":
        for sx in (-1, 1):
            pygame.draw.circle(img, ol, P(sx * 3, -38), 1.3 * k)
            pygame.draw.line(img, ol, P(sx * 1.5, -42.5), P(sx * 5, -41), max(1, round(1.2 * k)))
        pygame.draw.ellipse(img, ol, R(-1.5, -35, 3, 2.6))
        pygame.draw.circle(img, (120, 190, 240), P(8.5, -44), 1.8 * k)
    elif face == "ponder":
        for sx in (-1, 1):
            pygame.draw.circle(img, ol, P(sx * 3 + 1, -39.5), 1.3 * k)
        pygame.draw.line(img, ol, P(0, -34.5), P(3, -35), max(1, round(1.1 * k)))
    else:
        for sx in (-1, 1):
            pygame.draw.circle(img, ol, P(sx * 3, -38), 1.3 * k)
        pygame.draw.arc(img, ol, R(-2.2, -37.2, 4.4, 3.2), math.pi * 1.1, math.pi * 1.9,
                        max(1, round(1.1 * k)))
    # Gorro: primero la silueta en color de contorno, luego el relleno, para que se vea como una sola pieza
    puffs = [(-5, -53, 5), (5, -53, 5), (0, -56, 5.5)]
    band = R(-8, -52, 16, 9)
    for x, y, r in puffs:
        pygame.draw.circle(img, ol, P(x, y), r * k + w / 2)
    pygame.draw.rect(img, ol, band.inflate(w, w), border_radius=int(2 * k))
    for x, y, r in puffs:
        pygame.draw.circle(img, PALETTE["chef_white"], P(x, y), r * k - w / 2)
    pygame.draw.rect(img, PALETTE["chef_white"], band.inflate(-w, -w), border_radius=int(2 * k))
    pygame.draw.line(img, (230, 226, 220), P(-6, -46), P(6, -46), max(1, round(k)))

    angle = -lean + 6 * math.sin(wiggle * math.tau * 2)
    shift = 2.5 * k * math.sin(wiggle * math.tau * 2)
    blit_rotated(surf, img, (pos[0] + shift, pos[1] - hop * k), (fx * k, fy * k), angle)


def draw_lid(surf, c, k, angle=0.0, alpha=255):
    img = layer((34 * k, 22 * k))
    cx, cy = 17 * k, 14 * k
    pygame.draw.ellipse(img, PALETTE["outline"], pygame.Rect(0, 0, 26 * k, 10 * k).move(4 * k, 9 * k - 5 * k).inflate(_w(k), _w(k)))
    ellipse(img, PALETTE["metal_light"], (cx, cy), 13 * k, 5 * k, k)
    ellipse(img, PALETTE["metal_light"], (cx, cy - 2 * k), 10 * k, 5 * k, k)
    pygame.draw.ellipse(img, PALETTE["metal_light"], pygame.Rect(0, 0, 18 * k, 4 * k).move(cx - 9 * k, cy - 2 * k))
    circle(img, PALETTE["outline"], (cx, cy - 7.5 * k), 2.6 * k, k, ow=0)
    soft_ellipse(img, (255, 255, 255), (cx - 5 * k, cy - 3.5 * k), 3 * k, 1.3 * k, 160)
    blit_rotated(surf, img, c, (cx, cy), angle, alpha)


def draw_burner(surf, iso: Iso, x, *, available, lid, flame, steam, revealed_burnt, t_anim):
    """Fuego en la fila del fondo. lid: 0 tapada ... 1 sin tapa (se dibuja aparte)."""
    k = iso.k
    cx, cy = draw_block(surf, iso, x, 0, COUNTER_HEIGHT, grey=not available)
    base = PALETTE["burner_base"] if available else (170, 160, 150)
    ellipse(surf, base, (cx, cy), 15 * k, 7.5 * k, k)
    if flame > 0 and available:
        for i in (-1, 0, 1):
            fl = flame * (1 + 0.25 * math.sin(t_anim * math.tau * 4 + i * 2.1))
            fx = cx + i * 7 * k
            poly(surf, PALETTE["flame"], [(fx - 3.2 * k, cy + 1.5 * k), (fx, cy + 1.5 * k - 9 * fl * k),
                                          (fx + 3.2 * k, cy + 1.5 * k)], k, ow=0.9,
                 outline=PALETTE["flame_edge"])
    metal = PALETTE["metal"] if available else (196, 192, 186)
    # Olla: fondo, cuerpo y borde
    ellipse(surf, metal, (cx, cy - 3 * k), 13 * k, 5.5 * k, k)
    pygame.draw.rect(surf, metal, pygame.Rect(cx - 13 * k + _w(k) / 2, cy - 10 * k, 26 * k - _w(k), 7 * k))
    pygame.draw.line(surf, PALETTE["outline"], (cx - 13 * k, cy - 10 * k), (cx - 13 * k, cy - 3 * k), _w(k))
    pygame.draw.line(surf, PALETTE["outline"], (cx + 13 * k, cy - 10 * k), (cx + 13 * k, cy - 3 * k), _w(k))
    rrect(surf, (107, 84, 70), pygame.Rect(cx + 12 * k, cy - 9 * k, 11 * k, 3.2 * k), 1.5 * k, k, ow=1)
    rim = (cx, cy - 10 * k)
    ellipse(surf, (60, 56, 56) if revealed_burnt or lid > 0 else metal, rim, 13 * k, 5.5 * k, k)
    if revealed_burnt:
        draw_blob(surf, (cx, cy - 9 * k), DISH_COLORS["burnt"], 7 * k, k)
        burning(surf, (cx, cy - 15 * k), k, t_anim, seed=x)
    if lid < 1:
        draw_lid(surf, (rim[0], rim[1] - lid * 14 * k), k, angle=lid * 25)
    if steam > 0 and available and not revealed_burnt:
        draw_steam(surf, (cx, cy - 18 * k - lid * 14 * k), k, t_anim, steam)
    return rim


def draw_steam(surf, c, k, t, strength=1.0, color=(255, 255, 255)):
    for j, dx in enumerate((-4, 4)):
        for i in range(5):
            p = ((t * 0.8 + i / 5 + j * 0.37) % 1.0)
            x = c[0] + dx * k + 2.5 * k * math.sin(p * math.tau + j)
            y = c[1] - p * 18 * k
            soft_circle(surf, color, (x, y), (1.6 + 1.6 * p) * k, 200 * strength * (1 - p))


def burning(surf, c, k, t, seed=0):
    """Brasas rojas y amarillas que suben, más humo gris."""
    cyc = t % 1.0
    for i, (x, y, a) in enumerate(particles(1000 + seed * 7 + int(t), 7, cyc, c, 9 * k, 20 * k)):
        col = PALETTE["ember_yellow"] if i % 2 else PALETTE["ember_red"]
        soft_circle(surf, col, (x, y), 1.7 * k, a)
    draw_steam(surf, (c[0], c[1] - 4 * k), k, t * 0.6, 0.8, PALETTE["smoke"])


def draw_trash(surf, iso: Iso, open_amount=0.0, bounce=0.0):
    k = iso.k
    x, y = STATION_TILES["trash"]
    cx, cy = draw_block(surf, iso, x, y, COUNTER_HEIGHT)
    by = -bounce * 4 * k
    rrect(surf, PALETTE["trash"], pygame.Rect(cx - 9 * k, cy - 16 * k + by, 18 * k, 17 * k), 3 * k, k)
    for dx in (-3.5, 0, 3.5):
        pygame.draw.line(surf, PALETTE["trash_line"], (cx + dx * k, cy - 12 * k + by),
                         (cx + dx * k, cy - 2.5 * k + by), max(1, round(1.5 * k)))
    # Tapa que gira alrededor de su borde izquierdo
    piv = pygame.math.Vector2(cx - 11 * k, cy - 17.5 * k + by)
    ang = -55 * open_amount
    corners = [(0, -2.5), (22, -2.5), (22, 2.5), (0, 2.5)]
    pts = [piv + pygame.math.Vector2(px * k, py * k).rotate(ang) for px, py in corners]
    poly(surf, PALETTE["trash_lid"], pts, k)
    knob = piv + pygame.math.Vector2(11 * k, -4 * k).rotate(ang)
    pygame.draw.circle(surf, PALETTE["outline"], knob, 2 * k)
    return (cx, cy - 18 * k)


# ----------------------------------------------------------------- íconos
def _face(img, c, r, color, k):
    circle(img, color, c, r, k)


def draw_icon(surf, name, pos, size, alpha=255):
    """Ícono vectorial centrado en `pos`, de diámetro aproximado `size`."""
    k = size / 24
    img = layer((size * 1.8, size * 1.8))
    c = (img.get_width() / 2, img.get_height() / 2)
    cx, cy = c
    r = size / 2
    ol = PALETTE["outline"]
    lw = max(1, round(1.6 * k))

    def eyes(dy=-2.5, spread=4.0):
        for sx in (-1, 1):
            pygame.draw.circle(img, ol, (cx + sx * spread * k, cy + dy * k), 1.6 * k)

    if name in ("confused", "yum", "angry", "puke"):
        color = {"confused": PALETTE["face_yellow"], "yum": PALETTE["face_yellow"],
                 "angry": PALETTE["face_red"], "puke": PALETTE["face_green"]}[name]
        _face(img, c, r, color, k)
        if name == "confused":
            pygame.draw.circle(img, ol, (cx - 4 * k, cy - 2 * k), 1.6 * k)
            pygame.draw.circle(img, ol, (cx + 4 * k, cy - 3.5 * k), 1.6 * k)
            pts = [(cx - 5 * k + i * 2.5 * k, cy + 5 * k + (1.3 * k if i % 2 else -1.3 * k)) for i in range(5)]
            pygame.draw.lines(img, ol, False, pts, lw)
        elif name == "yum":
            for sx in (-1, 1):
                pygame.draw.arc(img, ol, pygame.Rect(cx + sx * 4 * k - 2.5 * k, cy - 4.5 * k, 5 * k, 4 * k),
                                0, math.pi, lw)
            pygame.draw.arc(img, ol, pygame.Rect(cx - 5 * k, cy - 1 * k, 10 * k, 8 * k), math.pi, math.tau, lw)
            pygame.draw.circle(img, (240, 110, 130), (cx + 2.5 * k, cy + 6 * k), 2.2 * k)
        elif name == "angry":
            eyes(-1.5)
            for sx in (-1, 1):
                pygame.draw.line(img, ol, (cx + sx * 1.5 * k, cy - 3.5 * k), (cx + sx * 6.5 * k, cy - 6.5 * k), lw)
            pygame.draw.arc(img, ol, pygame.Rect(cx - 4.5 * k, cy + 3 * k, 9 * k, 6 * k), 0.15, math.pi - 0.15, lw)
        else:
            for sx in (-1, 1):
                ex, ey = cx + sx * 4 * k, cy - 3 * k
                pygame.draw.line(img, ol, (ex - 1.6 * k, ey - 1.6 * k), (ex + 1.6 * k, ey + 1.6 * k), lw)
                pygame.draw.line(img, ol, (ex - 1.6 * k, ey + 1.6 * k), (ex + 1.6 * k, ey - 1.6 * k), lw)
            ellipse(img, (90, 150, 60), (cx, cy + 4 * k), 3.5 * k, 2.5 * k, k, ow=1)
            draw_blob(img, (cx + 1 * k, cy + 10 * k), (120, 200, 70), 2.6 * k, k)
    elif name == "check":
        circle(img, PALETTE["check_green"], c, r, k)
        pygame.draw.lines(img, (255, 255, 255), False,
                          [(cx - 5 * k, cy), (cx - 1.5 * k, cy + 4 * k), (cx + 5.5 * k, cy - 4 * k)],
                          max(2, round(3 * k)))
    elif name == "warning":
        pts = [(cx, cy - r), (cx + r * 1.05, cy + r * 0.8), (cx - r * 1.05, cy + r * 0.8)]
        poly(img, PALETTE["face_yellow"], pts, k)
        pygame.draw.line(img, ol, (cx, cy - 4.5 * k), (cx, cy + 3 * k), max(2, round(2.6 * k)))
        pygame.draw.circle(img, ol, (cx, cy + 6.5 * k), 1.6 * k)
    elif name in ("dots", "question"):
        bw, bh = size * 1.3, size * 0.8
        box = pygame.Rect(0, 0, bw, bh)
        box.center = c
        poly(img, (255, 255, 255), [(cx - 3 * k, box.bottom - 2), (cx + 3 * k, box.bottom - 2),
                                     (cx - 1 * k, box.bottom + 5 * k)], k)
        rrect(img, (255, 255, 255), box, bh / 2, k)
        pygame.draw.rect(img, (255, 255, 255), pygame.Rect(cx - 2.5 * k, box.bottom - 3 * k, 5 * k, 3 * k))
        if name == "dots":
            for i in (-1, 0, 1):
                pygame.draw.circle(img, ol, (cx + i * 5.5 * k, cy), 1.9 * k)
        else:
            text(img, "?", (cx, cy + 0.5 * k), int(size * 0.75), ol)
    elif name == "clock":
        circle(img, (255, 255, 255), c, r, k)
        pygame.draw.line(img, ol, c, (cx, cy - r * 0.6), lw)
        pygame.draw.line(img, ol, c, (cx + r * 0.45, cy + r * 0.1), lw)
    elif name == "coin":
        circle(img, PALETTE["coin"], c, r, k, ow=1.2)
        pygame.draw.circle(img, (255, 228, 120), c, r * 0.65, max(1, round(k)))
        text(img, "$", (cx, cy + 0.5 * k), int(size * 0.7), PALETTE["coin_text"])
    elif name == "minus_money":
        text(img, "-$", c, int(size * 0.95), PALETTE["money_red"], outline=(255, 255, 255))
    elif name == "plus_money":
        text(img, "+$", c, int(size * 0.95), PALETTE["money_green"], outline=(255, 255, 255))
    img.set_alpha(int(alpha))
    surf.blit(img, img.get_rect(center=(round(pos[0]), round(pos[1]))))


# ------------------------------------------------------------- vista previa
def _preview(path="results/figures/sprites_preview.png"):
    pygame.init()
    k = 2.0
    surf = pygame.Surface((int(470 * k * 1.6), int(320 * k)))
    surf.fill(PALETTE["bg"])
    iso = Iso((300 * k, 78 * k), 50 * k, 25 * k)
    draw_floor(surf, iso, 5)
    draw_block(surf, iso, 0, 0, COUNTER_HEIGHT)
    draw_block(surf, iso, 1, 0, COUNTER_HEIGHT)
    draw_burner(surf, iso, 2, available=True, lid=0, flame=1, steam=1, revealed_burnt=False, t_anim=0.2)
    draw_burner(surf, iso, 3, available=True, lid=1, flame=1, steam=0, revealed_burnt=True, t_anim=0.4)
    draw_burner(surf, iso, 4, available=False, lid=0, flame=0, steam=0, revealed_burnt=False, t_anim=0)
    draw_block(surf, iso, 0, 1, COUNTER_HEIGHT)
    draw_trash(surf, iso, 0.6, 0)
    for i, face in enumerate(("normal", "worried", "ponder")):
        draw_chef(surf, iso.to_screen(1.5 + i, 2.5), SHIRTS[i], face=face, k=k)
    draw_carrier(surf, iso.to_screen(3.2, 3.8, 10), "plate", DISH_COLORS["steak"], k)
    draw_carrier(surf, iso.to_screen(4.0, 3.2, 10), "board", DISH_COLORS["stirfry"], k)
    for i, name in enumerate(ICONS):
        draw_icon(surf, name, (520 * k + (i % 4) * 60 * k, 60 * k + (i // 4) * 60 * k), 32 * k)
    pygame.image.save(surf, path)
    print(f"Vista previa en {path}")


if __name__ == "__main__":
    _preview()
