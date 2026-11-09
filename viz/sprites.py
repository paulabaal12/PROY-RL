"""Primitivas de dibujo del estilo Toybox: bloques, chef, comida, fuegos, íconos.

Todas las medidas están en unidades del mockup (lienzo de 470 x 320) y se
multiplican por `k`. Las funciones que reciben un `Iso` toman `k` de él.

Uso para revisar a ojo:
    python -m viz.sprites   # guarda results/figures/sprites_preview.png
"""

import math
import random

import pygame

from envs.kitchen_config import BURNT_FROM, RAW_BELOW
from viz.iso import COUNTER_HEIGHT, STATION_TILES, WALL_HEIGHT, Iso
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
    k = iso.k
    # Base de la maqueta
    poly(surf, PALETTE["slab_left"], [p(0, size), p(size, size), p(size, size, -9), p(0, size, -9)], k)
    poly(surf, PALETTE["slab_right"], [p(size, 0), p(size, size), p(size, size, -9), p(size, 0, -9)], k)
    for x in range(size):
        for y in range(size):
            color = PALETTE["floor_b" if (x + y) % 2 else "floor_a"]
            pygame.draw.polygon(surf, color, [p(x, y), p(x + 1, y), p(x + 1, y + 1), p(x, y + 1)])
    for i in range(size + 1):
        pygame.draw.line(surf, (226, 200, 166), p(i, 0), p(i, size), max(1, round(0.7 * k)))
        pygame.draw.line(surf, (226, 200, 166), p(0, i), p(size, i), max(1, round(0.7 * k)))
    # Sombra que proyectan los mostradores sobre el piso
    shade = (214, 186, 150)
    pygame.draw.polygon(surf, shade, [p(1, 1), p(size, 1), p(size, 1.32), p(1.32, 1.32), p(1.32, size), p(1, size)])


def _wall_tiles(surf, iso, plane, length, k):
    """Azulejos de la parte baja de una pared (`plane`: "right" sobre x, "left" sobre y)."""
    p = (lambda a, z: iso.to_screen(a, 0, z)) if plane == "right" else (lambda a, z: iso.to_screen(0, a, z))
    tile_h, tile_w = 7.5, 0.4
    rows = int(WALL_HEIGHT * 0.56 / tile_h)
    for row in range(rows):
        offset = (row % 2) * tile_w / 2
        a = -offset
        i = 0
        while a < length:
            a0, a1 = max(a, 0), min(a + tile_w, length)
            z0, z1 = row * tile_h, (row + 1) * tile_h
            color = PALETTE["tile_light" if (i + row) % 2 else "tile_shade"]
            pts = [p(a0, z1), p(a1, z1), p(a1, z0), p(a0, z0)]
            pygame.draw.polygon(surf, color, pts)
            pygame.draw.lines(surf, PALETTE["tile_line"], True, pts, max(1, round(0.5 * k)))
            a += tile_w
            i += 1


def _shelf(surf, iso, x0, x1, z, k):
    p = lambda a, h, d=0.0: iso.to_screen(a, d, h)
    poly(surf, PALETTE["shelf"], [p(x0, z), p(x1, z), p(x1, z, 0.22), p(x0, z, 0.22)], k, ow=1.2)
    poly(surf, (140, 98, 66), [p(x0, z, 0.22), p(x1, z, 0.22), p(x1, z - 2, 0.22), p(x0, z - 2, 0.22)], k, ow=1.2)
    jars = [(0.25, (239, 111, 94), 9), (0.62, (108, 193, 90), 11), (1.0, (255, 210, 63), 8), (1.36, (155, 127, 212), 10)]
    for offset, color, height in jars:
        if x0 + offset > x1 - 0.12:
            continue
        cx, cy = p(x0 + offset, z, 0.12)
        rect = pygame.Rect(cx - 4 * k, cy - height * k, 8 * k, height * k)
        rrect(surf, (240, 246, 248), rect, 2 * k, k, ow=1.1)
        pygame.draw.rect(surf, color, pygame.Rect(rect.x + 1.4 * k, rect.y + 0.42 * rect.h, rect.w - 2.8 * k, rect.h * 0.5),
                         border_radius=max(1, round(k)))
        rrect(surf, PALETTE["shelf"], pygame.Rect(cx - 4.6 * k, rect.y - 2.2 * k, 9.2 * k, 3 * k), 1 * k, k, ow=1)


def _window(surf, iso, y0, y1, z0, z1, k):
    p = lambda y, z: iso.to_screen(0, y, z)
    poly(surf, PALETTE["frame"], [p(y0 - 0.1, z1 + 2.6), p(y1 + 0.1, z1 + 2.6), p(y1 + 0.1, z0 - 2.6), p(y0 - 0.1, z0 - 2.6)], k, ow=1.3)
    steps = 6
    for i in range(steps):
        z_a, z_b = z1 - (z1 - z0) * i / steps, z1 - (z1 - z0) * (i + 1) / steps
        pygame.draw.polygon(surf, _mix(PALETTE["sky"], PALETTE["sky_low"], i / (steps - 1)),
                            [p(y0, z_a), p(y1, z_a), p(y1, z_b), p(y0, z_b)])
    for cy, cz, r in ((y0 + 0.45, z0 + 10, 2.4), (y0 + 0.6, z0 + 11, 3.0), (y0 + 0.8, z0 + 10, 2.3)):
        pygame.draw.circle(surf, (255, 255, 255), p(cy, cz), r * k)
    pygame.draw.lines(surf, PALETTE["outline"], True, [p(y0, z1), p(y1, z1), p(y1, z0), p(y0, z0)], _w(k, 1.1))
    middle = (y0 + y1) / 2
    pygame.draw.line(surf, PALETTE["frame"], p(middle, z0), p(middle, z1), max(2, round(2 * k)))
    pygame.draw.line(surf, PALETTE["frame"], p(y0, (z0 + z1) / 2), p(y1, (z0 + z1) / 2), max(2, round(2 * k)))


def _awning(surf, iso, y0, y1, z, k):
    p = lambda y, h, d=0.0: iso.to_screen(d, y, h)
    stripes = 6
    for i in range(stripes):
        a = y0 + (y1 - y0) * i / stripes
        b = y0 + (y1 - y0) * (i + 1) / stripes
        poly(surf, PALETTE["awning_a" if i % 2 == 0 else "awning_b"],
             [p(a, z + 11), p(b, z + 11), p(b, z, 0.34), p(a, z, 0.34)], k, ow=0.9)


def _hanging_tools(surf, iso, x0, x1, z, k):
    p = lambda x, h: iso.to_screen(x, 0, h)
    pygame.draw.line(surf, PALETTE["metal"], p(x0, z), p(x1, z), max(2, round(1.6 * k)))
    for i, x in enumerate(((x0 + 0.2), (x0 + x1) / 2, (x1 - 0.2))):
        cx, cy = p(x, z)
        pygame.draw.line(surf, PALETTE["outline"], (cx, cy), (cx, cy + 4 * k), _w(k, 1.0))
        if i == 1:
            rrect(surf, PALETTE["handle"], pygame.Rect(cx - 1.2 * k, cy + 4 * k, 2.4 * k, 9 * k), 1 * k, k, ow=0.8)
            circle(surf, PALETTE["pan"], (cx, cy + 16 * k), 5 * k, k, ow=1.1)
        else:
            rrect(surf, PALETTE["metal_light"], pygame.Rect(cx - 1 * k, cy + 4 * k, 2 * k, 10 * k), 1 * k, k, ow=0.8)
            ellipse(surf, PALETTE["metal_light"], (cx, cy + 16 * k), 2.8 * k, 3.6 * k, k, ow=1.0)


def draw_walls(surf, iso: Iso, size: int):
    """Las dos paredes del fondo, con azulejos, repisa, utensilios, ventana y toldo."""
    k = iso.k
    h = WALL_HEIGHT
    right = lambda x, z: iso.to_screen(x, 0, z)
    left = lambda y, z: iso.to_screen(0, y, z)
    poly(surf, PALETTE["wall_right"], [right(0, h), right(size, h), right(size, 0), right(0, 0)], k)
    poly(surf, PALETTE["wall_left"], [left(0, h), left(size, h), left(size, 0), left(0, 0)], k)
    _wall_tiles(surf, iso, "right", size, k)
    _wall_tiles(surf, iso, "left", size, k)
    z_band = WALL_HEIGHT * 0.56 - 0.3
    pygame.draw.line(surf, PALETTE["outline"], right(0, z_band), right(size, z_band), _w(k, 1.0))
    pygame.draw.line(surf, PALETTE["outline"], left(0, z_band), left(size, z_band), _w(k, 1.0))
    _shelf(surf, iso, 0.2, 1.8, 40, k)
    _hanging_tools(surf, iso, 2.25, 4.75, 46, k)
    _window(surf, iso, 1.1, 3.1, 31, 46, k)
    _awning(surf, iso, 3.75, 4.95, 36, k)
    pygame.draw.lines(surf, PALETTE["outline"], False, [right(size, h), right(0, h), left(size, h)], _w(k, 1.6))


def draw_bell(surf, c, k):
    """Campana de servicio sobre la barra de entrega."""
    x, y = c
    ellipse(surf, PALETTE["brass_dark"], (x, y), 6.2 * k, 2.6 * k, k, ow=1.1)
    dome = [(x - 5 * k, y - 0.6 * k), (x - 3.6 * k, y - 4.4 * k), (x, y - 6 * k), (x + 3.6 * k, y - 4.4 * k),
            (x + 5 * k, y - 0.6 * k)]
    poly(surf, PALETTE["brass"], dome, k, ow=1.1)
    circle(surf, PALETTE["brass_dark"], (x, y - 7 * k), 1.3 * k, k, ow=0.9)
    soft_ellipse(surf, (255, 255, 255), (x - 1.8 * k, y - 3.8 * k), 1.1 * k, 0.8 * k, 180)


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


def draw_carrier(surf, c, kind, color, k=1.0, food=None, cook=0.0):
    """Tabla ("board") o plato ("plate") con la comida encima.

    Con `food` (plato) dibuja la comida de verdad; si no, una forma simple de color `color`.
    """
    x, y = c
    if kind == "board":
        poly(surf, PALETTE["board"], [(x - 15 * k, y), (x, y - 7.5 * k), (x + 15 * k, y),
                                      (x, y + 7.5 * k)], k)
    else:
        ellipse(surf, (222, 216, 205), (x, y + 1.2 * k), 14 * k, 6.6 * k, k)
        ellipse(surf, PALETTE["plate"], c, 14 * k, 6.4 * k, k)
        ellipse(surf, (255, 253, 248), c, 10 * k, 4.4 * k, k, ow=0)
    if food:
        draw_food(surf, (x, y - 1.5 * k), food, cook, 9 * k, k)
    else:
        draw_blob(surf, (x, y), color, 5.5 * k, k)


# ---------------------------------------------------------------- comida
def _clamp(u: float) -> float:
    return max(0.0, min(1.0, u))


def _mix(a, b, u: float):
    u = _clamp(u)
    return tuple(round(p + (q - p) * u) for p, q in zip(a, b))


def _doneness(cook: float) -> tuple[float, float]:
    """(cuánto se ha cocido, cuánto se ha quemado), ambos en [0, 1]."""
    return _clamp(cook / RAW_BELOW), _clamp((cook - BURNT_FROM) / 0.25)


CHAR = (43, 36, 34)


def _food_steak(surf, c, s, k, done, burnt):
    x, y = c
    meat = _mix(_mix((222, 92, 108), (150, 84, 54), done), CHAR, burnt)
    side = _mix(_mix((176, 62, 78), (104, 56, 36), done), (24, 20, 19), burnt)
    fat = _mix(_mix((255, 226, 214), (232, 196, 150), done), (70, 58, 52), burnt)
    pts = [(x - 9.6 * s, y - 0.6 * s), (x - 7 * s, y - 5 * s), (x - 1 * s, y - 6 * s), (x + 6 * s, y - 5.2 * s),
           (x + 9.8 * s, y - 1.5 * s), (x + 8.6 * s, y + 3.4 * s), (x + 2 * s, y + 5.2 * s),
           (x - 6 * s, y + 4.4 * s)]
    poly(surf, side, [(px, py + 2.2 * s) for px, py in pts], k, ow=1.0)
    poly(surf, fat, pts, k, ow=1.0)
    inner = [(x + (px - x) * 0.84, y + (py - y) * 0.82) for px, py in pts]
    pygame.draw.polygon(surf, meat, inner)
    if done > 0.45 and burnt < 0.8:
        mark = _mix((96, 52, 34), (58, 34, 24), done)
        alpha = int(255 * _clamp((done - 0.45) / 0.35))
        for dx in (-4.6, -0.6, 3.4):
            start, end = (x + (dx - 2.2) * s, y + 3.2 * s), (x + (dx + 2.6) * s, y - 3.4 * s)
            color = _mix(meat, mark, alpha / 255)
            pygame.draw.line(surf, color, start, end, max(1, round(1.5 * s)))
    soft_ellipse(surf, (255, 255, 255), (x - 3.4 * s, y - 3.2 * s), 2.8 * s, 0.9 * s, 110 * (1 - burnt))


def _food_egg(surf, c, s, k, done, burnt):
    x, y = c
    white = _mix(_mix((221, 230, 236), (255, 255, 252), done), (92, 74, 62), burnt)
    edge = _mix(_mix((205, 214, 222), (232, 184, 112), done), CHAR, burnt)
    lobes = [(-4.6, -0.8, 5.0, 3.5), (3.8, -1.6, 5.6, 3.7), (-0.6, 2.3, 6.0, 3.3), (5.2, 1.8, 3.8, 2.8)]
    w = _w(k, 1.0)
    for dx, dy, rx, ry in lobes:
        rect = pygame.Rect(0, 0, 2 * rx * s + 2 * w, 2 * ry * s + 2 * w)
        rect.center = (x + dx * s, y + dy * s)
        pygame.draw.ellipse(surf, PALETTE["outline"], rect)
    for dx, dy, rx, ry in lobes:
        rect = pygame.Rect(0, 0, 2 * rx * s, 2 * ry * s)
        rect.center = (x + dx * s, y + dy * s)
        pygame.draw.ellipse(surf, edge, rect)
    for dx, dy, rx, ry in lobes:
        rect = pygame.Rect(0, 0, 1.7 * rx * s, 1.65 * ry * s)
        rect.center = (x + dx * s, y + dy * s)
        pygame.draw.ellipse(surf, white, rect)
    yolk = _mix(_mix((255, 196, 48), (250, 166, 28), done), (58, 40, 30), burnt)
    yc = (x, y - 0.6 * s)
    ellipse(surf, yolk, yc, 3.9 * s, 2.9 * s, k, ow=1.0)
    soft_ellipse(surf, (255, 255, 255), (yc[0] - 1.3 * s, yc[1] - 1.0 * s), 1.2 * s, 0.7 * s, 200 * (1 - burnt))


# Trozos del salteado: (dx, dy, tipo)
_STIRFRY = [(-5.6, -2.0, "pepper"), (0.6, -3.6, "broccoli"), (5.8, -1.6, "carrot"), (-2.4, 0.6, "carrot"),
            (3.4, 1.4, "pepper"), (-6.4, 2.4, "broccoli"), (0.2, 3.6, "pepper"), (6.2, 3.0, "broccoli")]


def _food_stirfry(surf, c, s, k, done, burnt):
    x, y = c

    def tint(color):
        return _mix(_mix(color, tuple(int(v * 0.78) for v in color), done), CHAR, burnt)

    ellipse(surf, tint((226, 190, 120)), (x, y + 0.6 * s), 9.8 * s, 6.2 * s, k, ow=1.0)
    for dx, dy, kind in sorted(_STIRFRY, key=lambda p: p[1]):
        px, py = x + dx * s, y + dy * s
        if kind == "broccoli":
            pygame.draw.line(surf, tint((168, 214, 120)), (px, py), (px, py + 2.2 * s), max(1, round(1.8 * s)))
            green = tint((88, 178, 76))
            for ox, oy, r in ((-1.5, -0.6, 1.6), (1.5, -0.6, 1.6), (0, -1.9, 1.8)):
                circle(surf, green, (px + ox * s, py + oy * s), r * s, k, ow=0.7)
            soft_circle(surf, (255, 255, 255), (px - 0.6 * s, py - 2.4 * s), 0.6 * s, 150 * (1 - burnt))
        elif kind == "carrot":
            ellipse(surf, tint((244, 142, 52)), (px, py), 2.6 * s, 1.8 * s, k, ow=0.8)
            ellipse(surf, tint((252, 186, 104)), (px, py), 1.2 * s, 0.8 * s, k, ow=0)
        else:
            a, b = (px - 2.4 * s, py + 0.8 * s), (px + 2.4 * s, py - 0.8 * s)
            width = max(2, round(2.2 * s))
            pygame.draw.line(surf, PALETTE["outline"], a, b, width + 2 * _w(k, 0.7))
            for end in (a, b):
                pygame.draw.circle(surf, PALETTE["outline"], end, width / 2 + _w(k, 0.7))
            red = tint((232, 74, 66))
            pygame.draw.line(surf, red, a, b, width)
            for end in (a, b):
                pygame.draw.circle(surf, red, end, width / 2)


_FOOD = {"stirfry": _food_stirfry, "egg": _food_egg, "steak": _food_steak}


def draw_food(surf, c, dish, cook, r, k=1.0):
    """Dibuja el plato en perspectiva. `cook` es el nivel de cocción: cambia con él."""
    done, burnt = _doneness(cook)
    _FOOD[dish](surf, c, r / 10, k, done, burnt)


def draw_gauge(surf, c, cook, visual, k):
    """Barra de cocción: triángulo lleno = nivel real; hueco = lo que ve el agente."""
    width, height, top = 23 * k, 4.4 * k, 1.6       # la escala llega a `top` niveles de cocción
    rect = pygame.Rect(0, 0, width, height)
    rect.center = c

    def at(level):
        return rect.x + rect.width * _clamp(level / top)

    pygame.draw.rect(surf, PALETTE["gauge_raw"], rect)
    pygame.draw.rect(surf, PALETTE["gauge_ok"], pygame.Rect(at(RAW_BELOW), rect.y, at(BURNT_FROM) - at(RAW_BELOW), height))
    pygame.draw.rect(surf, PALETTE["gauge_burnt"], pygame.Rect(at(BURNT_FROM), rect.y, rect.right - at(BURNT_FROM), height))
    pygame.draw.rect(surf, PALETTE["outline"], rect.inflate(_w(k, 1.0), _w(k, 1.0)), _w(k, 1.0),
                     border_radius=max(1, round(1.2 * k)))
    if visual is not None:
        x, y = at(visual), rect.bottom + 1 * k
        pygame.draw.polygon(surf, PALETTE["hud_bg"], [(x, y), (x - 2.6 * k, y + 4.2 * k), (x + 2.6 * k, y + 4.2 * k)])
        pygame.draw.polygon(surf, PALETTE["outline"], [(x, y), (x - 2.6 * k, y + 4.2 * k), (x + 2.6 * k, y + 4.2 * k)],
                            _w(k, 0.9))
    if cook is not None:
        x, y = at(cook), rect.top - 1 * k
        pygame.draw.polygon(surf, PALETTE["outline"], [(x, y), (x - 3 * k, y - 4.6 * k), (x + 3 * k, y - 4.6 * k)])


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
    # Brazos, por detrás del cuerpo
    for sx in (-1, 1):
        rrect(img, shirt, R(sx * 11 - 3.2, -27, 6.4, 15), 3 * k, k, ow=1.2)
        circle(img, PALETTE["skin"], P(sx * 11, -11), 3.3 * k, k, ow=1.1)
    # Cuerpo (camisa) y delantal
    rrect(img, shirt, R(-10, -30, 20, 30), 10 * k, k)
    rrect(img, PALETTE["chef_white"], R(-6.5, -16, 13, 14), 4 * k, k, ow=1.1)
    pygame.draw.line(img, ol, P(-6, -16), P(-3.5, -27), max(1, round(k)))
    pygame.draw.line(img, ol, P(6, -16), P(3.5, -27), max(1, round(k)))
    # Pañuelo al cuello
    rrect(img, PALETTE["chef_white"], R(-6, -31, 12, 4), 2 * k, k, ow=1.0)
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
    """Tapa de vidrio: deja ver la comida."""
    img = layer((40 * k, 26 * k))
    cx, cy = 20 * k, 17 * k
    glass = pygame.Surface(img.get_size(), pygame.SRCALPHA)
    pygame.draw.ellipse(glass, (*PALETTE["glass"], 92), pygame.Rect(cx - 14 * k, cy - 6 * k, 28 * k, 12 * k))
    pygame.draw.ellipse(glass, (*PALETTE["glass"], 92), pygame.Rect(cx - 11.5 * k, cy - 9 * k, 23 * k, 12 * k))
    img.blit(glass, (0, 0))
    pygame.draw.arc(img, PALETTE["outline"], pygame.Rect(cx - 12 * k, cy - 9.5 * k, 24 * k, 13 * k), 0.05, math.pi - 0.05,
                    max(1, _w(k, 1.2)))
    pygame.draw.ellipse(img, PALETTE["outline"], pygame.Rect(cx - 14 * k, cy - 6 * k, 28 * k, 12 * k), max(1, _w(k, 1.2)))
    soft_ellipse(img, (255, 255, 255), (cx - 6 * k, cy - 4.8 * k), 3.6 * k, 1.1 * k, 200)
    circle(img, PALETTE["handle"], (cx, cy - 9.6 * k), 2.6 * k, k, ow=1.2)
    blit_rotated(surf, img, c, (cx, cy), angle, alpha)


def draw_burner(surf, iso: Iso, x, *, available, lid, flame, steam, revealed_burnt, t_anim,
                dish=None, cook=0.0):
    """Fuego con sartén en la fila del fondo. lid: 0 tapada ... 1 sin tapa."""
    k = iso.k
    cx, cy = draw_block(surf, iso, x, 0, COUNTER_HEIGHT, grey=not available)
    base = PALETTE["burner_base"] if available else (170, 160, 150)
    ellipse(surf, base, (cx, cy), 15.5 * k, 7.6 * k, k)
    if available:
        ellipse(surf, (112, 104, 102), (cx, cy), 11 * k, 5.2 * k, k, ow=0.8)
    if flame > 0 and available:
        for i in (-1, 0, 1):
            fl = flame * (1 + 0.25 * math.sin(t_anim * math.tau * 4 + i * 2.1))
            fx = cx + i * 8 * k
            poly(surf, PALETTE["flame"], [(fx - 3.2 * k, cy + 3.6 * k), (fx, cy + 3.6 * k - 8 * fl * k),
                                          (fx + 3.2 * k, cy + 3.6 * k)], k, ow=0.9,
                 outline=PALETTE["flame_edge"])
            poly(surf, PALETTE["ember_yellow"], [(fx - 1.5 * k, cy + 3.2 * k), (fx, cy + 3.2 * k - 4 * fl * k),
                                                 (fx + 1.5 * k, cy + 3.2 * k)], k, ow=0)
    pan = PALETTE["pan"] if available else (196, 192, 186)
    inner = PALETTE["pan_inner"] if available else (214, 208, 200)
    # Sartén: cuerpo bajo, mango y fondo
    rrect(surf, PALETTE["handle"], pygame.Rect(cx + 12 * k, cy - 6.2 * k, 12 * k, 3.4 * k), 1.5 * k, k, ow=1)
    ellipse(surf, pan, (cx, cy - 1.5 * k), 13.5 * k, 6.2 * k, k)
    pygame.draw.rect(surf, pan, pygame.Rect(cx - 13.5 * k + _w(k) / 2, cy - 5 * k, 27 * k - _w(k), 3.5 * k))
    pygame.draw.line(surf, PALETTE["outline"], (cx - 13.5 * k, cy - 5 * k), (cx - 13.5 * k, cy - 1.5 * k), _w(k))
    pygame.draw.line(surf, PALETTE["outline"], (cx + 13.5 * k, cy - 5 * k), (cx + 13.5 * k, cy - 1.5 * k), _w(k))
    rim = (cx, cy - 5 * k)
    ellipse(surf, pan, rim, 13.5 * k, 6.2 * k, k)
    ellipse(surf, inner, rim, 11.2 * k, 4.9 * k, k, ow=0)
    if dish:
        wobble = 0.5 * k * math.sin(t_anim * math.tau * 2 + x) if available else 0.0
        draw_food(surf, (rim[0], rim[1] - 0.8 * k + wobble), dish, cook, 10 * k, k)
    if revealed_burnt:
        burning(surf, (cx, cy - 11 * k), k, t_anim, seed=x)
    if lid < 1:
        draw_lid(surf, (rim[0], rim[1] - 0.5 * k - lid * 15 * k), k, angle=lid * 25)
    if steam > 0 and available and not revealed_burnt:
        draw_steam(surf, (cx, cy - 15 * k - lid * 14 * k), k, t_anim, steam)
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
    iso = Iso((300 * k, 92 * k), 50 * k, 25 * k)
    draw_floor(surf, iso, 5)
    draw_walls(surf, iso, 5)
    draw_block(surf, iso, 0, 0, COUNTER_HEIGHT)
    draw_block(surf, iso, 1, 0, COUNTER_HEIGHT)
    draw_burner(surf, iso, 2, available=True, lid=0, flame=1, steam=1, revealed_burnt=False, t_anim=0.2,
                dish="steak", cook=1.15)
    draw_burner(surf, iso, 3, available=True, lid=1, flame=1, steam=0, revealed_burnt=True, t_anim=0.4,
                dish="egg", cook=1.5)
    draw_burner(surf, iso, 4, available=False, lid=0, flame=0, steam=0, revealed_burnt=False, t_anim=0)
    draw_block(surf, iso, 0, 1, COUNTER_HEIGHT)
    draw_trash(surf, iso, 0.6, 0)
    for i, face in enumerate(("normal", "worried", "ponder")):
        draw_chef(surf, iso.to_screen(1.5 + i, 2.5), SHIRTS[i], face=face, k=k)
    draw_carrier(surf, iso.to_screen(3.2, 3.8, 10), "plate", DISH_COLORS["steak"], k, food="steak", cook=1.15)
    draw_carrier(surf, iso.to_screen(4.0, 3.2, 10), "board", DISH_COLORS["stirfry"], k, food="stirfry")
    for i, name in enumerate(ICONS):
        draw_icon(surf, name, (520 * k + (i % 4) * 60 * k, 60 * k + (i // 4) * 60 * k), 32 * k)
    # Cada plato crudo, en su punto y quemado
    for row, dish in enumerate(("stirfry", "egg", "steak")):
        for col, cook in enumerate((0.0, 0.6, 1.15, 1.55)):
            draw_food(surf, ((30 + 44 * col) * k, (40 + 44 * row) * k), dish, cook, 18 * k, k)
    pygame.image.save(surf, path)
    print(f"Vista previa en {path}")


if __name__ == "__main__":
    _preview()
