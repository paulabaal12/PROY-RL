"""Paleta y constantes del estilo "Toybox" (ver el mockup aprobado en el spec)."""


def _hex(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


PALETTE = {name: _hex(h) for name, h in {
    "bg": "#fdf3e4",
    "floor_a": "#f6dfc0",
    "floor_b": "#efd2ae",
    "counter_top": "#e9b98a",
    "counter_left": "#c98f62",
    "counter_right": "#b07a52",
    "outline": "#5b3b28",
    "serve": "#6cb7e8",
    "metal": "#9aa3ad",
    "metal_light": "#b8c0c8",
    "burner_base": "#4a4040",
    "flame": "#ff8a3a",
    "flame_edge": "#e0531f",
    "board": "#d9a46c",
    "plate": "#f4f1ea",
    "skin": "#f2c49b",
    "blush": "#f4a08c",
    "chef_white": "#ffffff",
    "trash": "#8d98a3",
    "trash_lid": "#a7b1bb",
    "trash_line": "#6b747d",
    "grey_top": "#d8cfc4",
    "grey_left": "#b9afa4",
    "grey_right": "#a59b90",
    "hud_bg": "#fffdf9",
    "hud_fg": "#5b3b28",
    "bar_ok": "#7cc56a",
    "bar_warn": "#f2a03d",
    "bar_late": "#ef6f5e",
    "orange_flash": "#ff9a2e",
    "red_flash": "#ff3b3b",
    "coin": "#ffcf3a",
    "coin_text": "#a26b00",
    "money_green": "#3f9b3a",
    "money_red": "#d8323a",
    "ember_red": "#ff4a2e",
    "ember_yellow": "#ffd23f",
    "smoke": "#6a6060",
    "face_yellow": "#ffd75e",
    "face_red": "#ff7a5c",
    "face_green": "#9bd46a",
    "check_green": "#5fbf4f",
    "wall_right": "#f7e6cf",
    "wall_left": "#efd8bb",
    "tile_light": "#fbf4e8",
    "tile_shade": "#ece0cd",
    "tile_line": "#dcc9ae",
    "slab_left": "#c7a27b",
    "slab_right": "#b08a64",
    "shelf": "#a97850",
    "sky": "#bfe3f5",
    "sky_low": "#e8f4ee",
    "frame": "#f6ead8",
    "awning_a": "#ef6f5e",
    "awning_b": "#fff6ea",
    "brass": "#e8bd4f",
    "brass_dark": "#b98a24",
    "glass": "#dcecf2",
    "pan": "#55504f",
    "pan_inner": "#3a3635",
    "handle": "#6b5446",
    "gauge_raw": "#f3e3c6",
    "gauge_ok": "#7cc56a",
    "gauge_burnt": "#4a3a34",
}.items()}

DISH_COLORS = {
    "stirfry": _hex("#6cc15a"),
    "egg": _hex("#ffd23f"),
    "steak": _hex("#e8707a"),
    "burnt": _hex("#2b2422"),
}
DISH_NAMES = {"stirfry": "Stir-fry", "egg": "Egg", "steak": "Steak"}

# Color de camisa del chef de cada panel (también el del título y el puntero de orden)
SHIRTS = [_hex(h) for h in ("#ef6f5e", "#4bb3a8", "#9b7fd4", "#e0b33a")]

# Fracciones de un paso: caminar, acción, asentarse
PHASES = (0.40, 0.45, 0.15)

# Los paneles se dibujan a este múltiplo de su tamaño y luego se reducen (antialiasing)
SUPERSAMPLE = 2

# El mockup se diseñó en un lienzo de 470 x 320; todo se escala desde ahí
DESIGN_SIZE = (470, 320)
