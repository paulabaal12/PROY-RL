"""Proyección isométrica y ubicación de las estaciones en la cocina."""

from dataclasses import dataclass

# Cocina de 5 x 5 baldosas. Fila del fondo (y = 0): preparación y los 3 fuegos.
# Columna izquierda (x = 0): basurero y ventana de entrega. El resto son mostradores.
FLOOR = 5
STATION_TILES = {
    "prep": (1, 0),
    "burner0": (2, 0),
    "burner1": (3, 0),
    "burner2": (4, 0),
    "trash": (0, 2),
    "serve": (0, 4),
    "counter": [(0, 0), (0, 1), (0, 3)],
}
# Punto del piso donde se para el chef para usar cada estación
STAND_TILES = {
    "prep": (1.5, 1.6),
    "burner0": (2.5, 1.6),
    "burner1": (3.5, 1.6),
    "burner2": (4.5, 1.6),
    "trash": (1.6, 2.5),
    "serve": (1.6, 4.4),
    "center": (2.6, 2.7),
}
COUNTER_HEIGHT = 18
WALL_HEIGHT = 50
# Origen de la cocina en el lienzo del mockup (470 x 320), a la derecha de la columna de órdenes
KITCHEN_ORIGIN = (300, 96)
TILE = (60, 30)


@dataclass(frozen=True)
class Layout:
    """Dónde va cada cosa, en unidades de un lienzo de `design`."""

    design: tuple[int, int]
    origin: tuple[int, int]     # esquina del fondo de la cocina
    tile: tuple[int, int]
    cards_x: int                # columna de tarjetas de órdenes
    cards_top: int
    cards_bottom: int
    columns: int                # columnas de tarjetas
    column_step: int


# Panel ancho: órdenes a la izquierda y cocina a la derecha
LANDSCAPE = Layout((470, 320), KITCHEN_ORIGIN, TILE, 22, 40, 286, 1, 0)
# Panel angosto (pantalla dividida): órdenes arriba y una cocina más grande debajo
PORTRAIT = Layout((470, 560), (235, 258), (90, 45), 23, 36, 158, 3, 150)


def layout_for(rect) -> Layout:
    return PORTRAIT if rect.width / rect.height < 1.15 else LANDSCAPE


def design_frame(rect, design=(470, 320)) -> tuple[float, float, float]:
    """Escala k y esquina (x0, y0) para dibujar el lienzo del mockup centrado en `rect`."""
    k = min(rect.width / design[0], rect.height / design[1])
    return k, rect.x + (rect.width - design[0] * k) / 2, rect.y + (rect.height - design[1] * k) / 2


class Iso:
    def __init__(self, origin: tuple[float, float], tile_w: float, tile_h: float):
        self.ox, self.oy = origin
        self.tile_w = tile_w
        self.tile_h = tile_h

    @property
    def k(self) -> float:
        """Escala respecto al mockup (baldosa de 50 px)."""
        return self.tile_w / 50

    def to_screen(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        return (self.ox + (x - y) * self.tile_w / 2,
                self.oy + (x + y) * self.tile_h / 2 - z * self.k)
